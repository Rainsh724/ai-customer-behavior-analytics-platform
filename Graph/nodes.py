## PATH: app/graph/nodes.py
from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

from .state import GraphState
from .llm_client import call_llm_with_tools
from .tools import TOOL_DEFINITIONS, execute_tool_call
from .audit import validate_answer, correct_answer as _real_correct_answer

logger = logging.getLogger(__name__)


# ============================================================
# SAFETY WRAPPER
# ============================================================

def safe_node(node_name: str) -> Callable:
    def decorator(fn: Callable[[GraphState], dict[str, Any]]) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: GraphState) -> dict[str, Any]:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001 - intentional catch-all at node boundary
                logger.exception("Node '%s' failed", node_name)
                return {"errors": [f"{node_name}: {exc}"]}
        return wrapper
    return decorator


# ============================================================
# سقف تعداد دور Agent<->Tools -- جلوگیری از حلقه‌ی بی‌نهایت (مثلاً اگه
# SQL هی خطا بده و LLM هی دوباره تلاش کنه). طبق سند معماری، حلقه‌ی
# self-correction باید وجود داشته باشه ولی نامحدود نباشه.
# ============================================================

MAX_ITERATIONS = 6

# اگه در تمام ابزارهای یک دور -- حتی اگه چندتا موازی صدا زده شده باشن --
# همه‌شون خطا برگردونن، این شمارنده +۱ می‌شه؛ به‌محض رسیدن به این سقف،
# قبل از رسیدن به MAX_ITERATIONS هم گراف به finalize می‌ره. محافظت
# زودتر و مستقل از سقف کلی دور -- مخصوص حالتی که ابزار (مثلاً SQL) مدام
# شکست می‌خوره ولی Agent هنوز دوباره امتحان می‌کنه.
MAX_CONSECUTIVE_TOOL_ERRORS = 3


# ============================================================
# نود AGENT -- «مغز» سیستم
# ============================================================
# دقیقاً طبق سند: نود جدای «نتیجه‌گیری» نداریم؛ همین یک نود هم انتخاب
# ابزار، هم تفسیر نتایج خام، هم تولید جواب نهایی رو انجام می‌ده. تشخیص
# می‌ده کِی از حافظه‌ی مکالمه (state["messages"]) به‌جای صدا زدن ابزار
# جدید استفاده کنه (سناریوی "چرا؟" که ادامه‌ی سوال آماری قبلیه).

@safe_node("agent")
def agent_node(state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)

    response = call_llm_with_tools(messages, TOOL_DEFINITIONS)

    return {
        "messages": [response],
        "iterations": iterations + 1,
    }


# ============================================================
# نود TOOLS -- اجرای یک یا چند ابزار که Agent در همین دور خواسته
# ============================================================
# اگه پیام آخرِ agent چند tool_call همزمان داشته باشه (مثلاً هم tool_sql
# هم tool_rag -- سناریوی «اجرای موازی» سند)، همه‌شون همین‌جا، در همین
# اجرای نود، پشت‌سرهم اجرا می‌شن و همه‌شون به‌عنوان پیام‌های "tool" جدا
# برمی‌گردن. (اجرای واقعاً هم‌زمان/async بحث جدایی‌ست؛ چیزی که این‌جا
# تضمین می‌شه اینه که هر دو ابزار در همون یک دور -- بدون رفت‌وبرگشت اضافه
# به Agent -- اجرا و جواب داده می‌شن.)

@safe_node("tools")
def tools_node(state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return {"errors": ["tools: پیام‌ای در state نبود"]}

    last_message = messages[-1]
    tool_calls = last_message.get("tool_calls") or []

    if not tool_calls:
        # حالت غیرمنتظره: route_after_agent فقط باید وقتی به اینجا بیاد
        # که tool_calls موجود باشه. برای اطمینان، یک پیام خطا برمی‌گردونیم
        # به‌جای crash.
        return {"errors": ["tools: پیام آخر هیچ tool_call ای نداشت"]}

    tool_messages: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    all_errored = True

    for call in tool_calls:
        call_id = call.get("id", "")
        fn = call.get("function", {})
        name = fn.get("name", "")

        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
            logger.warning("tools: آرگومان‌های نامعتبر JSON برای ابزار '%s'", name)

        result = execute_tool_call(name, arguments)
        ok = "error" not in result
        all_errored = all_errored and not ok

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }
        )

        # خلاصه‌ی کوتاه برای audit.py -- کل result رو نگه نمی‌داریم چون
        # ممکنه شامل ده‌ها ردیف SQL باشه؛ فقط یک نمای کلی کافیه.
        tool_trace.append(
            {
                "tool": name,
                "arguments": arguments,
                "ok": ok,
                "summary": (result.get("error") if not ok else str(result)[:300]),
            }
        )

    # اگه هیچ ابزاری این دور اجرا نشده بود (tool_calls خالی بود -- که طبق
    # چک بالاتر نباید برسه اینجا)، all_errored رو مصنوعی True نکن.
    consecutive_errors = state.get("consecutive_tool_errors", 0)
    consecutive_errors = consecutive_errors + 1 if (tool_calls and all_errored) else 0

    return {
        "messages": tool_messages,
        "tool_trace": tool_trace,
        "consecutive_tool_errors": consecutive_errors,
    }


# ============================================================
# نود FINALIZE -- استخراج جواب نهایی
# ============================================================
# دو حالت:
#   1. حالت عادی: پیام آخرِ agent دیگه tool_call نداره -> همون content
#      متنی، جواب نهاییه.
#   2. حالت سقف iterations: هنوز tool_call می‌خواد ولی اجازه نداریم دوباره
#      بریم سراغ tools -> یک تماس آخر با tool_choice="none" می‌زنیم تا
#      LLM مجبور به جمع‌بندی متنی بشه (به‌جای این‌که با دست‌خالی برگردیم).

@safe_node("finalize")
def finalize_node(state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}

    if not last_message.get("tool_calls"):
        return {"final_answer": last_message.get("content") or ""}

    # به اینجا فقط از دو مسیر می‌رسیم (نگاه کن به graph.py::route_after_agent):
    #   1. سقف MAX_ITERATIONS رد شده
    #   2. MAX_CONSECUTIVE_TOOL_ERRORS رد شده (ابزار مدام شکست می‌خوره)
    # هر دو یعنی هنوز جواب متنی نداریم؛ باید مشخص کنیم کدوم بوده تا در
    # errors واضح ثبت بشه (برای دیباگ/ممیزی).
    if state.get("consecutive_tool_errors", 0) >= MAX_CONSECUTIVE_TOOL_ERRORS:
        reason = f"{MAX_CONSECUTIVE_TOOL_ERRORS} خطای متوالی ابزار"
    else:
        reason = f"سقف {MAX_ITERATIONS} دور Agent<->Tools"

    logger.warning("finalize: %s رد شد بدون جواب نهایی -- تماس اجباری برای جمع‌بندی", reason)
    forced = call_llm_with_tools(messages, TOOL_DEFINITIONS, tool_choice="none")
    content = forced.get("content") or "متاسفانه در تعداد تلاش مجاز نتونستم به پاسخ قطعی برسم."

    return {
        "messages": [forced],
        "final_answer": content,
        "errors": [f"finalize: پاسخ اجباری به دلیل {reason}"],
    }


# ============================================================
# نود VALIDATE -- بررسی نرم و مستقل (نگاه کن به audit.py)
# ============================================================
# جواب کاربر از finalize قبلاً نهایی شده؛ این نود فقط ممیزی می‌کنه و
# نمره می‌ده (match_score). خودِ این نود دیگه final_answer رو دستکاری
# نمی‌کنه -- تصمیم "آیا لازمه اصلاح بشه یا نه" در graph.py::
# route_after_validate گرفته می‌شه، بر اساس همین match_score.

@safe_node("validate")
def validate_node(state: GraphState) -> dict[str, Any]:
    final_answer = state.get("final_answer", "")
    tool_trace = state.get("tool_trace", [])
    validation = validate_answer(final_answer, tool_trace)

    warnings = validation.get("warnings") or []
    extra_errors = [f"validate: {w}" for w in warnings] if warnings else []

    return {
        "validation": validation,
        **({"errors": extra_errors} if extra_errors else {}),
    }


def _extract_last_user_question(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


# ============================================================
# نود CORRECT_ANSWER -- اصلاح یک‌باره (نگاه کن به audit.py)
# ============================================================
# فقط وقتی به اینجا می‌رسیم که graph.py::route_after_validate تشخیص داده
# match_score زیر آستانه (پیش‌فرض ۷۰) بوده. این نود مستقیم به END می‌ره
# (نگاه کن به graph.py) -- هرگز به validate یا agent برنمی‌گرده، پس
# امکان لوپ اصلاح/تغییر وجود نداره: حداکثر یک بار جواب بازنویسی می‌شه.

@safe_node("correct_answer")
def correct_answer_node(state: GraphState) -> dict[str, Any]:
    validation = state.get("validation", {})
    warnings = validation.get("warnings") or []
    question = _extract_last_user_question(state.get("messages", []))

    corrected = _real_correct_answer(
        question=question,
        final_answer=state.get("final_answer", ""),
        warnings=warnings,
        tool_trace=state.get("tool_trace", []),
    )

    return {
        "final_answer": corrected,
        "errors": [f"correct_answer: جواب یک‌بار اصلاح شد (match_score={validation.get('match_score')} زیر آستانه)"],
    }

