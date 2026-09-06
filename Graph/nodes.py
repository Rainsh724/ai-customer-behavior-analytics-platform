## PATH: app/graph/nodes.py
from __future__ import annotations
import os
import functools
import json
import logging
from os import name
from typing import Any, Callable

from .state import GraphState
from .llm_client import call_llm_with_tools
from .tools import TOOL_DEFINITIONS, execute_tool_call
from .audit import validate_answer, correct_answer as _real_correct_answer
from .dataset_time import get_reference_date

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

# ============================================================
# نود AGENT -- «مغز» سیستم
# ============================================================
# دقیقاً طبق سند: نود جدای «نتیجه‌گیری» نداریم؛ همین یک نود هم انتخاب
# ابزار، هم تفسیر نتایج خام، هم تولید جواب نهایی رو انجام می‌ده. تشخیص
# می‌ده کِی از حافظه‌ی مکالمه (state["messages"]) به‌جای صدا زدن ابزار
# جدید استفاده کنه (سناریوی "چرا؟" که ادامه‌ی سوال آماری قبلیه).

def _compact_message_for_llm(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")

    if not isinstance(content, str):
        content = str(content)

    MAX_CHARS = 3500

    compacted: dict[str, Any] = dict(message)
    compacted["content"] = (
        content[:MAX_CHARS] + "\n...[truncated for LLM]"
        if len(content) > MAX_CHARS
        else content
    )

    # ---------------------------------------------------------
    # مهم: متن SQL که خودِ Agent می‌نویسه داخل content نیست، داخل
    # message["tool_calls"][i]["function"]["arguments"] است (رشته‌ی
    # JSON، معمولاً {"sql": "..."}). تا الان این رشته هیچ‌وقت
    # truncate نمی‌شد؛ بعد از چند بار retry روی یه سؤال، هر کدوم SQL
    # کامل خودشون رو تو تاریخچه جا می‌ذاشتن و حجم پیام‌ها بدون سقف
    # رشد می‌کرد -- همون چیزی که باعث خطای 413 (Request too large)
    # از Groq شد.
    # ---------------------------------------------------------
    MAX_ARG_CHARS = 800
    tool_calls = message.get("tool_calls")
    if tool_calls:
        new_tool_calls = []
        for tc in tool_calls:
            tc = dict(tc)
            fn = dict(tc.get("function", {}))
            args_str = fn.get("arguments")
            if isinstance(args_str, str) and len(args_str) > MAX_ARG_CHARS:
                try:
                    parsed_args = json.loads(args_str)
                except json.JSONDecodeError:
                    parsed_args = None
                if (
                    isinstance(parsed_args, dict)
                    and isinstance(parsed_args.get("sql"), str)
                ):
                    parsed_args["sql"] = (
                        parsed_args["sql"][:MAX_ARG_CHARS] + "...[truncated]"
                    )
                    fn["arguments"] = json.dumps(parsed_args, ensure_ascii=False)
                else:
                    fn["arguments"] = args_str[:MAX_ARG_CHARS] + "...[truncated]"
            tc["function"] = fn
            new_tool_calls.append(tc)
        compacted["tool_calls"] = new_tool_calls

    return compacted


@safe_node("agent")
def agent_node(state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    conversation_context = state.get(
        "conversation_context",
        {},
    )

    if conversation_context.get("is_follow_up"):
        context_parts = [
            "[FOLLOW-UP CONTROL]",
            "این سؤال ادامه‌ی مستقیم سؤال قبلی است.",
            "محصول، metric و بازه‌ی قبلی را تغییر نده.",
        ]

        if conversation_context.get("product_id") is not None:
            context_parts.append(
                f"product_id = {conversation_context['product_id']}"
            )

        if conversation_context.get("product_title"):
            context_parts.append(
                f"product_title = {conversation_context['product_title']}"
            )

        if conversation_context.get("metric"):
            context_parts.append(
                f"metric = {conversation_context['metric']}"
            )

        if conversation_context.get("metric_label"):
            context_parts.append(
                f"metric_label = {conversation_context['metric_label']}"
            )

        if conversation_context.get("period_label"):
            context_parts.append(
                f"period_label = {conversation_context['period_label']}"
            )

        if conversation_context.get("period_start"):
            context_parts.append(
                f"period_start = {conversation_context['period_start']}"
            )

        if conversation_context.get("period_end"):
            context_parts.append(
                f"period_end = {conversation_context['period_end']}"
            )

        if conversation_context.get("previous_result") is not None:
            context_parts.append(
                f"previous_result = "
                f"{conversation_context['previous_result']}"
            )

        context_parts.extend(
            [
                "",
                "برای «چرا؟» ranking جدید انجام نده.",
                "بازه‌ی زمانی جدید نساز.",
                "محصول یا product_id را تغییر نده.",
            ]
        )

    # ---------------------------------------------------------
    # پیام‌های کنترلیِ همین دور (follow-up + تاریخ مرجع) عمداً از
    # لیست اصلی `messages` جدا نگه داشته می‌شوند، نه append.
    #
    # قبلاً این پیام‌ها با role="system" به `messages` اضافه می‌شدند و
    # بعد فیلتر «فقط اولین پیام سیستمی» (`system_messages[:1]`) روی
    # کل لیست اجرا می‌شد -- که همین پیام‌های تازه را هم قبل از رسیدن
    # به مدل حذف می‌کرد (چون اولین پیام سیستمی، همیشه پرامپت اصلیِ
    # ثابتِ ابتدای مکالمه بود، نه این‌ها). با نگه‌داشتن جدا، این پیام‌ها
    # همیشه -- صرف‌نظر از این‌که چند پیام سیستمی دیگر در تاریخچه باشد --
    # به مدل می‌رسند.
    # ---------------------------------------------------------

    turn_control_messages: list[dict[str, Any]] = []

    if conversation_context.get("is_follow_up"):
        turn_control_messages.append(
            {
                "role": "system",
                "content": "\n".join(context_parts),
            }
        )

    reference_date = get_reference_date().isoformat()

    turn_control_messages.append(
        {
            "role": "system",
            "content": (
                "[DATASET TIME CONTROL]\n"
                f"Reference date: {reference_date}\n"
                "Interpret every relative or explicit time expression "
                "(day, week, month, quarter, year, recent periods, date ranges, etc.) "
                "relative to this reference date. Convert it to exact period_start "
                "and period_end values before querying. Never use today's date, "
                "system date, or MAX(timestamp) to determine the time range.\n"
                "Never compute the exact calendar date yourself by hand: always "
                "write the SQL bound as an expression relative to the literal "
                f"reference date, e.g. '{reference_date}'::date - INTERVAL 'N days/months', "
                "and let PostgreSQL evaluate it. Only PostgreSQL's own date "
                "arithmetic is trusted for this."
            ),
        }
    )

    # پرامپت اصلی (اولین پیام سیستمی تاریخچه) همیشه حفظ می‌شود.
    base_system_messages = [
        m for m in messages
        if m.get("role") == "system"
    ][:1]

    recent_messages = [
        m for m in messages
        if m.get("role") != "system"
    ][-7:]

    llm_messages = [
        _compact_message_for_llm(m)
        for m in base_system_messages + recent_messages + turn_control_messages
    ]

    response = call_llm_with_tools(
        llm_messages,
        TOOL_DEFINITIONS,
    )

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

def compact_tool_result(
    tool_name: str,
    result: Any,
) -> str:
    """
    خروجی ابزار را برای context LLM کوچک می‌کند.

    هدف:
    - جلوگیری از رشد شدید token
    - حفظ فیلدهای مهم برای Follow-up
    - حفظ خطاها
    """

    # ---------------------------------------------------------
    # Error
    # ---------------------------------------------------------

    if isinstance(result, dict) and result.get("error"):
        # قبلاً این بخش کل result رو بدون هیچ سقفی dump می‌کرد -- یعنی
        # هر بار SQL رد می‌شد (که با اعتبارسنجی‌های جدید بیشتر هم رخ
        # می‌ده)، کل متن SQL رد‌شده + پیام خطا بدون کوچیک‌سازی وارد
        # تاریخچه می‌شد. چند بار رد شدن پشت‌سرهم همین چیزیه که باعث رد
        # شدن از سقف TPM گروک شد (413 Request too large).
        trimmed = dict(result)
        if isinstance(trimmed.get("rejected_sql"), str) and len(trimmed["rejected_sql"]) > 500:
            trimmed["rejected_sql"] = trimmed["rejected_sql"][:500] + "...[truncated]"
        if isinstance(trimmed.get("error"), str) and len(trimmed["error"]) > 800:
            trimmed["error"] = trimmed["error"][:800] + "...[truncated]"
        return json.dumps(
            trimmed,
            ensure_ascii=False,
            default=str,
        )

    # ---------------------------------------------------------
    # String
    # ---------------------------------------------------------

    if isinstance(result, str):

        if len(result) <= 3000:
            return result

        return (
            result[:3000]
            + "\n...[tool result truncated]"
        )

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    if isinstance(result, list):

        compact = {
            "row_count": len(result),
            "rows": result[:8],
        }

        text = json.dumps(
            compact,
            ensure_ascii=False,
            default=str,
        )

        if len(text) > 3500:
            text = (
                text[:3500]
                + "\n...[tool result truncated]"
            )

        return text

    # ---------------------------------------------------------
    # Dict
    # ---------------------------------------------------------

    if isinstance(result, dict):

        # ابتدا فیلدهای مهم را حفظ کن
        important_keys = {
            "product_id",
            "id",
            "product_title",
            "title_fa",
            "metric",
            "metric_label",
            "units_sold",
            "purchase_cnt",
            "purchase_count",
            "revenue",
            "sales",
            "period_start",
            "period_end",
            "start_date",
            "end_date",
            "comparison",
            "change",
            "change_pct",
            "hit_count",
            "reviews",
            "summary",
        }

        compact: dict[str, Any] = {}

        for key, value in result.items():

            if key in important_keys:
                compact[key] = value

            elif isinstance(value, list):

                compact[key] = {
                    "row_count": len(value),
                    "rows": value[:8],
                }

        text = json.dumps(
            compact,
            ensure_ascii=False,
            default=str,
        )

        if len(text) <= 3500:
            return text

        return (
            text[:3500]
            + "\n...[tool result truncated]"
        )

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    text = str(result)

    if len(text) > 3500:
        text = (
            text[:3500]
            + "\n...[tool result truncated]"
        )

    return text

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

        print("\n===== TOOL CALL =====")
        print("TOOL:", name)
        print("ARGS:", arguments)
        print("=====================\n")

        result = execute_tool_call(name, arguments)

        print("\n===== TOOL RESULT =====")
        print(result)
        print("=======================\n")

        ok = "error" not in result
        all_errored = all_errored and not ok
        # ---------------------------------------------------------
        # مهم:
        # نتیجه کامل ابزار را مستقیماً وارد conversation نمی‌کنیم.
        # فقط نسخه compact شده برای LLM ارسال می‌شود.
        # ---------------------------------------------------------
        compact_result = compact_tool_result(name, result)

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": compact_result,
            }
        )

        tool_trace.append(
            {
                "tool": name,
                "arguments": arguments,
                "ok": ok,
                "summary": (
                    result.get("error")
                    if not ok
                    else compact_tool_result(name, result)
                ),
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
def finalize_node(state: GraphState):
    messages = state.get("messages", [])

    if not messages:
        return {
            "final_answer": "",
            "errors": ["finalize: no messages"]
        }

    last_message = messages[-1]

    if isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls")

        if tool_calls:
            return {
                "final_answer": "",
                "errors": [
                    "finalize: model returned tool_calls "
                    "instead of a final answer"
                ]
            }

        content = last_message.get("content", "")
    else:
        tool_calls = getattr(
            last_message,
            "tool_calls",
            None
        )

        if tool_calls:
            return {
                "final_answer": "",
                "errors": [
                    "finalize: model returned tool_calls "
                    "instead of a final answer"
                ]
            }

        content = getattr(
            last_message,
            "content",
            ""
        )

    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return {
        "final_answer": str(content).strip()
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
def correct_answer_node(
    state: GraphState,
) -> dict[str, Any]:

    validation = state.get(
        "validation",
        {},
    )

    warnings = validation.get(
        "warnings"
    ) or []

    conversation_context = state.get(
        "conversation_context",
        {},
    )

    # ---------------------------------------------------------
    # برای Follow-up، سؤال اصلی + context فعال را به correction
    # می‌دهیم؛ نه فقط «چرا؟»
    # ---------------------------------------------------------

    question = _extract_last_user_question(
        state.get("messages", [])
    )

    if conversation_context.get("is_follow_up"):

        context_parts = []

        if conversation_context.get("product_id") is not None:
            context_parts.append(
                f"product_id="
                f"{conversation_context['product_id']}"
            )

        if conversation_context.get("product_title"):
            context_parts.append(
                f"product_title="
                f"{conversation_context['product_title']}"
            )

        if conversation_context.get("metric"):
            context_parts.append(
                f"metric="
                f"{conversation_context['metric']}"
            )

        if conversation_context.get("period_start"):
            context_parts.append(
                f"period_start="
                f"{conversation_context['period_start']}"
            )

        if conversation_context.get("period_end"):
            context_parts.append(
                f"period_end="
                f"{conversation_context['period_end']}"
            )

        question = (
            "این سؤال یک Follow-up است.\n"
            f"سؤال فعلی: {question}\n"
            "Context فعال:\n"
            + "\n".join(context_parts)
        )

    corrected = _real_correct_answer(
        question=question,
        final_answer=state.get(
            "final_answer",
            "",
        ),
        warnings=warnings,
        tool_trace=state.get(
            "tool_trace",
            [],
        ),
    )

    return {
        "final_answer": corrected,
        "errors": [
            "correct_answer: جواب یک‌بار اصلاح شد "
            f"(match_score={validation.get('match_score')} "
            "زیر آستانه)"
        ],
    }

