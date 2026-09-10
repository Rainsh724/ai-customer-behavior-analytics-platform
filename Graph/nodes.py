## PATH: app/graph/nodes.py
from __future__ import annotations
import os
import functools
import json
import logging
from os import name
from typing import Any, Callable

from .state import GraphState
from .llm_client import call_llm_with_tools, call_llm_json
from .tools import TOOL_DEFINITIONS, execute_tool_call
from .audit import validate_answer, correct_answer as _real_correct_answer, CORRECTION_THRESHOLD
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

# اگه validate تشخیص بده match_score پایینه، به‌جای فقط بازنویسیِ متنیِ
# جواب (correct_answer، که دسترسی به ابزار نداره)، یک‌بار برمی‌گردیم به
# نود agent تا واقعاً بتونه -- اگه لازم بود -- یک tool_call جدید (مثلاً
# SQL اصلاح‌شده) بزنه. این سقف مستقل از MAX_ITERATIONS چک می‌شه که این
# حلقه هم بی‌نهایت نشه؛ بعد از این تعداد تلاش، اگه بازم امتیاز پایین
# بود، دیگه فقط correct_answer (بازنویسیِ متنیِ یک‌باره) اجرا می‌شه.
MAX_CORRECTION_RETRIES = int(os.getenv("MAX_CORRECTION_RETRIES", "1"))

# ============================================================
# MULTI-QUESTION SPLITTING -- کاهش مصرف توکن برای سوالات چندبخشی
# ============================================================
# مشکل: وقتی کاربر چند سوال مستقل رو در یک پیام می‌پرسه (مثلاً
# «پرفروش‌ترین محصول کدومه؟ و نظر مشتریا راجع‌به برند X چیه؟ و نرخ
# بازگشت ۳ ماه اخیر چقدره؟»)، همه‌ی این‌ها در یک حلقه‌ی واحد
# agent<->tools پردازش می‌شدن: هر بخش چندتا tool_call اضافه می‌کنه،
# نتیجه‌ی خام همه‌شون تو همون یک تاریخچه‌ی در حال رشد جمع می‌شه، و کل
# این حجم هر دور دوباره به مدل فرستاده می‌شه -- دقیقاً همون الگویی که
# باعث رد شدن از سقف توکن (413 Request too large / TPM) و متوقف شدن
# وسط کار می‌شه.
#
# راه‌حل: قبل از رسیدن به agent، یک تشخیص سبک (یک تماس JSON کوچیک، بدون
# ابزار) چک می‌کنه که آیا سوال واقعاً چند بخش *مستقل* داره یا نه. اگه
# نه (اکثر سوالات)، هیچ چیزی عوض نمی‌شه -- مسیر agent/tools/finalize/
# validate دقیقاً مثل قبل، بدون کوچیک‌ترین تغییر، اجرا می‌شه.
#
# اگه بله، هر بخش با یک context مستقل و کوچیک (فقط پرامپت اصلی + همون
# کنترل‌های همیشگی «تاریخ مرجع»/«follow-up» + خودِ همون بخش -- نه کل
# تاریخچه‌ی بخش‌های قبلی) پردازش می‌شه؛ یعنی حجم هر تماس LLM کوچیک و
# ثابت می‌مونه، صرف‌نظر از اینکه سوال چند بخش داره. هر بخش هم دقیقاً با
# همون audit.validate_answer/correct_answer که مسیر عادی استفاده می‌کنه
# بررسی و در صورت نیاز اصلاح می‌شه -- یعنی دقت/کیفیت هیچ بخشی نسبت به
# قبل کم نمی‌شه، فقط پردازش موازی/تکه‌تکه‌ست.
# ============================================================

ENABLE_MULTI_QUESTION_SPLIT = os.getenv(
    "ENABLE_MULTI_QUESTION_SPLIT", "true"
).strip().lower() in ("1", "true", "yes")

# حداکثر تعداد بخش‌های مستقل که یک سوال بهشون تفکیک می‌شه.
MAX_SUBQUESTIONS = int(os.getenv("MULTI_QUESTION_MAX_PARTS", "4"))

# سقف دور agent<->tools برای *هر بخش* (کمتر از MAX_ITERATIONS کلی، چون
# هر بخش قاعدتاً باید ساده‌تر از کل سوال چندبخشی باشه).
SUBQUESTION_MAX_ITERATIONS = int(os.getenv("MULTI_QUESTION_SUBITERATIONS", "4"))

MULTI_QUESTION_SPLIT_PROMPT = """
You are responsible for detecting whether a user's question (addressed to
a business-analysis Agent) actually contains several fully independent
parts or not.
 
Return only a JSON object in this format -- write no extra text:
 
{"is_multi": true|false, "questions": ["...", "..."]}
 
Important rules -- decide very conservatively:
 
- The default is false. Only return true when the question truly has two
  or more fully independent parts with no dependency on each other. A
  clear example of independent parts: "Which product sells best? And also
  what do users think of brand X? And what's the return rate over the last
  3 months?" -- three fully separate questions, none of which needs the
  answer to the others.
 
- If the parts of the question depend on each other (one needs the answer
  to another -- like "which product sells better, and why?" where "why"
  depends on the answer to the first part, or causal questions whose steps
  are chained per the "causal questions" rule), set is_multi to false;
  this is a single question, not several independent ones.
 
- If it's just one clause/sentence with several adjectives, conditions, or
  adverbs (not several separate questions with separate interrogative
  verbs), false.
 
- If true, rewrite each part as a complete, independent Persian question;
  if a shared qualifier (such as a time range, product/brand name, or
  metric type) appears only once in the original question but applies to
  all parts, repeat that same qualifier in every rewritten question so
  that no part loses this shared qualifier.
 
- Do not drop any detail, number, product/brand name, condition, or
  qualifier present in the original question; only separate the
  independent parts, don't lose any precision from the question.
 
- Maximum 4 parts. If you detect more than 4 independent parts, pick the 4
  most important ones.
"""


def _split_question(question: str) -> list[str]:
    """
    تشخیص می‌ده سوال چند بخش مستقل داره یا نه؛ اگه بله، لیست بخش‌های
    بازنویسی‌شده رو برمی‌گردونه، وگرنه [question] (یعنی بدون تغییر).

    fail-open: هر خطایی (شکست تماس LLM، JSON نامعتبر، ساختار غیرمنتظره)
    باعث برگشت به [question] می‌شه -- یعنی بدترین حالت همون رفتار قبلی
    (تک‌سوالی) است، نه شکست کل درخواست.
    """
    if not ENABLE_MULTI_QUESTION_SPLIT:
        return [question]

    if not question or not question.strip():
        return [question]

    try:
        result = call_llm_json(MULTI_QUESTION_SPLIT_PROMPT, question)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "multi_question: تفکیک سوال شکست خورد، به‌صورت تک‌سوالی ادامه می‌دیم: %s",
            exc,
        )
        return [question]

    if not isinstance(result, dict) or not result.get("is_multi"):
        return [question]

    raw_questions = result.get("questions")

    if not isinstance(raw_questions, list):
        return [question]

    cleaned = [str(q).strip() for q in raw_questions if str(q).strip()]

    if len(cleaned) < 2:
        return [question]

    return cleaned[:MAX_SUBQUESTIONS]



@safe_node("detect_multi_question")
def detect_multi_question_node(state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    question = _extract_last_user_question(messages)

    sub_questions = _split_question(question)
    is_multi = len(sub_questions) > 1

    print("\n===== DETECT MULTI-QUESTION =====")
    if is_multi:
        print(f"چندبخشی تشخیص داده شد -- {len(sub_questions)} بخش:")
        for i, q in enumerate(sub_questions, start=1):
            print(f"  {i}. {q}")
    else:
        print("تک‌بخشی -- مسیر عادی agent طی می‌شه.")
    print("==================================\n")

    return {
        "is_multi_question": is_multi,
        "sub_questions": sub_questions if is_multi else [],
    }



@safe_node("prepare_subquestion")
def prepare_subquestion_node(
    state: GraphState,
) -> dict[str, Any]:

    sub_questions = state.get("sub_questions") or []
    index = state.get("current_sub_question_index", 0)

    if index >= len(sub_questions):
        return {
            "errors": [
                "prepare_subquestion: index خارج از محدوده sub_questions است"
            ]
        }

    messages = state.get("messages", [])

    base_system_message = next(
        (
            message
            for message in messages
            if message.get("role") == "system"
        ),
        None,
    )

    if base_system_message is None:
        return {
            "errors": [
                "prepare_subquestion: پیام system اصلی پیدا نشد"
            ]
        }

    original_question = _extract_last_user_question(messages)

    conversation_context = state.get(
        "conversation_context",
        {},
    )

    control_messages: list[dict[str, Any]] = [
        _dataset_time_control_message()
    ]

    followup_control_message = _build_followup_control_message(
        conversation_context
    )

    if followup_control_message is not None:
        control_messages.append(
            followup_control_message
        )

    control_messages.append(
        {
            "role": "system",
            "content": (
                "[MULTI-QUESTION SUB-QUESTION]\n"
                "سوال اصلی کاربر چند بخش مستقل دارد.\n\n"
                f"سوال اصلی:\n{original_question}\n\n"
                "بخش فعلی را به‌صورت مستقل حل کن.\n"
                "فقط به بخش فعلی پاسخ بده.\n"
                "قیدهای مشترک سوال اصلی، در صورت ارتباط، "
                "همچنان معتبر هستند."
            ),
        }
    )

    current_question = sub_questions[index]

    print(
        f"\n>>> شروع پردازش بخش {index + 1} از "
        f"{len(sub_questions)}: {current_question}\n"
    )

    initial_messages = [
        base_system_message,
        *control_messages,
        {
            "role": "user",
            "content": current_question,
        },
    ]

    return {
        "current_sub_question": current_question,
        "sub_question_messages": initial_messages,
        "sub_question_tool_trace": [],
        "sub_question_iterations": 0,
        "sub_question_consecutive_tool_errors": 0,
        "sub_question_correction_attempts": 0,
        "sub_question_validation": {},
    }


@safe_node("sub_agent")
def sub_agent_node(
    state: GraphState,
) -> dict[str, Any]:

    messages = list(
        state.get("sub_question_messages", [])
    )

    if not messages:
        return {
            "errors": [
                "sub_agent: sub_question_messages خالی است"
            ]
        }

    iterations = state.get(
        "sub_question_iterations",
        0,
    )

    response = call_llm_with_tools(
        messages,
        TOOL_DEFINITIONS,
    )

    messages.append(response)

    return {
        "sub_question_messages": messages,
        "sub_question_iterations": iterations + 1,
    }


@safe_node("sub_tools")
def sub_tools_node(
    state: GraphState,
) -> dict[str, Any]:

    messages = list(
        state.get("sub_question_messages", [])
    )

    if not messages:
        return {
            "errors": [
                "sub_tools: sub_question_messages خالی است"
            ]
        }

    last_message = messages[-1]

    tool_calls = last_message.get(
        "tool_calls"
    ) or []

    if not tool_calls:
        return {
            "errors": [
                "sub_tools: پیام آخر tool_call ندارد"
            ]
        }

    all_errored = True
    tool_messages = []
    
    tool_trace = list(
        state.get("sub_question_tool_trace", [])
    )

    for call in tool_calls:

        call_id = call.get("id", "")
        function = call.get("function", {})

        name = function.get("name", "")

        try:
            arguments = json.loads(
                function.get("arguments") or "{}"
            )
        except json.JSONDecodeError:
            arguments = {}

        print("\n----- SUB-QUESTION TOOL CALL -----")
        print("TOOL:", name)
        print("ARGS:", arguments)
        print("-----------------------------------\n")

        result = execute_tool_call(
            name,
            arguments,
        )

        print("\n----- SUB-QUESTION TOOL RESULT -----")
        print(result)
        print("-------------------------------------\n")

        ok = "error" not in result

        if ok:
            all_errored = False

        compact_result = compact_tool_result(
            name,
            result,
        )

        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": compact_result,
        }

        tool_messages.append(
            tool_message
        )

        tool_trace.append(
            {
                "tool": name,
                "arguments": arguments,
                "ok": ok,
                "summary": (
                    result.get("error")
                    if not ok
                    else compact_result
                ),
            }
        )

    messages.extend(tool_messages)

    consecutive_errors = state.get(
        "sub_question_consecutive_tool_errors",
        0,
    )

    if all_errored:
        consecutive_errors += 1
    else:
        consecutive_errors = 0

    return {
        "sub_question_messages": messages,
        "sub_question_tool_trace": tool_trace,
        "sub_question_consecutive_tool_errors": (
            consecutive_errors
        ),
    }


@safe_node("sub_finalize")
def sub_finalize_node(
    state: GraphState,
) -> dict[str, Any]:

    messages = state.get(
        "sub_question_messages",
        [],
    )

    if not messages:
        return {
            "final_answer": "",
            "errors": [
                "sub_finalize: پیام وجود ندارد"
            ],
        }

    last_message = messages[-1]

    if not last_message.get("tool_calls"):
        content = _content_from_message(
            last_message
        )

        print(f"\n<<< پایان این بخش -- پاسخ:\n{content}\n")

        return {
            "final_answer": content
        }

    forced_control_message = {
        "role": "system",
        "content": (
            "دیگر اجازه‌ی tool_call جدید نداری. "
            "فقط بر اساس ابزارهایی که واقعاً اجرا شده‌اند "
            "یک پاسخ نهایی بده. اگر داده کافی نیست، صریح بگو."
        ),
    }

    forced_messages = [
        *messages,
        forced_control_message,
    ]

    forced_response = call_llm_with_tools(
        forced_messages,
        TOOL_DEFINITIONS,
        tool_choice="none",
    )

    content = _content_from_message(
        forced_response
    )

    print(f"\n<<< پایان این بخش (سقف دور رسید) -- پاسخ:\n{content}\n")

    return {
        "sub_question_messages": [
            *messages,
            forced_response,
        ],
        "final_answer": content,
    }


@safe_node("sub_validate")
def sub_validate_node(
    state: GraphState,
) -> dict[str, Any]:

    final_answer = state.get(
        "final_answer",
        "",
    )

    tool_trace = state.get(
        "sub_question_tool_trace",
        [],
    )

    question = state.get(
        "current_sub_question",
        "",
    )

    validation = validate_answer(
        final_answer,
        tool_trace,
        question,
    )

    return {
        "sub_question_validation": validation
    }


@safe_node("prepare_sub_retry")
def prepare_sub_retry_node(
    state: GraphState,
) -> dict[str, Any]:

    validation = state.get(
        "sub_question_validation",
        {},
    )

    warnings = validation.get(
        "warnings"
    ) or []

    messages = list(
        state.get(
            "sub_question_messages",
            [],
        )
    )

    retry_message = {
        "role": "system",
        "content": (
            "[SUB-QUESTION VALIDATION FAILED]\n"
            f"faithfulness_score={validation.get('faithfulness_score')}\n\n"
            "مشکلات پاسخ قبلی:\n"
            + "\n".join(
                f"- {warning}"
                for warning in warnings
            )
            + "\n\n"
            "این بخش را دوباره بررسی کن. "
            "اگر مشکل از SQL یا ابزار است، "
            "ابزار را با منطق اصلاح‌شده دوباره اجرا کن. "
            "هیچ داده‌ای را حدس نزن."
        ),
    }

    messages.append(
        retry_message
    )

    return {
        "sub_question_messages": messages,
        "sub_question_correction_attempts": (
            state.get(
                "sub_question_correction_attempts",
                0,
            )
            + 1
        ),
        "sub_question_iterations": 0,
        "sub_question_consecutive_tool_errors": 0,
    }


@safe_node("next_subquestion")
def next_subquestion_node(
    state: GraphState,
) -> dict[str, Any]:

    current_index = state.get(
        "current_sub_question_index",
        0,
    )

    current_question = state.get(
        "current_sub_question",
        "",
    )

    current_answer = state.get(
        "final_answer",
        "",
    )

    current_trace = state.get(
        "sub_question_tool_trace",
        [],
    )

    return {
        "sub_question_answers": [
            {
                "index": current_index,
                "question": current_question,
                "answer": current_answer,
            }
        ],
        "tool_trace": current_trace,
        "current_sub_question_index": (
            current_index + 1
        ),
        "final_answer": "",
        "sub_question_messages": [],
        "sub_question_tool_trace": [],
        "sub_question_iterations": 0,
        "sub_question_consecutive_tool_errors": 0,
        "sub_question_correction_attempts": 0,
        "sub_question_validation": {},
    }


@safe_node("combine_subanswers")
def combine_subanswers_node(
    state: GraphState,
) -> dict[str, Any]:

    answers = state.get(
        "sub_question_answers",
        [],
    )

    if not answers:
        return {
            "final_answer": "",
            "errors": [
                "combine_subanswers: پاسخی برای ترکیب وجود ندارد"
            ],
        }

    original_question = _extract_last_user_question(
        state.get("messages", [])
    )

    answer_text = "\n\n".join(
        (
            f"بخش {item['index'] + 1}:\n"
            f"سؤال: {item['question']}\n"
            f"پاسخ: {item['answer']}"
        )
        for item in answers
    )

    system_message = {
        "role": "system",
        "content": (
            "تو پاسخ نهایی یک دستیار مدیریتی هستی.\n"
            "پاسخ‌های بخش‌های مستقل را بدون تغییر "
            "در اعداد و facts ترکیب کن.\n"
            "هیچ داده یا نتیجه جدیدی تولید نکن.\n"
            "تناقضی را که در evidence وجود ندارد ایجاد نکن.\n"
            "پاسخ را فارسی، منظم و مدیریتی ارائه کن."
        ),
    }

    user_message = {
        "role": "user",
        "content": (
            f"سؤال اصلی:\n{original_question}\n\n"
            f"پاسخ بخش‌ها:\n{answer_text}"
        ),
    }

    response = call_llm_with_tools(
        [
            system_message,
            user_message,
        ],
        TOOL_DEFINITIONS,
        tool_choice="none",
    )

    final_answer = _content_from_message(
        response
    )

    print("\n===== ترکیب نهایی پاسخ‌های چندبخشی =====")
    print(final_answer)
    print("==========================================\n")

    # ---------------------------------------------------------
    # مهم: مسیر sub_* (prepare_subquestion/sub_agent/sub_tools/...) از
    # عمد کاملاً جدا از state["messages"] اصلی کار می‌کنه (روی
    # sub_question_messages) تا هر بخش context کوچیک و مستقل خودش رو
    # داشته باشه و توکن اضافه مصرف نشه. اما همین باعث می‌شد که --
    # برخلاف مسیر تک‌سوالی (که در agent_node هر پاسخ با
    # {"messages": [response]} به state اضافه می‌شه) -- جواب نهاییِ
    # ترکیبیِ سوال چندبخشی هیچ‌وقت وارد state["messages"] نشه. نتیجه:
    # main.py::run() با memory_store.save_messages این پیام assistant
    # رو در دیتابیس ذخیره نمی‌کرد، و در نتیجه هر follow-up بعدی («همین
    # جوابتو خلاصه‌تر بده» و مشابه آن) هیچ پیام assistant ای برای پیدا
    # کردن context قبلی در تاریخچه نمی‌دید.
    # با اضافه کردن همین کلید "messages" اینجا، جواب نهایی -- درست مثل
    # مسیر تک‌سوالی -- وارد تاریخچه‌ی اصلی مکالمه می‌شه.
    # ---------------------------------------------------------
    return {
        "final_answer": final_answer,
        "messages": [
            {
                "role": "assistant",
                "content": final_answer,
            }
        ],
    }
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

    MAX_CHARS = 3000

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
    MAX_ARG_CHARS = 2000
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


def _build_followup_control_message(
    conversation_context: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Builds the "[FOLLOW-UP CONTROL]" system message from conversation_context.

    Extracted from agent_node so multi_question_node (processing each part
    of a multi-part question) can reuse exactly the same logic and wording
    without duplication.

    Returns None if this is not a follow-up question.
    """

    if not conversation_context.get("is_follow_up"):
        return None

    context_parts = [
        "[FOLLOW-UP CONTROL]",
        "This question is a direct continuation of the previous question.",
        "Do NOT change the previous product, metric, or time period.",
        "",
        "EXCEPTION: If this question asks for an opinion, recommendation, "
        "idea, or advice (for example: 'What do you think?', 'What should I do?', "
        "'Do you have any ideas?'), do NOT create a new SQL query to further "
        "analyze the same data. According to Rule 8, call tool_knowledge_base "
        "first and use the context below (product/metric/time period) to provide "
        "a practical management recommendation or business idea, not another "
        "data table.",
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
            "For a 'Why?' question, do NOT perform a new ranking.",
            "Do NOT create a new time period.",
            "Do NOT change the product or product_id.",
        ]
    )

    return {
        "role": "system",
        "content": "\n".join(context_parts),
    }


def _dataset_time_control_message() -> dict[str, Any]:
    """
    پیام سیستمیِ «[DATASET TIME CONTROL]» (تاریخ مرجع دیتاست).

    از agent_node جدا شده تا multi_question_node هم بتونه همون متن
    دقیق رو -- بدون کپی -- برای هر بخش از سوال چندبخشی استفاده کنه.
    """
    reference_date = get_reference_date().isoformat()

    return {
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


def _build_bounded_llm_messages(
    messages: list[dict[str, Any]],
    turn_control_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    از تاریخچه‌ی کامل state["messages"] (که فقط رشد می‌کنه) یک لیست
    محدود و امن برای ارسال به LLM می‌سازه:

      - فقط اولین پیام سیستمی (پرامپت اصلی) نگه داشته می‌شه.
      - سوال اولیه‌ی کاربر همیشه pin می‌شه (حتی اگه چند دور tool_call
        از تاریخچه‌ی اخیر بیرونش زده باشه).
      - فقط ۴ پیام غیرسیستمیِ اخیر + پیام‌های کنترلیِ همین دور
        (turn_control_messages) اضافه می‌شن.
      - در نهایت، مجموع حجم زیر MAX_LLM_MESSAGE_CHARS نگه داشته
        می‌شه (محافظت در برابر سقف TPM ارائه‌دهنده).

    این تابع بین agent_node (تماس عادی) و finalize_node (تماسِ
    اجباریِ tool_choice="none" وقتی سقف iterations/خطا رد شده) به
    اشتراک گذاشته می‌شه تا هر دو دقیقاً همون محافظت در برابر رشد
    بی‌رویه‌ی context رو داشته باشن.
    """
    base_system_messages = [
        m for m in messages
        if m.get("role") == "system"
    ][:1]

    non_system_messages = [
        m for m in messages
        if m.get("role") != "system"
    ]

    original_user_message = next(
        (m for m in non_system_messages if m.get("role") == "user"),
        None,
    )

    recent_messages = non_system_messages[-4:]

    pinned_messages: list[dict[str, Any]] = []
    if original_user_message is not None and original_user_message not in recent_messages:
        pinned_messages = [original_user_message]

    llm_messages = [
        _compact_message_for_llm(m)
        for m in base_system_messages + pinned_messages + recent_messages + turn_control_messages
    ]

    # ---------------------------------------------------------
    # سقف نهایی context.
    #
    # Groq روی این مدل سقف 8000 TPM دارد. عمداً پایین‌تر از آن
    # نگه می‌داریم تا tool definitions و overhead API هم فضای امن داشته باشند.
    # ---------------------------------------------------------
    MAX_LLM_MESSAGE_CHARS = 24000

    def _message_size(message: dict[str, Any]) -> int:
        return len(
            json.dumps(
                message,
                ensure_ascii=False,
                default=str,
            )
        )

    fixed_messages = [
        m
        for m in llm_messages
        if m.get("role") == "system"
    ]

    dynamic_messages = [
        m
        for m in llm_messages
        if m.get("role") != "system"
    ]

    current_size = sum(_message_size(m) for m in fixed_messages)

    selected_dynamic: list[dict[str, Any]] = []

    # از جدیدترین پیام‌ها شروع می‌کنیم تا نتیجه‌ی آخرین tool همیشه حفظ شود.
    for message in reversed(dynamic_messages):
        size = _message_size(message)

        if current_size + size > MAX_LLM_MESSAGE_CHARS:
            continue

        selected_dynamic.append(message)
        current_size += size

    selected_dynamic.reverse()

    return fixed_messages + selected_dynamic


@safe_node("agent")
def agent_node(state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    conversation_context = state.get(
        "conversation_context",
        {},
    )

    followup_control_message = _build_followup_control_message(conversation_context)

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

    if followup_control_message is not None:
        turn_control_messages.append(followup_control_message)

    # ---------------------------------------------------------
    # فیدبک ممیزی (اگه validate جواب قبلی رو رد کرده و از prepare_retry
    # برگشتیم اینجا) -- دقیقاً مثل پیام‌های follow-up/dataset-time، جدا
    # از `messages` نگه داشته می‌شه، نه append، چون فیلتر «فقط اولین
    # پیام سیستمی» آن را حذف می‌کرد.
    # ---------------------------------------------------------

    retry_feedback = state.get("retry_feedback")

    if retry_feedback:
        warnings = retry_feedback.get("warnings") or []
        warning_lines = (
            "\n".join(f"- {w}" for w in warnings)
            if warnings
            else "- ممیز دلیل مشخصی اعلام نکرد، ولی امتیاز faithfulness_score خیلی پایین بود."
        )

        turn_control_messages.append(
            {
                "role": "system",
                "content": (
                    "[VALIDATION FAILED -- یک فرصت دیگه برای اصلاح داری]\n"
                    f"جواب قبلی‌ات رد شد (faithfulness_score="
                    f"{retry_feedback.get('faithfulness_score')}).\n"
                    "دلایل/ادعاهای بی‌پایه:\n"
                    f"{warning_lines}\n\n"
                    "اگه مشکل از خودِ کوئری SQL بود (فیلتر اشتباه، ستون "
                    "اشتباه، منطق ناقص، threshold نامناسب)، الان یک "
                    "tool_call جدید با SQL اصلاح‌شده بزن و دوباره تلاش "
                    "کن. اگه واقعاً بعد از بررسی، داده‌ی کافی برای این "
                    "سوال وجود نداره، صریح همینو بگو -- چیزی که در "
                    "نتیجه‌ی ابزارها نبوده اختراع نکن."
                ),
            }
        )

    turn_control_messages.append(_dataset_time_control_message())

    # پرامپت اصلی همیشه حفظ می‌شه، سوال اولیه‌ی کاربر pin می‌شه، و کل
    # لیست زیر سقف حجم TPM نگه داشته می‌شه -- نگاه کن به
    # _build_bounded_llm_messages (این منطق بین agent_node و finalize_node
    # مشترکه).
    llm_messages = _build_bounded_llm_messages(messages, turn_control_messages)

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

    اصول:
    - JSON همیشه معتبر باقی بماند.
    - فیلدهای مهم برای Follow-up حفظ شوند.
    - SQL و RAG بیش از حد وارد history نشوند.
    - هیچ JSONای بعد از json.dumps با slicing بریده نشود.
    """

    # ---------------------------------------------------------
    # Helper: compact rows بدون خراب کردن ساختار JSON
    # ---------------------------------------------------------
    def compact_rows(rows: list[Any], max_rows: int = 10) -> list[Any]:

        EXCLUDED_KEYS = {
            "embedding",
            "embedded_comment",
            "comment_embedding",
            "vector",
            "feature_vector",
            "embedding_vector",
        }

        compacted_rows: list[Any] = []

        for row in rows[:max_rows]:

            if not isinstance(row, dict):
                compacted_rows.append(row)
                continue

            compacted_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in EXCLUDED_KEYS
                }
            )

        return compacted_rows

    # ---------------------------------------------------------
    # Error
    # ---------------------------------------------------------
    if isinstance(result, dict) and result.get("error"):
        trimmed = {}

        for key in (
            "error",
            "rejected_sql",
            "tool",
            "message",
        ):
            if key in result:
                value = result[key]

                if isinstance(value, str):
                    limit = 2000 if key == "error" else 350
                    trimmed[key] = (
                        value[:limit] + "...[truncated]"
                        if len(value) > limit
                        else value
                    )
                else:
                    trimmed[key] = value

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
            "rows": compact_rows(result, max_rows=10),
        }

        return json.dumps(
            compact,
            ensure_ascii=False,
            default=str,
        )

    # ---------------------------------------------------------
    # Dict
    # ---------------------------------------------------------
    if isinstance(result, dict):

        EXCLUDED_KEYS = {
            "embedding",
            "embedded_comment",
            "comment_embedding",
            "vector",
            "feature_vector",
            "embedding_vector",
            "chartjs_config",
            "echarts_option",
            "plotly_figure",
        }

        compact: dict[str, Any] = {}

        for key, value in result.items():

            if key in EXCLUDED_KEYS:
                continue

            if isinstance(value, list):

                compact[key] = {
                    "row_count": len(value),
                    "rows": compact_rows(
                        value,
                        max_rows=10,
                    ),
                }

            else:
                compact[key] = value

        return json.dumps(
            compact,
            ensure_ascii=False,
            default=str,
        )

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------
    text = str(result)

    if len(text) <= 3000:
        return text

    return (
        text[:3000]
        + "\n...[tool result truncated]"
    )


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
    has_error = False

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
        if not ok:
            has_error = True
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

        trace_entry = {
                    "tool": name,
                    "arguments": arguments,
                    "ok": ok,
                    "summary": (
                        result.get("error")
                        if not ok
                        else compact_tool_result(name, result)
                    ),
        }
        if name == "tool_chart" and ok:
            trace_entry["chart_data"] = {
                "chart_type": result.get("chart_type"),
                "title": result.get("title"),
                "x_field": result.get("x_field"),
                "y_field": result.get("y_field"),
                "raw_data": result.get("raw_data"),
            }
        tool_trace.append(trace_entry)

    # اگه هیچ ابزاری این دور اجرا نشده بود (tool_calls خالی بود -- که طبق
    # چک بالاتر نباید برسه اینجا)، all_errored رو مصنوعی True نکن.
    consecutive_errors = state.get("consecutive_tool_errors", 0)
    consecutive_errors = (
        consecutive_errors + 1
        if has_error
        else 0
    )

    return {
        "messages": tool_messages,
        "tool_trace": tool_trace,
        "consecutive_tool_errors": consecutive_errors,
        "tool_error": has_error,
    }


# ============================================================
# نود FINALIZE -- استخراج جواب نهایی
# ============================================================
# دو حالت:
#   1. حالت عادی: پیام آخرِ agent دیگه tool_call نداره -> همون content
#      متنی، جواب نهاییه.
#   2. حالت سقف iterations/خطا: هنوز tool_call می‌خواد ولی اجازه نداریم
#      دوباره بریم سراغ tools -> یک تماس آخر با tool_choice="none" می‌زنیم
#      تا LLM مجبور به جمع‌بندی متنی بشه، دقیقاً مثل چیزی که
#      multi_question_node در پایان هر زیرسوال انجام می‌ده (نگاه کن
#      به بالا).
#
# باگ قبلی این نود: حالت ۲ فقط در کامنت توضیح داده شده بود ولی هیچ‌وقت
# واقعاً پیاده نشده بود -- وقتی پیام آخر tool_call داشت، این نود صرفاً
# با final_answer="" و یک خطا برمی‌گشت، بدون اینکه هیچ تماسی برای
# مجبور کردن مدل به جمع‌بندی بزنه. نتیجه: هر بار که Agent دقیقاً روی
# سقف MAX_ITERATIONS/MAX_CONSECUTIVE_TOOL_ERRORS به یک tool_call جدید
# نیاز داشت (یعنی دقیقاً همون سوال‌های سخت‌تری که چند دور اصلاح SQL
# لازم دارن -- مثل زدن به رد شدنِ production_validator و تلاش دوباره)،
# جواب نهایی خالی می‌موند، validate آن را "خالی" (match_score=0) اعلام
# می‌کرد، و در نهایت correct_answer (که فقط بازنویسیِ متنیه، بدون
# دسترسی به SQL/داده‌ی تازه) یک جواب نوعیِ "داده‌ی کافی نیست" تحویل
# می‌داد -- حتی وقتی tool_trace از قبل داده‌ی معتبر و کامل داشت.

def _content_from_message(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content or "").strip()


def _tool_calls_from_message(message: Any):
    if isinstance(message, dict):
        return message.get("tool_calls")
    return getattr(message, "tool_calls", None)


@safe_node("finalize")
def finalize_node(state: GraphState):
    messages = state.get("messages", [])

    if not messages:
        return {
            "final_answer": "",
            "errors": ["finalize: no messages"]
        }

    last_message = messages[-1]

    if not _tool_calls_from_message(last_message):
        # حالت ۱: مدل خودش مستقیم جواب متنی داده -- چیز اضافه‌ای لازم نیست.
        return {
            "final_answer": _content_from_message(last_message)
        }

    # حالت ۲: به سقف iterations/خطای متوالی رسیدیم درحالی‌که مدل هنوز
    # یک tool_call جدید می‌خواست. نمی‌ذاریم اون tool_call اجرا بشه (وگرنه
    # سقف بی‌معنی می‌شد)، ولی هم نمی‌تونیم با دست‌خالی برگردیم -- یک
    # تماس آخر، اجباری و بدون امکان ابزار جدید، می‌زنیم.
    forced_control_message = {
        "role": "system",
        "content": (
            "You are not allowed to make any additional tool calls. "
            "The maximum number of tool rounds or attempts has been reached. "
            "Now provide a final answer in Persian that is clear, natural, "
            "and management-oriented, using only the tool results that were "
            "actually executed so far (not information from tools you intended "
            "to call but did not run). "

            "If the available tool results are not sufficient to fully answer "
            "the user's question, explicitly state which parts cannot be verified "
            "with the available data. However, do not discard information that is "
            "actually supported by the tool results by simply saying 'there is "
            "not enough data'. Report all verified findings completely. "

            "Never go beyond the evidence provided by tool outputs. "
            "If the data only shows correlation, do not present a definite cause "
            "or a guaranteed recommendation. "

            "Report aspects using exactly their original names and do not invent "
            "or add interpretations to their meanings."
        ),
    }

    turn_control_messages = [
        _dataset_time_control_message(),
        forced_control_message,
    ]

    llm_messages = _build_bounded_llm_messages(messages, turn_control_messages)

    forced_response = call_llm_with_tools(
        llm_messages,
        TOOL_DEFINITIONS,
        tool_choice="none",
    )

    forced_content = _content_from_message(forced_response)

    if not forced_content:
        # حتی تماس اجباریِ tool_choice="none" هم متن خالی برگردوند --
        # این دیگه واقعاً یک شکست غیرمنتظره‌ست (نه رفتار عادیِ سقف)،
        # پس به‌عنوان خطا ثبتش می‌کنیم تا در لاگ/errors دیده بشه.
        return {
            "messages": [forced_response],
            "final_answer": "",
            "errors": [
                "finalize: forced tool_choice='none' call still "
                "returned empty content"
            ],
        }

    return {
        "messages": [forced_response],
        "final_answer": forced_content,
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
    question = _extract_last_user_question(state.get("messages", []))
    validation = validate_answer(final_answer, tool_trace, question)

    warnings = validation.get("warnings") or []
    extra_errors = [f"validate: {w}" for w in warnings] if warnings else []

    return {
        "validation": validation,
        **({"errors": extra_errors} if extra_errors else {}),
    }


# ============================================================
# نود PREPARE_RETRY -- آماده‌سازی یک تلاش واقعی برای اصلاح از طریق Agent
# ============================================================
# فقط وقتی به اینجا می‌رسیم که graph.py::route_after_validate تشخیص
# داده match_score زیر آستانه بوده و هنوز به MAX_CORRECTION_RETRIES
# نرسیدیم. برخلاف correct_answer (که فقط متن رو بازنویسی می‌کنه و
# دسترسی به ابزار نداره)، اینجا کاری با final_answer نداریم -- فقط
# فیدبک ممیزی رو در state["retry_feedback"] می‌ذاریم و شمارنده رو
# +۱ می‌کنیم؛ یال گراف از اینجا مستقیم می‌ره به "agent" (نگاه کن به
# graph.py) که خودش این فیدبک رو (به‌صورت یک پیام system موقت) می‌بینه
# و می‌تونه یک tool_call جدید بزنه.

@safe_node("prepare_retry")
def prepare_retry_node(state: GraphState) -> dict[str, Any]:
    validation = state.get("validation", {})

    return {
        "retry_feedback": {
            "faithfulness_score": validation.get("faithfulness_score"),
            "warnings": validation.get("warnings") or [],
        },
        "correction_attempts": state.get("correction_attempts", 0) + 1,
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
            f"(faithfulness_score={validation.get('faithfulness_score')} "
            "زیر آستانه)"
        ],
    }