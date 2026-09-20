# PATH: app/graph/graph.py

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import GraphState

from .nodes import (
    # main flow
    agent_node,
    tools_node,
    finalize_node,
    validate_node,
    correct_answer_node,
    prepare_retry_node,

    # multi question flow
    detect_multi_question_node,
    prepare_subquestion_node,
    sub_agent_node,
    sub_tools_node,
    sub_finalize_node,
    sub_validate_node,
    prepare_sub_retry_node,
    next_subquestion_node,
    combine_subanswers_node,

    MAX_ITERATIONS,
    FOLLOWUP_MAX_ITERATIONS,
    MAX_CONSECUTIVE_TOOL_ERRORS,
    MAX_CORRECTION_RETRIES,
    SUBQUESTION_MAX_ITERATIONS,
)

from .audit import CORRECTION_THRESHOLD

# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_detect_multi_question(state: GraphState) -> str:
    """
    یال شرطی خروجی از detect_multi_question -- اگه سوال کاربر واقعاً
    چند بخش مستقل داشته باشه (تشخیص در nodes.py::_split_question)،
    مسیر جدا و مستقل "multi_question" رو طی می‌کنه (هر بخش با context
    کوچیک و جدا از بقیه، برای جلوگیری از رشد بی‌رویه‌ی توکن). در غیر
    این صورت (اکثر سوالات) دقیقاً همون مسیر قدیمیِ "agent" -- بدون
    کوچیک‌ترین تغییر -- طی می‌شه.
    """
    if state.get("fatal_error"):
        return "single_question"
    if state.get("is_multi_question"):
        return "multi_question"
    return "single_question"


def route_after_agent(state: GraphState) -> str:
    """
    یال شرطی خروجی از Agent (دقیقاً همون "یال‌های شرطی" سند معماری):

        - پیام آخر tool_call داره، و نه به سقف کل دورها رسیدیم نه به سقف
          خطای متوالی ابزار -> "tools"
        - پیام آخر tool_call داره ولی یکی از دو سقف بالا رد شده ->
          "finalize" (که خودش تماس آخر با tool_choice="none" می‌زنه)
        - پیام آخر tool_call نداره (یعنی LLM مستقیم جواب نهایی داده) ->
          "finalize" (یال شرطی مستقیم به پایان -- طبق سند: "در صورتی که
          ابزاری نیاز نباشد، این یال مستقیماً به پایان ختم می‌شود")

    سقف consecutive_tool_errors عمداً جدا از سقف iterations چک می‌شه:
    محافظت زودتری در برابر حالتی می‌ده که یک ابزار (مثلاً SQL) مدام شکست
    می‌خوره -- لازم نیست صبر کنیم کل ۶ دور مصرف بشه تا متوجه بشیم.
    """
    if state.get("fatal_error"):
        return "finalize"

    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}

    # برای سوالات فالوآپ سقف کوتاه‌تر (حداکثر ۳ دور) برای جلوگیری از وسواس و مصرف توکن
    is_follow_up = bool(state.get("conversation_context", {}).get("is_follow_up"))
    effective_max_iterations = FOLLOWUP_MAX_ITERATIONS if is_follow_up else MAX_ITERATIONS

    reached_iteration_cap = state.get("iterations", 0) >= effective_max_iterations
    reached_error_cap = state.get("consecutive_tool_errors", 0) >= MAX_CONSECUTIVE_TOOL_ERRORS

    if last_message.get("tool_calls") and not reached_iteration_cap and not reached_error_cap:
        return "tools"

    return "finalize"


def route_after_validate(state: GraphState) -> str:
    """
    یال شرطی خروجی از Validate -- تصمیم می‌گیره جواب نیاز به اصلاح داره
    یا نه، بر اساس faithfulness_score که audit.py برمی‌گردونه (قبلاً
    اسمش match_score بود -- تغییر فقط اسمیه، منطق همون قبلیه). سه مقصد
    ممکنه:

        - "ok"      -> مستقیم END (جواب قبوله)
        - "retry"    -> prepare_retry -> agent؛ یعنی واقعاً یک فرصت
                        دیگه با دسترسی به ابزار (مثلاً اصلاح SQL و
                        اجرای دوباره) داده می‌شه. فقط تا سقف
                        MAX_CORRECTION_RETRIES (در nodes.py) مجازه --
                        بعدش دیگه به "retry" برنمی‌گردیم تا حلقه
                        بی‌نهایت نشه.
        - "correct" -> correct_answer (فقط بازنویسیِ متنی، بدون
                        دسترسی به ابزار) -- آخرین خط دفاعی، وقتی سقف
                        retry تموم شده.

    توجه: relevance_score و confidence_score (محورهای جدید audit.py) در
    این تصمیم دخیل نیستن -- فقط لاگ/ارزیابی می‌شن (نگاه کن به
    memory_store.py::log_evaluation). تصمیم retry/correct هنوز فقط بر
    اساس faithfulness_score گرفته می‌شه.

    اگه validation خاموش بود (VALIDATION_ENABLED=false) یا
    faithfulness_score به هر دلیلی نداشتیم (مثلاً خودِ تماس validate
    شکست خورد)، فیل-سیف "ok" برمی‌گردونیم -- بدون امتیاز، نمی‌شه تصمیم
    به اصلاح گرفت.
    """
    if state.get("fatal_error"):
        return "ok"

    validation = state.get("validation", {})

    if validation.get("skipped"):
        return "ok"

    faithfulness_score = validation.get("faithfulness_score")
    if faithfulness_score is None or faithfulness_score >= CORRECTION_THRESHOLD:
        return "ok"

    attempts = state.get("correction_attempts", 0)
    if attempts < MAX_CORRECTION_RETRIES:
        return "retry"

    return "correct"


def route_after_sub_agent(
    state: GraphState,
) -> str:
    if state.get("fatal_error"):
        return "finalize"

    messages = state.get(
        "sub_question_messages",
        [],
    )

    last_message = (
        messages[-1]
        if messages
        else {}
    )

    reached_iteration_cap = (
        state.get(
            "sub_question_iterations",
            0,
        )
        >= MAX_ITERATIONS
    )

    reached_error_cap = (
        state.get(
            "sub_question_consecutive_tool_errors",
            0,
        )
        >= MAX_CONSECUTIVE_TOOL_ERRORS
    )

    if (
        last_message.get("tool_calls")
        and not reached_iteration_cap
        and not reached_error_cap
    ):
        return "tools"

    return "finalize"


def route_after_sub_validate(
    state: GraphState,
) -> str:
    if state.get("fatal_error"):
        return "accept"

    validation = state.get(
        "sub_question_validation",
        {},
    )

    if validation.get("skipped"):
        return "next"

    faithfulness_score = validation.get(
        "faithfulness_score"
    )

    if (
        faithfulness_score is None
        or faithfulness_score >= CORRECTION_THRESHOLD
    ):
        return "next"

    attempts = state.get(
        "sub_question_correction_attempts",
        0,
    )

    if attempts < MAX_CORRECTION_RETRIES:
        return "retry"

    return "accept"


def route_after_next_subquestion(
    state: GraphState,
) -> str:
    if state.get("fatal_error"):
        return "combine"

    index = state.get(
        "current_sub_question_index",
        0,
    )

    sub_questions = state.get(
        "sub_questions",
        [],
    )

    if index < len(sub_questions):
        return "next"

    return "combine"


def route_after_tools(state: GraphState) -> str:
    if state.get("fatal_error"):
        return "finalize"

    consecutive_errors = state.get(
        "consecutive_tool_errors",
        0,
    )

    if consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
        return "finalize"

    return "agent"

# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    builder = StateGraph(GraphState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("validate", validate_node)
    builder.add_node("correct_answer", correct_answer_node)
    builder.add_node("prepare_retry", prepare_retry_node)
    builder.add_node("detect_multi_question", detect_multi_question_node)
    builder.add_node("prepare_subquestion", prepare_subquestion_node)
    builder.add_node("sub_agent", sub_agent_node)
    builder.add_node("sub_tools", sub_tools_node)
    builder.add_node("sub_finalize", sub_finalize_node)
    builder.add_node("sub_validate", sub_validate_node)
    builder.add_node("prepare_sub_retry", prepare_sub_retry_node)
    builder.add_node("next_subquestion", next_subquestion_node)
    builder.add_node("combine_subanswers", combine_subanswers_node)

    # یال ورود: درخواست اولیه‌ی کاربر (که main.py قبلاً به‌عنوان یک پیام
    # "user" به state["messages"] اضافه کرده) اول از یک تشخیص سبک
    # (چندبخشی بودن یا نه) رد می‌شه. برای سوالات معمولی (تک‌بخشی) این
    # فقط یک تماس کوچیک اضافه‌ست و مسیر بعدش دقیقاً همون "agent" قدیمیه.
    builder.add_edge(
        START,
        "detect_multi_question",
    )

    builder.add_conditional_edges(
        "detect_multi_question",
        route_after_detect_multi_question,
        {
            "single_question": "agent",
            "multi_question": "prepare_subquestion",
        },
    )


# --------------------MULTI QUESTION--------------------

    builder.add_edge(
        "prepare_subquestion",
        "sub_agent",
    )

    builder.add_conditional_edges(
        "sub_agent",
        route_after_sub_agent,
        {
            "tools": "sub_tools",
            "finalize": "sub_finalize",
        },
    )

    builder.add_edge(
        "sub_tools",
        "sub_agent",
    )

    builder.add_edge(
        "sub_finalize",
        "sub_validate",
    )

    builder.add_conditional_edges(
        "sub_validate",
        route_after_sub_validate,
        {
            "retry": "prepare_sub_retry",
            "next": "next_subquestion",
            "accept": "next_subquestion",
        },
    )

    builder.add_edge(
        "prepare_sub_retry",
        "sub_agent",
    )

    builder.add_conditional_edges(
        "next_subquestion",
        route_after_next_subquestion,
        {
            "next": "prepare_subquestion",
            "combine": "combine_subanswers",
        },
    )

    builder.add_edge(
        "combine_subanswers",
        "validate",
    )

    # یال‌های شرطی خروجی از Agent.
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )

    # یال بازگشتی: بعد از اجرای ابزار(ها)، نتیجه‌ی خام همیشه به Agent
    # برمی‌گرده تا تفسیر بشه -- همین یال، هم "حلقه‌ی اصلاح خطا" و هم
    # "اجرای موازی" (چند tool_call در یک دور) و هم ادامه‌ی استدلال
    # چندمرحله‌ای رو، بدون نیاز به هیچ نود میانی دیگه، پیاده می‌کنه.
    # builder.add_edge("tools", "agent")

    builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "agent": "agent",
        "finalize": "finalize",
    },
)

    # finalize قبلاً جواب نهایی رو قطعی کرده؛ validate ممیزی می‌کنه و
    # امتیاز می‌ده. اگه امتیاز پایین بود و هنوز سقف retry تموم نشده،
    # از prepare_retry دوباره می‌ره به agent (فرصت واقعی برای اصلاح
    # SQL و اجرای دوباره‌ی ابزار). وقتی سقف retry تموم شده باشه، فقط
    # correct_answer (بازنویسیِ متنیِ یک‌باره، مستقیم به END) اجرا می‌شه.
    builder.add_edge("finalize", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "ok": END,
            "retry": "prepare_retry",
            "correct": "correct_answer",
        },
    )
    builder.add_edge("prepare_retry", "agent")
    builder.add_edge("correct_answer", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
