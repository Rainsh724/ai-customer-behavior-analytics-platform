## PATH: app/graph/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import (
    agent_node,
    tools_node,
    finalize_node,
    validate_node,
    correct_answer_node,
    MAX_ITERATIONS,
    MAX_CONSECUTIVE_TOOL_ERRORS,
)
from .audit import CORRECTION_THRESHOLD


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

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
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}

    reached_iteration_cap = state.get("iterations", 0) >= MAX_ITERATIONS
    reached_error_cap = state.get("consecutive_tool_errors", 0) >= MAX_CONSECUTIVE_TOOL_ERRORS

    if last_message.get("tool_calls") and not reached_iteration_cap and not reached_error_cap:
        return "tools"

    return "finalize"


def route_after_validate(state: GraphState) -> str:
    """
    یال شرطی خروجی از Validate -- تصمیم می‌گیره جواب نیاز به اصلاح داره
    یا نه، بر اساس match_score که audit.py برمی‌گردونه.

    نکته‌ی امنیتی مهم: این تابع هرگز به "validate" برنمی‌گرده -- فقط دو
    مقصد ممکن داره: "ok" (مستقیم END) یا "correct" (correct_answer، که
    خودش هم مستقیم به END می‌ره، نه به validate یا agent). یعنی حتی اگه
    جواب اصلاح‌شده هم کامل grounded نباشه، تلاش دومی برای اصلاح یا
    ممیزی مجدد وجود نداره -- حداکثر یک تماس اضافه‌ی LLM به‌ازای هر پاسخ.

    اگه validation خاموش بود (VALIDATION_ENABLED=false) یا match_score
    به هر دلیلی نداشتیم (مثلاً خودِ تماس validate شکست خورد)، فیل-سیف
    "ok" برمی‌گردونیم -- بدون امتیاز، نمی‌شه تصمیم به اصلاح گرفت.
    """
    validation = state.get("validation", {})

    if validation.get("skipped"):
        return "ok"

    match_score = validation.get("match_score")
    if match_score is not None and match_score < CORRECTION_THRESHOLD:
        return "correct"

    return "ok"


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

    # یال ورود: درخواست اولیه‌ی کاربر (که main.py قبلاً به‌عنوان یک پیام
    # "user" به state["messages"] اضافه کرده) مستقیم به نود Agent می‌ره.
    builder.add_edge(START, "agent")

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
    builder.add_edge("tools", "agent")

    # finalize قبلاً جواب نهایی رو قطعی کرده؛ validate ممیزی می‌کنه و
    # امتیاز می‌ده. اگه امتیاز پایین بود، یک‌بار (و فقط یک‌بار --
    # correct_answer مستقیم به END می‌ره، نه به validate) اصلاح می‌شه.
    builder.add_edge("finalize", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "ok": END,
            "correct": "correct_answer",
        },
    )
    builder.add_edge("correct_answer", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
