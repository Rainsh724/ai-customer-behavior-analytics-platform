## PATH: app/graph/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import (
    agent_node,
    tools_node,
    finalize_node,
    validate_node,
    MAX_ITERATIONS,
    MAX_CONSECUTIVE_TOOL_ERRORS,
)


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


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    builder = StateGraph(GraphState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("validate", validate_node)

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

    # finalize قبلاً جواب نهایی رو قطعی کرده؛ validate فقط برای ممیزی/لاگه
    # (نگاه کن به audit.py) و final_answer رو دستکاری نمی‌کنه.
    builder.add_edge("finalize", "validate")
    builder.add_edge("validate", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
