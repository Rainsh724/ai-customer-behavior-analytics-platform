## PATH: app/graph/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import GraphState

from .nodes import (
    question_analyzer,
    query_router,
    hybrid_planner,
    sql_agent,
    sql_diagnosis,
    need_qualitative_evidence,
    retrieval_planner,
    aspect_aggregator,
    temporal_aspect_analysis,
    aspect_ranker,
    vector_retriever,
    hybrid_join,
    join_wait,
    EXPECTED_PARALLEL_BRANCHES,
    evidence_normalizer,
    evidence_fusion,
    insight_generator,
    bi_planner,
    bi_builder,
    final_join,
    final_wait,
    answer_validator,
)


# ============================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================

def route_question(state: GraphState) -> str:
    """
    Route after query_router.
    """
    route = state.get("route")
    if route not in ("sql", "rag", "hybrid"):
        raise ValueError(f"query_router produced an invalid route: {route!r}")
    return route


def route_hybrid_mode(state: GraphState) -> str | list[str]:
    """
    Route after hybrid_planner.

    Returns a LIST of labels for "parallel" so LangGraph fans out to both
    branches directly, in the same superstep — no dummy marker node
    needed. Returns a single label for "sequential".
    """
    mode = state.get("hybrid_mode")

    if mode == "parallel":
        return ["parallel_sql", "parallel_rag"]

    if mode == "sequential":
        return "sequential"

    raise ValueError(f"hybrid_planner produced an invalid hybrid_mode: {mode!r}")


def post_sql_diagnosis_router(state: GraphState) -> str:
    """
    The ONE outgoing decision from sql_diagnosis (it must never also have a
    plain add_edge alongside this — see the note in build_graph below for
    why that combination is a bug).

    - "hybrid/parallel": RAG is already running independently (fired
      unconditionally by hybrid_planner), so sql_diagnosis must not also
      decide whether to trigger it — it just reports to the join barrier.
    - everything else ("sql"-only route, or "hybrid/sequential"): the
      classic decision — do we need to go fetch qualitative evidence.
    """
    if state.get("hybrid_mode") == "parallel":
        return "join"
    return need_qualitative_evidence(state)


def post_vector_retriever_router(state: GraphState) -> str:
    """
    vector_retriever is the tail of the RAG pipeline. In "hybrid/parallel"
    mode it must report to the join barrier instead of going straight to
    evidence_normalizer, because the SQL leg is running concurrently.
    In every other case (plain "rag" route, or the "rag" leg of
    hybrid/sequential) it's the only active branch, so it goes straight
    through.
    """
    if state.get("hybrid_mode") == "parallel":
        return "join"
    return "direct"


def join_router(state: GraphState) -> str:
    """
    Only let the FINAL arriving leg of the "parallel" hybrid path continue
    to evidence_normalizer. The leg that arrives first dead-ends at
    join_wait until the counter says both legs are in.
    """
    if state.get("hybrid_arrivals", 0) >= EXPECTED_PARALLEL_BRANCHES:
        return "proceed"
    return "wait"


def route_after_evidence_normalizer(state: GraphState) -> str | list[str]:
    """
    evidence_normalizer is the convergence point for EVERY combination of
    route/hybrid_mode (sql-only, rag-only, hybrid-sequential,
    hybrid-parallel-via-join). This is exactly where the BI branch fans
    out from, so that "BI" ends up combined with all three routes instead
    of needing its own copy of sql_agent/vector_retriever.

    Returns a LIST (fan-out, same technique as route_hybrid_mode) when a
    chart was requested, so the narrative branch (evidence_fusion) and the
    BI branch (bi_planner) both launch in the same superstep. Returns a
    single label otherwise.
    """
    if state.get("wants_chart"):
        return ["fusion", "bi"]
    return "fusion"


def final_join_router(state: GraphState) -> str:
    """
    Only let the leg that COMPLETES the final barrier continue to
    answer_validator. Mirrors join_router, but the expected arrival count
    is dynamic (state["bi_expected_legs"], set by query_router) instead of
    a fixed constant: 1 leg (just the narrative) when no chart was
    requested, 2 legs (narrative + BI) when one was.
    """
    expected = state.get("bi_expected_legs", 1)
    if state.get("final_arrivals", 0) >= expected:
        return "proceed"
    return "wait"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    builder = StateGraph(GraphState)

    # ========================================================
    # ADD NODES
    # ========================================================

    builder.add_node("question_analyzer", question_analyzer)
    builder.add_node("query_router", query_router)
    builder.add_node("hybrid_planner", hybrid_planner)
    builder.add_node("sql_agent", sql_agent)
    builder.add_node("sql_diagnosis", sql_diagnosis)
    builder.add_node("retrieval_planner", retrieval_planner)
    builder.add_node("aspect_aggregator", aspect_aggregator)
    builder.add_node("temporal_aspect_analysis", temporal_aspect_analysis)
    builder.add_node("aspect_ranker", aspect_ranker)
    builder.add_node("vector_retriever", vector_retriever)
    builder.add_node("hybrid_join", hybrid_join)
    builder.add_node("join_wait", join_wait)
    builder.add_node("evidence_normalizer", evidence_normalizer)
    builder.add_node("evidence_fusion", evidence_fusion)
    builder.add_node("insight_generator", insight_generator)
    builder.add_node("bi_planner", bi_planner)
    builder.add_node("bi_builder", bi_builder)
    builder.add_node("final_join", final_join)
    builder.add_node("final_wait", final_wait)
    builder.add_node("answer_validator", answer_validator)

    # ========================================================
    # START
    # ========================================================

    builder.add_edge(START, "question_analyzer")
    builder.add_edge("question_analyzer", "query_router")

    # ========================================================
    # ROUTE: SQL / RAG / HYBRID
    # (wants_chart از همین‌جا -- در query_router -- روی state ست می‌شه؛
    # خودش اینجا تاثیری در انتخاب مسیر نداره، فقط بعداً در
    # route_after_evidence_normalizer خونده می‌شه.)
    # ========================================================

    builder.add_conditional_edges(
        "query_router",
        route_question,
        {
            "sql": "sql_agent",
            "rag": "retrieval_planner",
            "hybrid": "hybrid_planner",
        }
    )

    # ========================================================
    # SQL PIPELINE (shared by "sql"-only route AND "hybrid" routes)
    # ========================================================

    builder.add_edge("sql_agent", "sql_diagnosis")

    # IMPORTANT: sql_diagnosis has exactly ONE outgoing decision.
    # The original version had BOTH a plain add_edge("sql_diagnosis",
    # "evidence_normalizer") AND add_conditional_edges("sql_diagnosis", ...)
    # — in LangGraph both fire unconditionally, so every "sql"-only query
    # was silently also triggering the RAG branch. That plain edge is
    # removed entirely; "skip_rag" below covers what it used to handle.
    builder.add_conditional_edges(
        "sql_diagnosis",
        post_sql_diagnosis_router,
        {
            "join": "hybrid_join",
            "rag": "retrieval_planner",
            "skip_rag": "evidence_normalizer",
        }
    )

    # ========================================================
    # RAG PIPELINE
    # ========================================================

    builder.add_edge("retrieval_planner", "aspect_aggregator")
    builder.add_edge("aspect_aggregator", "temporal_aspect_analysis")
    builder.add_edge("temporal_aspect_analysis", "aspect_ranker")
    builder.add_edge("aspect_ranker", "vector_retriever")

    # See post_vector_retriever_router: only detours through the join
    # barrier when this is the RAG leg of a "parallel" hybrid run.
    builder.add_conditional_edges(
        "vector_retriever",
        post_vector_retriever_router,
        {
            "join": "hybrid_join",
            "direct": "evidence_normalizer",
        }
    )

    # ========================================================
    # HYBRID "PARALLEL" JOIN BARRIER
    #
    # Both legs (sql_diagnosis and vector_retriever) route here instead of
    # straight to evidence_normalizer. Each arrival increments
    # hybrid_arrivals (additive reducer); only the arrival that completes
    # the barrier (both legs in) proceeds — the other dead-ends at
    # join_wait. This is what actually fixes the double/premature
    # execution of evidence_normalizer -> evidence_fusion ->
    # insight_generator -> answer_validator under "parallel" hybrid mode.
    # ========================================================

    builder.add_conditional_edges(
        "hybrid_join",
        join_router,
        {
            "proceed": "evidence_normalizer",
            "wait": "join_wait",
        }
    )
    # join_wait deliberately has no outgoing edge — it's where the leg
    # that arrives first stops.

    # ========================================================
    # HYBRID PLANNING
    #
    # "parallel"   -> fan out directly to sql_agent AND retrieval_planner
    #                 in the same superstep (no marker node required).
    # "sequential" -> sql_agent first, then sql_diagnosis decides (via
    #                 post_sql_diagnosis_router above) whether RAG runs too.
    # ========================================================

    builder.add_conditional_edges(
        "hybrid_planner",
        route_hybrid_mode,
        {
            "parallel_sql": "sql_agent",
            "parallel_rag": "retrieval_planner",
            "sequential": "sql_agent",
        }
    )

    # ========================================================
    # BI FAN-OUT (نمودار/داشبورد)
    #
    # evidence_normalizer نقطه‌ی تلاقی هر ترکیبی از sql/rag/hybrid هست، پس
    # دقیقاً همون‌جا بهترین جای شاخه‌شدن به سمت BI ـه: هر داده‌ای که تا
    # این‌جا جمع شده (sql_result / aspect_statistics / qualitative_evidence)
    # برای هم روایت هم نمودار در دسترسه.
    #
    # بدون نمودار: فقط "fusion" -> evidence_fusion (رفتار قبلی، بدون تغییر).
    # با نمودار:   هم "fusion" هم "bi" در همون سوپراستپ فعال می‌شن (fan-out،
    #              دقیقاً همون تکنیک route_hybrid_mode) -- یعنی زنجیره‌ی
    #              روایت (evidence_fusion -> insight_generator) و زنجیره‌ی
    #              BI (bi_planner -> bi_builder) موازی اجرا می‌شن.
    # ========================================================

    builder.add_conditional_edges(
        "evidence_normalizer",
        route_after_evidence_normalizer,
        {
            "fusion": "evidence_fusion",
            "bi": "bi_planner",
        }
    )

    builder.add_edge("evidence_fusion", "insight_generator")
    builder.add_edge("bi_planner", "bi_builder")

    # ========================================================
    # FINAL JOIN BARRIER (روایت + BI)
    #
    # insight_generator و bi_builder به‌جای رفتن مستقیم به answer_validator
    # (که وقتی طول دو زنجیره فرق داره، دقیقاً همون باگ "دوبار اجرا شدن"ی
    # می‌شد که hybrid_join بالاتر برای sql/rag حلش کرده)، از final_join رد
    # می‌شن. IMPORTANT: مثل sql_diagnosis، insight_generator و bi_builder هم
    # نباید هم‌زمان یک add_edge مستقیم به answer_validator داشته باشن --
    # فقط از طریق final_join.
    # ========================================================

    builder.add_edge("insight_generator", "final_join")
    builder.add_edge("bi_builder", "final_join")

    builder.add_conditional_edges(
        "final_join",
        final_join_router,
        {
            "proceed": "answer_validator",
            "wait": "final_wait",
        }
    )
    # final_wait deliberately has no outgoing edge — it's where the leg
    # that arrives first stops (mirrors join_wait above).

    # ========================================================
    # FINAL PIPELINE
    # ========================================================

    builder.add_edge("answer_validator", END)

    # ========================================================
    # COMPILE
    # ========================================================

    return builder.compile()


# Built lazily via get_graph() rather than at import time — see main.py.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
