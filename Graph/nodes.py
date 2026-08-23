## PATH: app/graph/nodes.py
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Callable

from .state import GraphState
from .sql_agent import sql_agent as _real_sql_agent
from .vector_retriever import (
    retrieval_planner as _real_retrieval_planner,
    vector_retriever as _real_vector_retriever,
)
from .bi_agent import (
    bi_planner as _real_bi_planner,
    bi_builder as _real_bi_builder,
)

logger = logging.getLogger(__name__)


# ============================================================
# 0. SAFETY WRAPPER
# ============================================================
# `errors` exists in GraphState but nothing ever wrote to it. Wrapping every
# node means a failure in one node (LLM timeout, bad SQL, etc.) is recorded
# and surfaced instead of crashing the whole graph.run() silently.

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
# 1. QUESTION ANALYZER
# ============================================================

@safe_node("question_analyzer")
def question_analyzer(state: GraphState) -> dict[str, Any]:
    """
    Analyze the manager's question.

    Later:
    - LLM structured output
    - Intent classification
    - Entity extraction
    - Metric extraction
    - Time range extraction
    """

    question = state.get("question", "")
    if not question.strip():
        return {
            "intent": "unknown",
            "entities": {},
            "metrics": [],
            "dimensions": [],
            "time_range": {},
            "analysis_goal": "unknown",
            "errors": ["question_analyzer: empty question"],
        }

    # TODO: replace with real LLM-based structured extraction using `question`.
    return {
        "intent": "unknown",
        "entities": {},
        "metrics": [],
        "dimensions": [],
        "time_range": {},
        "analysis_goal": "unknown",
    }


# ============================================================
# 2. QUERY ROUTER
# ============================================================

# کلیدواژه‌های فارسی/انگلیسی‌ای که نشون می‌دن کاربر علاوه بر جواب، یک
# نمودار/داشبورد هم خواسته. مثل route فعلی (که هنوز rule-based و placeholder
# هست، نه LLM)، این هم یک قانون ساده‌ست تا با یک LLM classifier واقعی
# جایگزین بشه -- نکته‌ی مهم اینه که وابسته به `route` نیست و کاملاً orthogonal
# تصمیم‌گیری می‌شه، دقیقاً همون‌جایی (query_router) که کاربر ازمون سوال
# می‌پرسه.
_CHART_KEYWORDS = re.compile(
    r"(نمودار|چارت|گراف|دیاگرام|رسم\s*کن|پلات|داشبورد|نمایش\s*بده|بکش|"
    r"chart|graph|plot|dashboard|visuali[sz]e)",
    re.IGNORECASE,
)


def _detect_wants_chart(question: str) -> bool:
    return bool(_CHART_KEYWORDS.search(question or ""))


@safe_node("query_router")
def query_router(state: GraphState) -> dict[str, Any]:
    """
    Decide (در یک مرحله):
        1. route:  sql | rag | hybrid          -- داده از کجا میاد
        2. wants_chart: bool                    -- آیا BI/نمودار هم لازمه

    این دو تصمیم عمداً orthogonal ان: `route` می‌گه چطور شواهد جمع بشه،
    `wants_chart` می‌گه آیا علاوه بر جواب روایی، یک نمودار/داشبورد هم باید
    ساخته بشه. bi_planner/bi_builder (در bi_agent.py) دقیقاً به همین
    `wants_chart` گوش می‌دن و بعد از هر سه‌ی sql/rag/hybrid فعال می‌شن --
    یعنی BI با هر سه مسیر "ترکیب" می‌شه بدون این‌که route چهارمی لازم باشه.

    Later this can use an LLM with structured output (هم برای route هم
    برای wants_chart، در همون یک تماس).
    """

    question = state.get("question", "")
    intent = state.get("intent", "")

    # Temporary rule-based placeholder
    if intent in {"metric_lookup", "trend_analysis"}:
        route = "sql"

    elif intent in {"review_analysis", "qualitative_analysis"}:
        route = "rag"

    else:
        route = "hybrid"

    wants_chart = _detect_wants_chart(question)

    return {
        "route": route,
        "wants_chart": wants_chart,
        # final_join باید بدونه چند شاخه منتظر بمونه: بدون نمودار فقط
        # زنجیره‌ی روایتی (۱)، با نمودار هم روایت هم BI (۲).
        "bi_expected_legs": 2 if wants_chart else 1,
    }


# ============================================================
# 3. HYBRID PLANNER
# ============================================================

@safe_node("hybrid_planner")
def hybrid_planner(state: GraphState) -> dict[str, Any]:
    """
    Decide whether SQL and RAG should run:

        parallel
        sequential
    """

    intent = state.get("intent", "")
    analysis_goal = state.get("analysis_goal", "")

    # Placeholder logic
    if intent == "root_cause_analysis":
        hybrid_mode = "sequential"

    elif analysis_goal == "compare_quantitative_and_qualitative":
        hybrid_mode = "parallel"

    else:
        hybrid_mode = "parallel"

    return {
        "hybrid_mode": hybrid_mode
    }


# ============================================================
# 4. SQL AGENT
# ============================================================

@safe_node("sql_agent")
def sql_agent(state: GraphState) -> dict[str, Any]:
    """Real implementation: LLM-generated SQL, validated, run read-only
    against Postgres. See sql_agent.py. (چارت‌پسند شدن خروجی وقتی
    wants_chart=True هم همون‌جا مدیریت می‌شه.)"""
    return _real_sql_agent(state)


# ============================================================
# 5. SQL DIAGNOSIS
# ============================================================

@safe_node("sql_diagnosis")
def sql_diagnosis(state: GraphState) -> dict[str, Any]:
    """
    Interpret SQL results.

    Example:
        Sales: -20%
        Conversion: -18%

    Later this can be deterministic analytics
    or an LLM-based analytical node.
    """

    return {
        "sql_diagnosis": {}
    }


# ============================================================
# 6. NEED QUALITATIVE EVIDENCE (conditional edge helper)
# ============================================================

def need_qualitative_evidence(state: GraphState) -> str:
    """
    Conditional node.

    Returns:
        "rag"
        "skip_rag"

    Used after sql_diagnosis on BOTH the "sql"-only route and the
    "hybrid/sequential" route — see graph.py for why this must be the
    *only* outgoing decision from sql_diagnosis.
    """

    diagnosis = state.get("sql_diagnosis", {})

    # Placeholder: until sql_diagnosis is implemented, this always says
    # "yes, go fetch qualitative evidence". Flip the default once
    # sql_diagnosis actually produces a verdict.
    needs_rag = diagnosis.get("needs_qualitative_evidence", True)

    if needs_rag:
        return "rag"

    return "skip_rag"


# ============================================================
# 7. RETRIEVAL PLANNER
# ============================================================

@safe_node("retrieval_planner")
def retrieval_planner(state: GraphState) -> dict[str, Any]:
    """Real implementation: LLM turns the question into metadata filters
    + focused search phrases. See vector_retriever.py."""
    return _real_retrieval_planner(state)


# ============================================================
# 8. ASPECT AGGREGATOR
# ============================================================

@safe_node("aspect_aggregator")
def aspect_aggregator(state: GraphState) -> dict[str, Any]:
    """
    Aggregate ABSA outputs.

    Example:

        battery: 850
        price: 430
        build_quality: 90
    """

    return {
        "aspect_statistics": {}
    }


# ============================================================
# 9. TEMPORAL ASPECT ANALYSIS
# ============================================================

@safe_node("temporal_aspect_analysis")
def temporal_aspect_analysis(state: GraphState) -> dict[str, Any]:
    """
    Compare aspect complaints across periods.

    Example:

        price:
            previous = 120
            current = 600
            growth = 400%
    """

    return {
        "aspect_trends": {}
    }


# ============================================================
# 10. ASPECT RANKER
# ============================================================

@safe_node("aspect_ranker")
def aspect_ranker(state: GraphState) -> dict[str, Any]:
    """
    Rank aspects based on:

        frequency
        trend increase
        recency
        severity
        ABSA confidence
    """

    return {
        "ranked_aspects": []
    }


# ============================================================
# 11. VECTOR RETRIEVER
# ============================================================

@safe_node("vector_retriever")
def vector_retriever(state: GraphState) -> dict[str, Any]:
    """Real implementation: embeds each search phrase, queries pgvector
    (comments_embedding, HNSW/cosine), normalizes hits. See
    vector_retriever.py."""
    return _real_vector_retriever(state)


# ============================================================
# 12. EVIDENCE NORMALIZER
# ============================================================

@safe_node("evidence_normalizer")
def evidence_normalizer(state: GraphState) -> dict[str, Any]:
    """
    Convert outputs from SQL / ABSA / RAG into a unified evidence format.

    Rebuilds the full list from current state on every call — see the
    comment on `structured_evidence` in state.py for why this field must
    NOT use an additive reducer.

    این نود، صرف‌نظر از این‌که سوال از کدوم route (sql/rag/hybrid) اومده،
    نقطه‌ی تلاقی همیشگی همه‌شونه -- به همین دلیل در graph.py دقیقاً از
    همین‌جا شاخه‌ی BI (bi_planner) هم فعال می‌شه: هر داده‌ای که تا این‌جا
    جمع شده (sql_result / aspect_statistics / qualitative_evidence) برای
    ساخت نمودار هم در دسترسه.
    """

    evidence = []

    # SQL Evidence
    if state.get("sql_result"):
        evidence.append(
            {
                "source_type": "sql",
                "claim": "Structured data result",
                "evidence": state["sql_result"],
            }
        )

    # ABSA Evidence
    if state.get("aspect_statistics"):
        evidence.append(
            {
                "source_type": "absa",
                "claim": "Aspect distribution",
                "evidence": state["aspect_statistics"],
            }
        )

    # RAG Evidence
    for item in state.get("qualitative_evidence", []):
        evidence.append(
            {
                "source_type": "review",
                "claim": item.get("claim", ""),
                "evidence": item,
            }
        )

    return {
        "structured_evidence": evidence
    }


# ============================================================
# 13. EVIDENCE FUSION
# ============================================================

@safe_node("evidence_fusion")
def evidence_fusion(state: GraphState) -> dict[str, Any]:
    """
    Combine all evidence.

    Important:
    This node should NOT invent causal relationships.
    """

    fused = state.get("structured_evidence", [])

    return {
        "fused_evidence": fused
    }


# ============================================================
# 14. INSIGHT GENERATOR
# ============================================================

@safe_node("insight_generator")
def insight_generator(state: GraphState) -> dict[str, Any]:
    """
    Generate evidence-based managerial insight.

    Later:
    - LLM
    - Grounded generation
    """

    return {
        "insight": ""
    }


# ============================================================
# 12b. HYBRID JOIN BARRIER
# ============================================================
# Both legs of the "parallel" hybrid path (sql_diagnosis -> here, and
# vector_retriever -> here) route into this node instead of straight into
# evidence_normalizer. It just counts arrivals; graph.py's join_router
# decides whether to actually proceed (both legs done) or dead-end this
# particular arrival at `join_wait` (the other leg hasn't finished yet).

EXPECTED_PARALLEL_BRANCHES = 2


@safe_node("hybrid_join")
def hybrid_join(state: GraphState) -> dict[str, Any]:
    return {"hybrid_arrivals": 1}


@safe_node("join_wait")
def join_wait(state: GraphState) -> dict[str, Any]:
    """Dead end on purpose: the first leg to arrive at the join stops here;
    only the leg that completes the barrier continues on to
    evidence_normalizer. No outgoing edges from this node."""
    return {}


# ============================================================
# 14b. BI PLANNER / BUILDER (نمودار و داشبورد)
# ============================================================
# فقط وقتی state["wants_chart"] True باشه، graph.py این دو نود رو (به‌صورت
# موازی با evidence_fusion -> insight_generator) بعد از evidence_normalizer
# فعال می‌کنه. جزئیات واقعی در bi_agent.py.

@safe_node("bi_planner")
def bi_planner(state: GraphState) -> dict[str, Any]:
    """Real implementation: LLM chart_request رو از سوال + شکل داده‌ی
    موجود (sql_result / aspect_statistics) می‌سازه. See bi_agent.py."""
    return _real_bi_planner(state)


@safe_node("bi_builder")
def bi_builder(state: GraphState) -> dict[str, Any]:
    """Real implementation: Agent (نه LLM) که chart_request رو به یک یا
    چند chart_spec واقعی (dashboard) تبدیل می‌کنه. See bi_agent.py."""
    return _real_bi_builder(state)


# ============================================================
# 14c. FINAL JOIN BARRIER (روایت + BI)
# ============================================================
# insight_generator (انتهای زنجیره‌ی روایت) و bi_builder (انتهای زنجیره‌ی
# BI) طول متفاوتی دارن، پس ممکنه در سوپراستپ‌های متفاوت اجرا بشن. دقیقاً
# همون الگوی hybrid_join/join_wait، اما تعداد شاخه‌های موردانتظار ثابت
# نیست: وقتی wants_chart=False فقط ۱ شاخه (insight_generator) داریم، وقتی
# True دو شاخه. مقدار موردانتظار در state["bi_expected_legs"] (که
# query_router ست کرده) نگه‌داری می‌شه -- نه یک ثابت مثل
# EXPECTED_PARALLEL_BRANCHES، چون این‌جا تعداد شاخه‌ها per-run فرق می‌کنه.

@safe_node("final_join")
def final_join(state: GraphState) -> dict[str, Any]:
    return {"final_arrivals": 1}


@safe_node("final_wait")
def final_wait(state: GraphState) -> dict[str, Any]:
    """Dead end on purpose: شاخه‌ای (روایت یا BI) که زودتر برسه این‌جا
    متوقف می‌شه؛ فقط شاخه‌ای که barrier رو تکمیل می‌کنه به answer_validator
    ادامه می‌ده. بدون outgoing edge."""
    return {}


# ============================================================
# 15. ANSWER VALIDATOR
# ============================================================

@safe_node("answer_validator")
def answer_validator(state: GraphState) -> dict[str, Any]:
    """
    Validate:

        - groundedness
        - unsupported claims
        - correlation vs causation
        - time consistency
        - evidence sufficiency

    نمودار/داشبورد (اگه ساخته شده باشه) از قبل در state["chart_spec"] /
    state["dashboard"] هست (bi_builder نوشتتش) و چون LangGraph فیلدهای
    ننوشته‌شده رو دست‌نخورده نگه می‌داره، خودبه‌خود در خروجی نهایی می‌مونه.
    این‌جا فقط یک پرچم خلاصه به validation اضافه می‌کنیم تا مصرف‌کننده‌ی
    final_answer بدونه آیا نموداری هم همراهش هست یا نه.
    """

    return {
        "validation": {
            "grounded": True,
            "confidence": "unknown",
            "warnings": state.get("errors", []),
            "has_chart": bool(state.get("chart_spec")),
        },
        "final_answer": state.get("insight", "")
    }
