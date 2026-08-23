## PATH: app/graph/state.py
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
from operator import add


# -----------------------------
# Type aliases
# -----------------------------

Route = Literal["sql", "rag", "hybrid"]
HybridMode = Literal["parallel", "sequential"]

# نوع نمودار پیشنهادی برای BI. جدول ("table") هم به‌عنوان یک "چارت" حساب
# می‌شه چون گاهی بهترین نمایش یک سوال، خودِ جدول دیتاست.
ChartType = Literal["bar", "line", "pie", "area", "scatter", "table"]

# BI از کدوم منبع داده باید بسازه: فقط SQL، فقط ABSA/کیفی، یا هر دو
# (وقتی هم sql_result و هم aspect_statistics/aspect_trends موجود باشن،
# مثلاً در حالت hybrid+BI -> یک داشبورد دو-نموداره).
BIDataSource = Literal["sql", "absa", "both"]


# -----------------------------
# Main LangGraph State
# -----------------------------

class GraphState(TypedDict, total=False):
    # =============================
    # User Input
    # =============================
    question: str

    # =============================
    # Question Analysis
    # =============================
    intent: str
    entities: dict[str, Any]
    metrics: list[str]
    dimensions: list[str]
    time_range: dict[str, Any]
    analysis_goal: str

    # =============================
    # Routing
    # =============================
    route: Route
    hybrid_mode: HybridMode

    # -----------------------------
    # BI (نمودار/داشبورد) -- تصمیم‌گیری در همون لحظه‌ی query_router
    # -----------------------------
    # `wants_chart` مستقل از `route` است: `route` تعیین می‌کنه داده از کجا
    # میاد (sql/rag/hybrid)، `wants_chart` تعیین می‌کنه علاوه بر جواب
    # متنی، یک نمودار/داشبورد هم باید ساخته بشه یا نه. این یعنی BI به
    # هر سه‌ی sql/rag/hybrid "ترکیب" می‌شه بدون این‌که نیاز به route چهارم
    # یا نسخه‌ی دوم از sql_agent/vector_retriever باشه.
    # نگاه کن به: nodes.py::query_router و bi_agent.py برای جزئیات.
    wants_chart: bool

    # تعداد شاخه‌هایی که باید به final_join برسن قبل از عبور به
    # answer_validator: بدون BI فقط ۱ شاخه (insight_generator)، با BI
    # دو شاخه (insight_generator + bi_builder). در query_router ست میشه.
    bi_expected_legs: int

    # =============================
    # Execution Plan
    # =============================
    execution_plan: dict[str, Any]

    # =============================
    # SQL Branch
    # =============================
    sql_query: str
    sql_result: dict[str, Any]
    sql_diagnosis: dict[str, Any]

    # =============================
    # Retrieval / ABSA Branch
    # =============================
    retrieval_plan: dict[str, Any]

    metadata_filters: dict[str, Any]

    aspect_statistics: dict[str, Any]

    aspect_trends: dict[str, Any]

    ranked_aspects: list[dict[str, Any]]

    search_queries: dict[str, str]

    # =============================
    # RAG Results
    # =============================
    # NOTE: additive on purpose — future multi-hop retrieval nodes may each
    # contribute a batch of documents in the same run.
    retrieved_documents: Annotated[list[dict[str, Any]], add]

    qualitative_evidence: Annotated[list[dict[str, Any]], add]

    # =============================
    # Unified Evidence
    # =============================
    # NOT additive: `evidence_normalizer` rebuilds this list from scratch
    # every time it runs (it re-reads sql_result / aspect_statistics /
    # qualitative_evidence from state). If this field were additive
    # (Annotated[..., add]) and the node fires more than once in a single
    # run (which happens on the "parallel" hybrid path, since the SQL leg
    # reaches this node in fewer hops than the RAG leg), the same evidence
    # would be duplicated. Plain override is what we actually want here.
    structured_evidence: list[dict[str, Any]]

    fused_evidence: list[dict[str, Any]]

    # =============================
    # BI (نمودار/داشبورد) -- خروجی
    # =============================
    # `chart_request`: خروجی bi_planner (چه نموداری، از کدوم منبع، با
    # کدوم محورها). `chart_spec`: اولین/اصلی‌ترین نمودار ساخته‌شده توسط
    # bi_builder. `dashboard`: لیست همه‌ی نمودارهایی که bi_builder ساخته
    # (وقتی هم SQL هم ABSA موجوده -> بیشتر از یک عضو).
    chart_request: dict[str, Any]
    chart_spec: dict[str, Any]
    dashboard: list[dict[str, Any]]

    # =============================
    # Final Reasoning
    # =============================
    insight: str

    final_answer: str

    # =============================
    # Validation
    # =============================
    validation: dict[str, Any]

    # =============================
    # Debugging / Observability
    # =============================
    errors: Annotated[list[str], add]

    # =============================
    # Hybrid "parallel" join barrier
    # =============================
    # The SQL leg and the RAG leg of the "parallel" hybrid path have
    # different hop counts, so they don't reach a shared downstream node in
    # the same superstep. `hybrid_join` (see nodes.py) is a barrier node
    # both legs route through; each arrival increments this counter via the
    # additive reducer, and the graph only proceeds to evidence_normalizer
    # once it reaches 2 (both legs done). This IS meant to be additive.
    hybrid_arrivals: Annotated[int, add]

    # =============================
    # Final join barrier (narrative + BI)
    # =============================
    # همون الگوی hybrid_arrivals، اما برای نقطه‌ی پایانی گراف: شاخه‌ی
    # روایت (evidence_fusion -> insight_generator) و شاخه‌ی BI
    # (bi_planner -> bi_builder) طول متفاوتی دارن، پس ممکنه در سوپراستپ‌های
    # متفاوت به هم برسن. هر دو به‌جای رفتن مستقیم به answer_validator، از
    # final_join عبور می‌کنن؛ این شمارنده هر ورود رو جمع می‌زنه و فقط وقتی
    # به bi_expected_legs برسه (۱ بدون BI، ۲ با BI) اجازه‌ی عبور به
    # answer_validator داده می‌شه. دقیقاً همون باگی که hybrid_arrivals حلش
    # کرده بود، اینجا هم بدون این شمارنده رخ می‌داد.
    final_arrivals: Annotated[int, add]
