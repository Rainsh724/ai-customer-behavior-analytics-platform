## PATH: app/graph/state.py
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
from operator import add


# -----------------------------
# Type aliases
# -----------------------------

Route = Literal["sql", "rag", "hybrid"]
HybridMode = Literal["parallel", "sequential"]


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
