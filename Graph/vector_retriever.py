## PATH: app/graph/vector_retriever.py
"""
پیاده‌سازی واقعی retrieval_planner + vector_retriever.

جریان کار:
    retrieval_planner:
        LLM سوال آزاد کاربر رو به یک "فرمت ساختاریافته" تبدیل می‌کنه:
        فیلترهای متادیتا (product_id / بازه‌ی زمانی / خریدار بودن) +
        چند عبارت جست‌وجوی کوتاه که هرکدوم روی یک "جنبه" (aspect) تمرکز دارن.
        این خروجی رو در state["retrieval_plan"] و state["metadata_filters"] می‌ذاریم.

    vector_retriever:
        Agent (نه خودِ LLM!) به‌ازای هر search phrase یک embedding می‌گیره،
        متادیتا فیلترها رو به SQL تبدیل می‌کنه، pgvector رو کوئری می‌کنه،
        و نتیجه‌ها رو به فرمت qualitative_evidence نرمال می‌کنه.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .db import vector_similarity_search
from .llm_client import call_llm_json, embed_text
from .state import GraphState

logger = logging.getLogger(__name__)

RETRIEVAL_PLANNER_SYSTEM_PROMPT = """
تو یک برنامه‌ریز بازیابی (retrieval planner) هستی. سوال مدیر رو بگیر و
فقط یک JSON با این فرمت برگردون -- هیچ متن اضافه‌ای ننویس:

{
  "product_id": <int یا null>,
  "sentiment_filter": "negative" | "positive" | "neutral" | null,
  "only_buyers": <true|false>,
  "days_back": <int یا null>,   // مثلا 30 یعنی فقط کامنت‌های ۳۰ روز اخیر
  "search_phrases": ["<عبارت کوتاه فارسی متمرکز روی یک جنبه/aspect>", ...]  // ۲ تا ۵ تا
}

فقط بر اساس چیزی که در سوال یا entities/time_range اومده فیلتر بذار؛
اگه چیزی مشخص نبود null/false بذار. search_phrases باید عبارت‌های کوتاه
و مرتبط با نظرات مشتری‌ها باشن (نه کل سوال کاربر)، چون این عبارت‌ها مستقیم
embed و روی کامنت‌ها جست‌وجوی شباهت میشن.
"""


def retrieval_planner(state: GraphState) -> dict[str, Any]:
    question = state.get("question", "")
    entities = state.get("entities", {}) or {}
    time_range = state.get("time_range", {}) or {}

    user_prompt = (
        f"سوال: {question}\n"
        f"موجودیت‌های شناسایی‌شده: {entities}\n"
        f"بازه‌ی زمانی شناسایی‌شده: {time_range}\n"
    )
    plan = call_llm_json(RETRIEVAL_PLANNER_SYSTEM_PROMPT, user_prompt)

    metadata_filters = {
        "product_id": plan.get("product_id"),
        "sentiment_filter": plan.get("sentiment_filter"),
        "only_buyers": bool(plan.get("only_buyers", False)),
        "days_back": plan.get("days_back"),
    }

    return {
        "retrieval_plan": plan,
        "metadata_filters": metadata_filters,
    }


def _build_where_clause(metadata_filters: dict[str, Any]) -> tuple[str, tuple]:
    """فیلترهای متادیتا رو به شرط SQL پارامتریزه تبدیل می‌کنه (بدون string concat خام)."""
    clauses: list[str] = []
    params: list[Any] = []

    if metadata_filters.get("product_id"):
        clauses.append("c.product_id = %s")
        params.append(metadata_filters["product_id"])

    if metadata_filters.get("only_buyers"):
        clauses.append("c.is_buyer = TRUE")

    days_back = metadata_filters.get("days_back")
    if days_back:
        cutoff = datetime.utcnow() - timedelta(days=int(days_back))
        clauses.append("c.created_at >= %s")
        params.append(cutoff)

    sentiment = metadata_filters.get("sentiment_filter")
    if sentiment:
        clauses.append(
            "EXISTS (SELECT 1 FROM comment_aspects ca "
            "WHERE ca.comment_id = c.id AND ca.sentiment = %s)"
        )
        params.append(sentiment)

    return " AND ".join(clauses), tuple(params)


def vector_retriever(state: GraphState) -> dict[str, Any]:
    plan = state.get("retrieval_plan", {}) or {}
    metadata_filters = state.get("metadata_filters", {}) or {}

    search_phrases: list[str] = plan.get("search_phrases") or [state.get("question", "")]
    where_sql, where_params = _build_where_clause(metadata_filters)

    search_queries: dict[str, str] = {}
    retrieved_documents: list[dict[str, Any]] = []
    qualitative_evidence: list[dict[str, Any]] = []

    for i, phrase in enumerate(search_phrases):
        if not phrase.strip():
            continue
        key = f"aspect_{i}"
        search_queries[key] = phrase

        embedding = embed_text(phrase)
        hits = vector_similarity_search(
            query_embedding=embedding,
            where_sql=where_sql,
            where_params=where_params,
            top_k=8,
        )

        for hit in hits:
            retrieved_documents.append({**hit, "matched_phrase": phrase})

        if hits:
            qualitative_evidence.append(
                {
                    "aspect_query": phrase,
                    "claim": f"شواهد کیفی برای «{phrase}» از {len(hits)} نظر مرتبط",
                    "comment_ids": [h["comment_id"] for h in hits],
                    "sample_comments": [h["raw_text_normalized"] for h in hits[:3]],
                    "avg_distance": sum(h["distance"] for h in hits) / len(hits),
                }
            )

    return {
        "search_queries": search_queries,
        "retrieved_documents": retrieved_documents,
        "qualitative_evidence": qualitative_evidence,
    }
