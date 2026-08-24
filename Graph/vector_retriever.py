## PATH: app/graph/vector_retriever.py
"""
پیاده‌سازی واقعی Tool_RAG.

استراتژی فیلتر (دقیقاً طبق سند معماری):
    - جست‌وجوی فیلترشده: وقتی Agent یک "aspect" مشخص (مثلاً "باتری")
      بهمون بده -> علاوه بر جست‌وجوی برداری، شرط
      `comment_aspects.term = <aspect>` هم به WHERE اضافه می‌شه.
    - جست‌وجوی آزاد: وقتی aspect خالی/None باشه (سوال کلی، مثل "چرا مردم
      از برند X شاکی هستند؟") -> فقط جست‌وجوی برداری روی کل کامنت‌های
      همون محصول/برند، بدون فیلتر aspect.

تفاوت با نسخه‌ی قبلی (پایپ‌لاین): قبلاً یک نود جدا به اسم
retrieval_planner با LLM خودش، سوال خام رو به فیلتر تبدیل می‌کرد. الان
این تحلیل رو خودِ Agent (در همون تماس اصلی‌اش) انجام می‌ده و مستقیم
search_topic/aspect/product_id رو به‌عنوان آرگومان ابزار می‌فرسته --
یعنی یک تماس LLM کمتر در مسیر RAG.
"""
from __future__ import annotations

import logging
from typing import Any

from .db import vector_similarity_search
from .llm_client import embed_text

logger = logging.getLogger(__name__)


def _build_where_clause(aspect: str | None, product_id: int | None) -> tuple[str, tuple]:
    """فیلترهای متادیتا رو به شرط SQL پارامتریزه تبدیل می‌کنه (بدون string concat خام)."""
    clauses: list[str] = []
    params: list[Any] = []

    if product_id:
        clauses.append("c.product_id = %s")
        params.append(product_id)

    if aspect:
        # جست‌وجوی فیلترشده: فقط کامنت‌هایی که یک ردیف comment_aspects با
        # همین term دارن (نگاه کن به داکیورمنت بالای فایل).
        clauses.append(
            "EXISTS (SELECT 1 FROM comment_aspects ca "
            "WHERE ca.comment_id = c.id AND ca.term = %s)"
        )
        params.append(aspect)

    return " AND ".join(clauses), tuple(params)


def run_rag_tool(search_topic: str, aspect: str | None = None, product_id: int | None = None) -> dict[str, Any]:
    """
    ورودی: search_topic (موضوع جست‌وجو در نظرات)، aspect (اختیاری --
    وجودش یعنی جست‌وجوی فیلترشده، نبودش یعنی جست‌وجوی آزاد)، product_id
    (اختیاری).
    خروجی: dict که مستقیم به‌صورت JSON در پیام "tool" به Agent برمی‌گرده.
    """
    if not search_topic or not search_topic.strip():
        return {"error": "search_topic خالی بود."}

    where_sql, where_params = _build_where_clause(aspect, product_id)

    try:
        embedding = embed_text(search_topic)
        hits = vector_similarity_search(
            query_embedding=embedding,
            where_sql=where_sql,
            where_params=where_params,
            top_k=8,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_rag_tool: search failed: %s", exc)
        return {"error": f"جست‌وجوی معنایی شکست خورد: {exc}"}

    if not hits:
        return {
            "search_topic": search_topic,
            "aspect": aspect,
            "search_mode": "filtered" if aspect else "open",
            "hits": [],
            "note": "هیچ نظر مرتبطی پیدا نشد.",
        }

    return {
        "search_topic": search_topic,
        "aspect": aspect,
        "search_mode": "filtered" if aspect else "open",
        "hit_count": len(hits),
        "sample_comments": [h["raw_text_normalized"] for h in hits[:5]],
        "comment_ids": [h["comment_id"] for h in hits],
        "avg_distance": sum(h["distance"] for h in hits) / len(hits),
    }
