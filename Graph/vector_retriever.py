## PATH: app/graph/vector_retriever.py
"""
پیاده‌سازی واقعی Tool_RAG.

بدون فیلتر مبتنی بر ABSA (aspect/sentiment).
---------------------------------------------
تصمیم گرفته شد از هرگونه فیلتر مبتنی بر خروجی مدل ABSA (چه aspect/term
چه sentiment) صرف‌نظر بشه -- دقت پایین استخراج اسپکت (~۳۳٪) باعث می‌شد
فیلتر سخت (EXISTS) بخش زیادی از نظرات واقعاً مرتبط رو بی‌صدا حذف کنه.
پس این یک RAG کاملاً معمولیه: فقط جست‌وجوی معنایی روی متن نظرات، با
تنها فیلتر مجازِ product_id (که چون مستقیم از comments.product_id واقعی
دیتابیس میاد، نه پیش‌بینی یک مدل، منبع خطا نیست).

خلاصه‌سازی قبل از رسیدن به LLM (بدون LLM).
--------------------------------------------
به‌جای برگردوندن فقط ۵-۶ نظر مشابه‌تر، حالا top_k=20 نتیجه از جست‌وجوی
برداری می‌گیریم -- پوشش بهتر از فضای نظرات مرتبط -- و قبل از این‌که
چیزی به Agent برسه، این ۲۰ تا با summarize_comments (در text_summary.py،
کاملاً بدون LLM/API خارجی) به یک "خلاصه‌ی فشرده" تبدیل می‌شن: مضامین
تکرارشونده (کلمات پرتکرار) + چند نظر نماینده. این باعث می‌شه هم حجم
توکن ارسالی به Agent خیلی کمتر از ۲۰ نظر خام باشه، هم Agent یک نمای کلی
از موضوعات غالب رو ببینه، نه فقط چند جمله‌ی پراکنده.

تفاوت با نسخه‌ی قبلی (پایپ‌لاین اول این پروژه): قبلاً یک نود جدا به اسم
retrieval_planner با LLM خودش سوال خام رو تحلیل می‌کرد. الان این کار
اصلاً لازم نیست -- Agent خودش در همون تماس اول search_topic و (اختیاری)
product_id رو مستقیم می‌ده. هیچ تماس LLM اضافه‌ای در مسیر RAG نیست
(embed_text هم یک مدل محلی‌ست، نه LLM؛ summarize_comments هم فقط
فرکانس‌شماری کلمات است).
"""
from __future__ import annotations

import logging
from typing import Any
import psycopg2

from .db import vector_similarity_search
from .llm_client import embed_text
from .text_summary import summarize_comments

logger = logging.getLogger(__name__)

# پوشش بیشتر از فضای نظرات مرتبط نسبت به نسخه‌ی قبلی (که ۸ تا می‌گرفت)؛
# چون این ۲۰ تا خام دیگه مستقیم به Agent داده نمی‌شن (خلاصه می‌شن)، حجم
# پاسخ نهایی افزایش پیدا نمی‌کنه.
TOP_K_HITS = 20


def _sanitize_int(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("none", "null", "") or not cleaned.isdigit():
            return None
        return int(cleaned)
    if isinstance(val, (int, float)):
        return int(val)
    return None


def _resolve_names(
    product_id: int | None = None,
    brand_id: int | None = None,
    category_id: int | None = None,
    hits: list[dict[str, Any]] | None = None,
) -> dict[str, str | None]:
    names: dict[str, str | None] = {
        "product_title": None,
        "brand_name": None,
        "category_name": None,
    }
    if hits:
        first = hits[0]
        names["product_title"] = first.get("product_title")
        names["brand_name"] = first.get("brand_name")
        names["category_name"] = first.get("category_name")
        return names

    # Fallback lookup if hits is empty so the model still knows the name of the entity
    try:
        from .db import run_readonly_query
        if brand_id is not None:
            r = run_readonly_query("SELECT name FROM brands WHERE brand_id = %s LIMIT 1;", (brand_id,))
            if r:
                names["brand_name"] = r[0].get("name")
        if category_id is not None:
            r = run_readonly_query("SELECT category2 FROM categories WHERE category_id = %s LIMIT 1;", (category_id,))
            if r:
                names["category_name"] = r[0].get("category2")
        if product_id is not None:
            r = run_readonly_query("SELECT title_fa FROM products WHERE id = %s LIMIT 1;", (product_id,))
            if r:
                names["product_title"] = r[0].get("title_fa")
    except Exception as e:
        logger.debug("_resolve_names fallback lookup failed: %s", e)
    return names


def _build_where_clause(
    product_id: Any = None,
    brand_id: Any = None,
    category_id: Any = None,
) -> tuple[str, tuple]:
    """فیلتر متادیتا رو به شرط SQL پارامتریزه تبدیل می‌کنه (بدون string concat خام).
    پشتیبانی از product_id (روی comments c) و brand_id / category_id (روی products p)."""
    conditions: list[str] = []
    params: list[Any] = []

    clean_pid = _sanitize_int(product_id)
    if clean_pid is not None:
        conditions.append("c.product_id = %s")
        params.append(clean_pid)

    clean_bid = _sanitize_int(brand_id)
    if clean_bid is not None:
        conditions.append("p.brand_id = %s")
        params.append(clean_bid)

    clean_cid = _sanitize_int(category_id)
    if clean_cid is not None:
        conditions.append("p.category_id = %s")
        params.append(clean_cid)

    if not conditions:
        return "", ()
    return " AND ".join(conditions), tuple(params)


def run_rag_tool(
    search_topic: str,
    product_id: int | None = None,
    brand_id: int | None = None,
    category_id: int | None = None,
) -> dict[str, Any]:
    """
    ورودی: search_topic (موضوع جست‌وجو در نظرات)،
    product_id (اختیاری -- فیلتر نظرات همان محصول)،
    brand_id (اختیاری -- فیلتر نظرات کل محصولات یک برند)،
    category_id (اختیاری -- فیلتر نظرات محصولات یک دسته‌بندی).
    خروجی: dict که مستقیم به‌صورت JSON در پیام "tool" به Agent برمی‌گرده
    -- شامل خلاصه (نه ۲۰ نظر خام).
    """
    if not search_topic or not search_topic.strip():
        return {"error": "search_topic خالی بود."}

    clean_pid = _sanitize_int(product_id)
    clean_bid = _sanitize_int(brand_id)
    clean_cid = _sanitize_int(category_id)

    where_sql, where_params = _build_where_clause(
        product_id=clean_pid,
        brand_id=clean_bid,
        category_id=clean_cid,
    )

    try:
        embedding = embed_text(search_topic)
        hits = vector_similarity_search(
            query_embedding=embedding,
            where_sql=where_sql,
            where_params=where_params,
            top_k=TOP_K_HITS,
        )
    except psycopg2.OperationalError as exc:
        logger.error("run_rag_tool: database operational/connection error: %s", exc)
        return {
            "error": "ارتباط با پایگاه داده برقرار نشد. لطفاً وضعیت سرویس پایگاه داده را بررسی کنید.",
            "fatal_error": True,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_rag_tool: search failed: %s", exc)
        return {"error": f"جست‌وجوی معنایی شکست خورد: {exc}"}

    if not hits:
        names = _resolve_names(clean_pid, clean_bid, clean_cid, hits=[])
        res = {
            "search_topic": search_topic,
            "product_id": clean_pid,
            "product_title": names["product_title"],
            "brand_id": clean_bid,
            "brand_name": names["brand_name"],
            "category_id": clean_cid,
            "category_name": names["category_name"],
            "hit_count": 0,
            "note": "هیچ نظر مرتبطی در پایگاه داده پیدا نشد.",
        }
        return {k: v for k, v in res.items() if v is not None}

    comment_dicts = [
        {
            "text": h["raw_text_normalized"],
            "comment_id": h["comment_id"],
            "rate": h["rate"],
            "likes": h["likes"],
            "dislikes": h["dislikes"],
            "recommendation_status": h["recommendation_status"],
        }
        for h in hits
    ]
    summary = summarize_comments(comment_dicts)
    names = _resolve_names(clean_pid, clean_bid, clean_cid, hits=hits)

    res = {
        "search_topic": search_topic,
        "product_id": clean_pid,
        "product_title": names["product_title"],
        "brand_id": clean_bid,
        "brand_name": names["brand_name"],
        "category_id": clean_cid,
        "category_name": names["category_name"],
        "hit_count": len(hits),
        "top_keywords": summary["top_keywords"],
        "representative_comments": summary["representative_comments"],
        "comment_ids": [h["comment_id"] for h in hits],
        "avg_distance": sum(h["distance"] for h in hits) / len(hits),
    }
    return {k: v for k, v in res.items() if v is not None}
