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

from .db import vector_similarity_search
from .llm_client import embed_text
from .text_summary import summarize_comments

logger = logging.getLogger(__name__)

# پوشش بیشتر از فضای نظرات مرتبط نسبت به نسخه‌ی قبلی (که ۸ تا می‌گرفت)؛
# چون این ۲۰ تا خام دیگه مستقیم به Agent داده نمی‌شن (خلاصه می‌شن)، حجم
# پاسخ نهایی افزایش پیدا نمی‌کنه.
TOP_K_HITS = 20


def _build_where_clause(product_id: int | None) -> tuple[str, tuple]:
    """فیلتر متادیتا رو به شرط SQL پارامتریزه تبدیل می‌کنه (بدون string concat خام).
    عمداً یک تابع جداست تا اگه بعداً فیلتر دیگه‌ای (غیر از ABSA) لازم شد،
    جای اضافه کردنش مشخص باشه."""
    if not product_id:
        return "", ()
    return "c.product_id = %s", (product_id,)


def run_rag_tool(search_topic: str, product_id: int | None = None) -> dict[str, Any]:
    """
    ورودی: search_topic (موضوع جست‌وجو در نظرات)، product_id (اختیاری --
    وجودش یعنی جست‌وجو فقط روی نظرات همون محصول).
    خروجی: dict که مستقیم به‌صورت JSON در پیام "tool" به Agent برمی‌گرده
    -- شامل خلاصه (نه ۲۰ نظر خام).
    """
    if not search_topic or not search_topic.strip():
        return {"error": "search_topic خالی بود."}

    where_sql, where_params = _build_where_clause(product_id)

    try:
        embedding = embed_text(search_topic)
        hits = vector_similarity_search(
            query_embedding=embedding,
            where_sql=where_sql,
            where_params=where_params,
            top_k=TOP_K_HITS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_rag_tool: search failed: %s", exc)
        return {"error": f"جست‌وجوی معنایی شکست خورد: {exc}"}

    if not hits:
        return {
            "search_topic": search_topic,
            "product_id": product_id,
            "hit_count": 0,
            "note": "هیچ نظر مرتبطی پیدا نشد.",
        }

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

    return {
        "search_topic": search_topic,
        "product_id": product_id,
        "hit_count": len(hits),
        "top_keywords": summary["top_keywords"],
        "representative_comments": summary["representative_comments"],
        "comment_ids": [h["comment_id"] for h in hits],
        "avg_distance": sum(h["distance"] for h in hits) / len(hits),
    }
