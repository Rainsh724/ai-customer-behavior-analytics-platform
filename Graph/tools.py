## PATH: app/graph/tools.py
"""
تعریف سه ابزار (Tool_SQL / Tool_RAG / Tool_BI) به فرمت OpenAI function-
calling، + یک dispatcher که اسم ابزار و آرگومان‌هاش (که از خروجی LLM
می‌رسه) رو می‌گیره و پیاده‌سازی واقعی مربوطه رو صدا می‌زنه.

این فایل تنها جایی‌ست که "شکل" ابزارها تعریف می‌شه؛ منطق واقعی هرکدوم در
فایل خودش می‌مونه (sql_agent.py / vector_retriever.py / bi_agent.py) --
دقیقاً همون تفکیک قبلی، فقط حالا این‌ها node نیستن بلکه tool function ان.
"""
from __future__ import annotations

import logging
from typing import Any

from .sql_agent import run_sql_tool
from .vector_retriever import run_rag_tool
from .bi_agent import run_bi_tool

logger = logging.getLogger(__name__)


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tool_sql",
            "description": (
                "داده‌های ساختاریافته و عددی (فروش، قیمت، تعداد نظرات، آمار KPI) "
                "رو با یک کوئری SQL امن از دیتابیس محصولات/فروش/رفتار کاربر می‌گیره."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_need": {
                        "type": "string",
                        "description": (
                            "توضیح دقیق فارسی از داده‌ای که لازم داری، نه لزوماً "
                            "عین سوال کاربر (مثلاً 'فروش محصول X در سه ماه اخیر "
                            "به تفکیک ماه')."
                        ),
                    }
                },
                "required": ["data_need"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_rag",
            "description": (
                "جست‌وجوی معنایی در نظرات مشتری‌ها برای پیدا کردن دلایل کیفی/"
                "احساسات پشت یک رفتار یا شکایت."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_topic": {
                        "type": "string",
                        "description": "موضوع/سوالی که باید در نظرات جست‌وجو بشه.",
                    },
                    "aspect": {
                        "type": ["string", "null"],
                        "description": (
                            "اگه کاربر به یک جنبه‌ی خاص اشاره کرده (مثلاً باتری، "
                            "قیمت، بسته‌بندی) اسمش رو اینجا بذار تا جست‌وجو "
                            "فیلترشده و دقیق بشه؛ اگه سوال کلی و عمومی است "
                            "(مثلاً 'چرا مردم از برند X شاکی هستند؟') null بذار "
                            "تا جست‌وجوی آزاد روی کل نظرات انجام بشه."
                        ),
                    },
                    "product_id": {"type": ["integer", "null"]},
                },
                "required": ["search_topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_bi",
            "description": (
                "وقتی کاربر صریحاً نمودار/داشبورد/مصورسازی خواسته، یک لینک "
                "فیلترشده و آماده‌کلیک از داشبورد Power BI می‌سازه."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "متریکی که باید نمایش داده بشه (مثلاً sales, rating, complaints, price).",
                    },
                    "dimension": {
                        "type": ["string", "null"],
                        "description": "بعدی که باید بر اساسش تفکیک بشه (مثلاً product, brand, category, month).",
                    },
                    "product_id": {"type": ["integer", "null"]},
                    "brand_id": {"type": ["integer", "null"]},
                    "days_back": {"type": ["integer", "null"]},
                },
                "required": ["metric"],
            },
        },
    },
]


def execute_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatcher: اسم ابزار + آرگومان‌های parse‌شده (از JSON خروجی مدل) رو
    می‌گیره، پیاده‌سازی واقعی رو صدا می‌زنه، و همیشه یک dict برمی‌گردونه
    (حتی در خطا -- با کلید "error") تا هیچ‌وقت اجرای گراف با یک exception
    خام قطع نشه؛ خطا باید به‌صورت پیام "tool" به خودِ Agent برسه تا طبق
    سند معماری (حلقه‌ی self-correction) خودش تصمیم بگیره.
    """
    try:
        if name == "tool_sql":
            return run_sql_tool(data_need=arguments.get("data_need", ""))

        if name == "tool_rag":
            return run_rag_tool(
                search_topic=arguments.get("search_topic", ""),
                aspect=arguments.get("aspect"),
                product_id=arguments.get("product_id"),
            )

        if name == "tool_bi":
            return run_bi_tool(
                metric=arguments.get("metric", ""),
                dimension=arguments.get("dimension"),
                product_id=arguments.get("product_id"),
                brand_id=arguments.get("brand_id"),
                days_back=arguments.get("days_back"),
            )

        return {"error": f"ابزار ناشناخته: {name}"}

    except Exception as exc:  # noqa: BLE001 - این‌جا هم آخرین خط دفاعیه
        logger.exception("execute_tool_call: tool '%s' crashed", name)
        return {"error": f"اجرای ابزار '{name}' با خطای غیرمنتظره شکست خورد: {exc}"}
