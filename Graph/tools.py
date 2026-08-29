## PATH: app/graph/tools.py
"""
تعریف ابزارهای Agent به فرمت OpenAI function-calling، + یک dispatcher که
اسم ابزار و آرگومان‌هاش (که از خروجی LLM می‌رسه) رو می‌گیره و پیاده‌سازی
واقعی مربوطه رو صدا می‌زنه.

ابزارهای فعال فعلی: tool_sql / tool_rag / tool_chart.
ابزار غیرفعال (منتظر آماده‌شدن محتوا): tool_knowledge_base -- پایین همین
فایل، هم در TOOL_DEFINITIONS هم در execute_tool_call، عمداً به‌صورت کامنت
نگه داشته شده. نگاه کن به knowledge_base_agent.py برای توضیح کامل و
نحوه‌ی فعال‌سازی.

نکته درباره‌ی description تکراری اسکیما: SCHEMA_CONTEXT فقط یک‌بار، در
description ابزار tool_sql، کامل نوشته می‌شه. description ابزار tool_chart
فقط یک ارجاع کوتاه بهش داره -- چون همه‌ی تعریف‌های ابزار (TOOL_DEFINITIONS)
در یک تماس واحد به مدل داده می‌شن، مدل از قبل توی همون تماس اسکیما رو
دیده؛ تکرار کاملش فقط توکن اضافه مصرف می‌کنه.
"""
from __future__ import annotations

import logging
from typing import Any

from .sql_agent import run_sql_tool, SCHEMA_CONTEXT
from .vector_retriever import run_rag_tool
from .chart_agent import run_chart_tool, VALID_CHART_TYPES

# وقتی knowledge_base_agent.py آماده شد، این ایمپورت رو هم از حالت کامنت خارج کن:
# from .knowledge_base_agent import run_knowledge_base_tool

logger = logging.getLogger(__name__)


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tool_sql",
            "description": (
                "داده‌های ساختاریافته و عددی (فروش، قیمت، تعداد نظرات، آمار KPI) "
                "رو از دیتابیس Postgres می‌گیره. خودت مستقیم یک کوئری SQL معتبر "
                "(فقط SELECT یا WITH...SELECT) بر اساس اسکیمای زیر بنویس -- هیچ "
                "مرحله‌ی میانی‌ای این کوئری رو برات نمی‌سازه.\n\n"
                "قوانین اجباری:\n"
                "- فقط SELECT/WITH؛ هیچ‌وقت INSERT/UPDATE/DELETE/DDL ننویس.\n"
                "- فقط از جدول/ستون‌های اسکیمای زیر استفاده کن.\n"
                "- همیشه LIMIT بذار (حداکثر ۲۰۰) مگر aggregate/COUNT باشه.\n"
                "- فقط یک کوئری؛ چند statement با ; از هم جدا ننویس.\n"
                "- تاریخ‌ها رو با NOW()/INTERVAL بساز، هاردکد نکن مگر کاربر "
                "تاریخ دقیق داده باشه.\n"
                "- برای خوندن متن نظرات یا جست‌وجوی معنایی از این ابزار استفاده "
                "نکن -- اون کار tool_rag است.\n\n"
                f"اسکیمای دیتابیس:\n{SCHEMA_CONTEXT}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "متن کامل کوئری SQL (SELECT/WITH) که خودت بر اساس اسکیمای بالا نوشتی.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_rag",
            "description": (
                "جست‌وجوی معنایی ساده در متن نظرات مشتری‌ها برای پیدا کردن "
                "دلایل کیفی/احساسات پشت یک رفتار یا شکایت. جست‌وجو بر اساس "
                "نزدیکی معنایی (embedding) به search_topic روی ۲۰ نظر مرتبط‌تر "
                "انجام می‌شه و نتیجه از قبل خلاصه شده (مضامین تکرارشونده + چند "
                "نظر نماینده) برمی‌گرده -- نه ۲۰ نظر خام. اگه سوال مشخصاً به یک "
                "محصول خاص اشاره داره، product_id همون محصول رو بده تا جست‌وجو "
                "به نظرات همون محصول محدود بشه."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_topic": {
                        "type": "string",
                        "description": (
                            "موضوع/سوالی که باید در نظرات جست‌وجو بشه -- برای "
                            "پیدا کردن دلایل نارضایتی، موضوع رو در همون جهت "
                            "بنویس (مثلاً 'دلایل نارضایتی و شکایت از محصول X')، "
                            "نه فقط اسم محصول به‌تنهایی."
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
            "name": "tool_chart",
            "description": (
                "وقتی کاربر صریحاً نمودار/چارت/داشبورد/مصورسازی خواسته، یک "
                "نمودار واقعی از داده‌ی دیتابیس می‌سازه و در سه فرمت آماده‌ی "
                "رندر (Chart.js، ECharts، Plotly) برمی‌گردونه -- فرانت‌اند هرکدوم "
                "از این سه کتابخانه رو استفاده کنه، مستقیم قابل‌استفاده‌ست.\n\n"
                "خودت مستقیم یک کوئری SQL (دقیقاً با همون قوانین و اسکیمای "
                "ابزار tool_sql) بنویس که داده‌ی نمودار رو برگردونه -- معمولاً "
                "یک ستون برچسب/دسته (برای محور X) و یک ستون عددی (برای محور Y)، "
                f"با GROUP BY مناسب. انواع مجاز chart_type: {sorted(VALID_CHART_TYPES)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "کوئری SELECT که داده‌ی نمودار رو برمی‌گردونه (طبق قوانین/اسکیمای tool_sql).",
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": sorted(VALID_CHART_TYPES),
                        "description": "نوع نمودار مناسب سوال کاربر.",
                    },
                    "title": {"type": ["string", "null"]},
                    "x_field": {
                        "type": ["string", "null"],
                        "description": "نام ستونی از نتیجه‌ی sql که باید محور X/برچسب باشه؛ اگه ندی حدس زده می‌شه.",
                    },
                    "y_field": {
                        "type": ["string", "null"],
                        "description": "نام ستونی از نتیجه‌ی sql که باید محور Y/مقدار باشه؛ اگه ندی حدس زده می‌شه.",
                    },
                },
                "required": ["sql", "chart_type"],
            },
        },
    },

    # ============================================================
    # tool_knowledge_base -- غیرفعال تا آماده شدن محتوای پایگاه‌دانش.
    # وقتی knowledge_base_agent.py پیاده‌سازی شد، این بلوک رو از حالت
    # کامنت خارج کن (و همراهش execute_tool_call پایین + قانون مربوطه در
    # main.py::AGENT_SYSTEM_PROMPT).
    # ============================================================
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "tool_knowledge_base",
    #         "description": (
    #             "جست‌وجو در پایگاه‌دانش آموزشی درباره‌ی چطور باید پیشنهاد "
    #             "مدیریتی داد (چارچوب‌ها/اصول تحلیل کسب‌وکار). قبل از دادن "
    #             "هرگونه پیشنهاد یا توصیه‌ی مدیریتی به کاربر، حتماً این ابزار "
    #             "رو صدا بزن تا پیشنهادت رو بر اساس این دانش + دانش عمومی "
    #             "خودت بسازی، نه فقط از حافظه‌ی خودت."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": "موضوع/سوالی که باید در پایگاه‌دانش جست‌وجو بشه.",
    #                 }
    #             },
    #             "required": ["query"],
    #         },
    #     },
    # },
]


def execute_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatcher: اسم ابزار + آرگومان‌های parse‌شده (از JSON خروجی مدل) رو
    می‌گیره، پیاده‌سازی واقعی رو صدا می‌زنه، و همیشه یک dict برمی‌گردونه
    (حتی در خطا -- با کلید "error") تا هیچ‌وقت اجرای گراف با یک exception
    خام قطع نشه؛ خطا باید به‌صورت پیام "tool" به خودِ Agent برسه تا طبق
    سند معماری (حلقه‌ی self-correction) خودش تصمیم بگیره چطور اصلاح کنه.
    """
    try:
        if name == "tool_sql":
            return run_sql_tool(sql=arguments.get("sql", ""))

        if name == "tool_rag":
            return run_rag_tool(
                search_topic=arguments.get("search_topic", ""),
                product_id=arguments.get("product_id"),
            )

        if name == "tool_chart":
            return run_chart_tool(
                sql=arguments.get("sql", ""),
                chart_type=arguments.get("chart_type", "bar"),
                title=arguments.get("title"),
                x_field=arguments.get("x_field"),
                y_field=arguments.get("y_field"),
            )

        # if name == "tool_knowledge_base":
        #     return run_knowledge_base_tool(query=arguments.get("query", ""))

        return {"error": f"ابزار ناشناخته: {name}"}

    except Exception as exc:  # noqa: BLE001 - این‌جا هم آخرین خط دفاعیه
        logger.exception("execute_tool_call: tool '%s' crashed", name)
        return {"error": f"اجرای ابزار '{name}' با خطای غیرمنتظره شکست خورد: {exc}"}
