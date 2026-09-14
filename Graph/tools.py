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
# تا وقتی نسخه‌ی واقعی run_knowledge_base_tool توسط بقیه‌ی اعضا آماده بشه،
# از پلیس‌هولدر موقت استفاده می‌کنیم تا گراف قابل دیباگ باشه.
# TODO: وقتی نسخه‌ی واقعی آماده شد، این خط رو به
#   from .knowledge_base_agent import run_knowledge_base_tool
# تغییر بده و در execute_tool_call پایین هم فراخوانی رو عوض کن.
from .knowledge_base_agent import run_knowledge_base_tool_debug_placeholder as run_knowledge_base_tool

logger = logging.getLogger(__name__)


TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "tool_sql",
            "description": (
                "Fetches structured, numeric data (sales, prices, comment "
                "counts, KPI stats) from the Postgres database. Write a "
                "valid SQL query yourself directly (SELECT or "
                "WITH...SELECT only) based on the schema below -- no "
                "intermediate step builds this query for you.\n\n"
                "Mandatory rules:\n"
                "- SELECT/WITH only; never write INSERT/UPDATE/DELETE/DDL.\n"
                "- Only use tables/columns from the schema below.\n"
                "- Always include LIMIT (max 200) unless it's an "
                "aggregate/COUNT query.\n"
                "- Only one query; never separate multiple statements "
                "with ;.\n"
                "- This dataset is not real-time -- for \"today\", never "
                "use Postgres's real NOW()/CURRENT_DATE. Instead use the "
                "reference date given to you in the conversation's system "
                "message as the dataset's \"today\" (e.g. instead of "
                "NOW() - INTERVAL '30 days', write "
                "'<reference date>'::date - INTERVAL '30 days').\n"
                "- When filtering up to the reference date (upper boundary e.g. timestamp < ...), "
                "always write '<reference date>'::date + INTERVAL '1 day' to include the full 24 hours of that reference day.\n"
                "- Do not use this tool to read review text or do "
                "semantic search -- that's tool_rag's job.\n\n"
                """
                Sales-specific rules:
 
                - In user_behavior_logs, each record with event_type='purchase' is one purchase event.
                - The columns quantity and order_id do not exist in this table.
                - A product's sales/purchase count must be computed with COUNT(*) over purchase events.
                - "Best-selling" defaults to ranking by units_sold.
                - Only compute a monetary amount if the user explicitly asks about sales amount/revenue.
                - products.price is the product's current price, so price * purchase_count
                is only an estimated_sales figure, not necessarily real historical revenue.
                - When joining products with user_behavior_logs, don't run SUM/AVG directly on
                products columns like price; first aggregate the child table in a CTE
                and only then JOIN to products.
 
                - Deterministic sort: Whenever you use ORDER BY with LIMIT, you can include an id column as secondary tie-breaker (e.g. ORDER BY total_views DESC, product_id ASC).
                - Entity names: When analyzing or grouping by brands, categories, or products, always SELECT their name/title alongside their ID (e.g. b.name AS brand_name, cat.category2 AS category_name, p.title_fa AS product_title) so reports contain actual entity names instead of raw IDs.
                - Realistic Data Scale: In this dataset, maximum product views is 33 (avg: 3.8, p90: 7). Top 10% high-traffic products have views >= 8. NEVER filter total_views > 20 or > 50 or > 100!
                - For high-traffic low-conversion products, underperforming products, or products with untapped sales potential, query kpi.product_360 where managerial_action_tag = 'High Traffic, Low Conversion (نیازمند بررسی قیمت)' or total_views >= 8.
                - Best-sellers scale: Maximum product purchases is 9. Top-sellers have total_purchases >= 2 or >= 3. Never filter purchases > 10!
                - For co-purchased products (bundles/market basket), self-join user_behavior_logs on session_id where event_type='purchase' and l1.product_id < l2.product_id. Always use HAVING COUNT(*) >= 1 (never >= 2).
                - For customer purchase journey / repurchase intervals, analyze user_behavior_logs using timestamp per user_id/session_id.
                - For customer churn, at-risk customers, or segmentation, query kpi.ml_user_clusters (where cluster_name = 'churned_customers') or kpi.user_segments.
 
 
                Rules for working with large tables:
 
                - The comments, comment_aspects, and user_behavior_logs tables have millions of rows.
                - Never run SELECT * on these tables.
                - Before joining these tables, first reduce the data volume with WHERE, LIMIT, or aggregation.
                - Use COUNT, SUM, AVG, and GROUP BY for statistical analysis.
                - Avoid joining several large tables without a time filter or a limiting condition.
                """
                f"Database schema:\n{SCHEMA_CONTEXT}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The full SQL query text (SELECT/WITH) that you wrote yourself based on the schema above.",
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
                "Semantic search over customer review text to find "
                "qualitative evidence about a topic.\n\n"
                "Filter rules:\n"
                "- If product_id is specified, only reviews for that exact product count as valid evidence.\n"
                "- If brand_id is specified, reviews across all products belonging to that brand are searched.\n"
                "- If category_id is specified, reviews across all products belonging to that category are searched.\n\n"
                "If a specific filter is given and hit_count=0, you must not drop the filter or attribute "
                "results from other products/brands to this one. State explicitly that no evidence was found.\n\n"
                "search_topic must reflect the direction of the question; "
                "e.g. for an increase in sales, use 'reasons for "
                "satisfaction and positive reception', and "
                "for a complaint, use 'reasons for dissatisfaction and "
                "complaints'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_topic": {
                        "type": "string",
                        "description": (
                            "The topic/question to search for in the "
                            "reviews -- to find reasons for "
                            "dissatisfaction, phrase the topic in that "
                            "same direction (e.g. 'reasons for "
                            "dissatisfaction and complaints about brand/product "
                            "X'), not just the name alone."
                        ),
                    },
                    "product_id": {
                        "type": ["integer", "null"],
                        "description": "Optional: Filter comments for a single specific product_id.",
                    },
                    "brand_id": {
                        "type": ["integer", "null"],
                        "description": "Optional: Filter comments for all products belonging to this brand_id.",
                    },
                    "category_id": {
                        "type": ["integer", "null"],
                        "description": "Optional: Filter comments for all products belonging to this category_id.",
                    },
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
                "When the user explicitly asks for a chart/graph/"
                "dashboard/visualization, builds a real chart from the "
                "database data which the frontend renders automatically. "
                "Do NOT include JSON configs, code blocks, or raw chart objects in your final response -- the frontend renders the chart graphically from tool data. Only provide managerial text analysis.\n\n"
                "Write a SQL query yourself directly (following exactly "
                "the same rules and schema as tool_sql) that returns the "
                "chart's data -- usually one label/category column (for "
                "the X axis) and one numeric column (for the Y axis), "
                f"with an appropriate GROUP BY. Allowed chart_type values: {sorted(VALID_CHART_TYPES)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SELECT query that returns the chart data (following tool_sql's rules/schema).",
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": sorted(VALID_CHART_TYPES),
                        "description": "Chart type matching the user's question.",
                    },
                    "title": {"type": ["string", "null"]},
                    "x_field": {
                        "type": ["string", "null"],
                        "description": "Column name from the sql result that should be the X axis/label; guessed if omitted.",
                    },
                    "y_field": {
                        "type": ["string", "null"],
                        "description": "Column name from the sql result that should be the Y axis/value; guessed if omitted.",
                    },
                },
                "required": ["sql", "chart_type"],
            },
        },
    },
 
    # tool_knowledge_base -- disabled until knowledge-base content is ready.
    {
        "type": "function",
        "function": {
            "name": "tool_knowledge_base",
            "description": (
                "Searches the training knowledge base for how to give "
                "managerial suggestions (business-analysis "
                "frameworks/principles). Before giving any suggestion or "
                "managerial recommendation to the user, you must always "
                "call this tool first, so you build your suggestion from "
                "this knowledge plus your own general knowledge, not from "
                "memory alone.\n\n"
                "\u26a0\ufe0f A placeholder/debug version (a general, "
                "generic summary) is currently active, not real vector "
                "search -- until the rest of the team builds the final "
                "version."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic/question to search for in the knowledge base.",
                    }
                },
                "required": ["query"],
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
    سند معماری (حلقه‌ی self-correction) خودش تصمیم بگیره چطور اصلاح کنه.
    """
    try:
        if name == "tool_sql":
            return run_sql_tool(sql=arguments.get("sql", ""))

        if name == "tool_rag":
            return run_rag_tool(
                search_topic=arguments.get("search_topic", ""),
                product_id=arguments.get("product_id"),
                brand_id=arguments.get("brand_id"),
                category_id=arguments.get("category_id"),
            )

        if name == "tool_chart":
            return run_chart_tool(
                sql=arguments.get("sql", ""),
                chart_type=arguments.get("chart_type", "bar"),
                title=arguments.get("title"),
                x_field=arguments.get("x_field"),
                y_field=arguments.get("y_field"),
            )

        if name == "tool_knowledge_base":
            return run_knowledge_base_tool(query=arguments.get("query", ""))

        return {"error": f"ابزار ناشناخته: {name}"}

    except Exception as exc:  # noqa: BLE001 - این‌جا هم آخرین خط دفاعیه
        logger.exception("execute_tool_call: tool '%s' crashed", name)
        return {"error": f"اجرای ابزار '{name}' با خطای غیرمنتظره شکست خورد: {exc}"}
