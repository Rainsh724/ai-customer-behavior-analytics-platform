## PATH: app/graph/sql_agent.py
"""
پیاده‌سازی واقعی Tool_SQL.

تفاوت با نسخه‌ی قبلی (پایپ‌لاین): قبلاً sql_agent یک نود گراف بود که کل
`state["question"]` رو می‌گرفت. الان این یک تابع ابزارِ معمولی‌ست که
Agent (نود مرکزی) صداش می‌زنه و بهش یک "data_need" مشخص و کوچیک می‌ده
(نه لزوماً کل سوال کاربر -- ممکنه Agent یک زیرسوال دقیق‌تر بسازه، مثلاً
در سناریوی استدلال چندمرحله‌ای اول یک data_need برای "روند فروش" می‌سازه).

مسئولیت این تابع («Agent» به‌معنای کلاسیک، نه LLM):
    1. ساخت prompt با context اسکیمای واقعی دیتابیس
    2. صدا زدن LLM برای گرفتن SQL
    3. VALIDATE کردن خروجی LLM قبل از اجرا (LLM قابل‌اعتماد نیست!)
    4. اجرای SQL روی Postgres واقعی (از طریق db.py)
    5. اگر خطا داد -> خودش یک بار (repair loop سبک، فقط برای خطای
       نحوی/اسکیمایی) تلاش می‌کنه اصلاح کنه؛ اگه بازم شکست خورد، خطای
       واقعی رو برمی‌گردونه تا خودِ Agent (LLM بالادست) در چرخه‌ی
       self-correction سند معماری تصمیم بگیره -- دوباره با data_need
       متفاوت صدا بزنه، یا از کاربر توضیح بیشتر بخواد.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .db import run_readonly_query
from .llm_client import call_llm_json

logger = logging.getLogger(__name__)

# ============================================================
# SCHEMA CONTEXT -- دقیقاً همون جدول/ستون‌هایی که در
# lod_data_to_database2.py پروژه‌ی اصلی بهشون INSERT می‌شه.
# ============================================================

SCHEMA_CONTEXT = """
جداول مجاز (فقط از همین‌ها و همین ستون‌ها استفاده کن):

products(id BIGINT PK, title_fa TEXT, brand_id INT FK->brands.brand_id,
         category_id INT FK->categories.category_id, seller_id INT FK->sellers.seller_id,
         price BIGINT, min_price_last_month BIGINT, is_fake BOOLEAN,
         rate DOUBLE PRECISION, rate_cnt BIGINT)

brands(brand_id PK, name TEXT)
categories(category_id PK, category1 TEXT, category2 TEXT, sub_category TEXT)
sellers(seller_id PK, seller_title TEXT)

users(user_id BIGINT PK)
cities(city_id PK, name TEXT)
sessions(session_id TEXT PK, user_id FK->users.user_id, city_id FK->cities.city_id)

user_behavior_logs(log_id PK, session_id FK->sessions.session_id,
                    product_id FK->products.id, event_type TEXT,  -- مثل 'view','add_to_cart','purchase'
                    timestamp TIMESTAMPTZ)

comments(id BIGINT PK, product_id FK->products.id, is_buyer BOOLEAN,
         rate DOUBLE PRECISION, recommendation_status TEXT, likes INT, dislikes INT,
         raw_text_normalized TEXT, created_at TIMESTAMPTZ)
         -- توجه: embedding کامنت‌ها این‌جا نیست، جدول جدای comments_embedding را ببین.

comments_embedding(id BIGINT PK/FK->comments.id, embedded_comment VECTOR(768))
                    -- این جدول فقط برای RAG/similarity search استفاده می‌شه.

comment_aspects(aspect_id PK, comment_id FK->comments.id, term TEXT,
                 sentiment TEXT, negative_pct DOUBLE, neutral_pct DOUBLE, positive_pct DOUBLE)
"""

ALLOWED_TABLES = {
    "products", "brands", "categories", "sellers", "users", "cities",
    "sessions", "user_behavior_logs", "comments", "comment_aspects",
    "comments_embedding",
}

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|COPY|"
    r"CALL|EXECUTE|MERGE|VACUUM|ATTACH|DETACH|--|;.*\S)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = f"""
تو یک مولد کوئری Postgres هستی. فقط و فقط یک شیء JSON با این فرمت برگردون:
{{"sql": "<یک کوئری SELECT معتبر>", "explanation": "<توضیح کوتاه فارسی>"}}

قوانین سخت‌گیرانه:
- فقط SELECT (یا WITH ... SELECT). هیچ‌وقت INSERT/UPDATE/DELETE/DDL ننویس.
- فقط از جداول/ستون‌های زیر استفاده کن، هیچ جدول فرضی نساز.
- همیشه یک LIMIT (حداکثر 200) بذار مگر این‌که aggregate/COUNT باشه.
- از placeholder یا چندین statement (جداشده با ;) استفاده نکن -- فقط یک کوئری.
- تاریخ‌ها را با NOW() / INTERVAL بساز، هاردکد نکن مگر کاربر تاریخ دقیق داده.

اسکیمای دیتابیس:
{SCHEMA_CONTEXT}
"""

MAX_REPAIR_ATTEMPTS = 2


def _validate_sql(sql: str) -> str | None:
    """برمی‌گردونه: پیام خطا اگه SQL مشکل داره، وگرنه None."""
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return "کوئری باید با SELECT یا WITH شروع بشه."
    if FORBIDDEN_KEYWORDS.search(stripped):
        return "کوئری شامل کلمات/الگوهای غیرمجاز است (DDL/DML یا چند statement)."
    used_tables = set(re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", stripped, re.IGNORECASE))
    used_tables = {t for pair in used_tables for t in pair if t}
    unknown = used_tables - ALLOWED_TABLES
    if unknown:
        return f"جدول(های) غیرمجاز استفاده شده: {unknown}"
    if "limit" not in stripped.lower() and "count(" not in stripped.lower() and "sum(" not in stripped.lower():
        return "کوئری باید LIMIT داشته باشه (مگر aggregate باشه)."
    return None


def _generate_sql(data_need: str, prior_error: str | None = None, prior_sql: str | None = None) -> dict[str, Any]:
    user_prompt = f"داده‌ی موردنیاز: {data_need}"
    if prior_error and prior_sql:
        user_prompt += (
            f"\n\nتلاش قبلی رد شد. SQL قبلی:\n{prior_sql}\n"
            f"دلیل رد شدن: {prior_error}\nلطفاً اصلاح‌شده رو برگردون."
        )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)


def run_sql_tool(data_need: str) -> dict[str, Any]:
    """
    ورودی: توضیح فارسی و مشخصِ داده‌ی موردنیاز (چیزی که Agent در آرگومان
    tool_sql فرستاده -- نگاه کن به tools.py).
    خروجی: dict که مستقیم به‌صورت JSON در پیام "tool" به Agent برمی‌گرده؛
    اگه کلید "error" داشته باشه، Agent می‌فهمه شکست خورده و می‌تونه
    (طبق سند معماری) دوباره با data_need اصلاح‌شده صدا بزنه.
    """
    if not data_need or not data_need.strip():
        return {"error": "data_need خالی بود."}

    last_sql = ""
    last_error: str | None = None

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        llm_out = _generate_sql(data_need, prior_error=last_error, prior_sql=last_sql)
        sql = (llm_out.get("sql") or "").strip()
        last_sql = sql

        validation_error = _validate_sql(sql)
        if validation_error:
            last_error = validation_error
            logger.warning("run_sql_tool: validation attempt %d failed: %s", attempt, validation_error)
            continue

        try:
            rows = run_readonly_query(sql)
            return {
                "sql_query": sql,
                "rows": rows,
                "row_count": len(rows),
                "explanation": llm_out.get("explanation", ""),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            logger.warning("run_sql_tool: execution attempt %d failed: %s", attempt, last_error)
            continue

    # حتی بعد از تلاش‌های داخلی هم شکست خورد -- این خطا رو (نه یک
    # exception) برمی‌گردونیم تا به‌صورت پیام "tool" به خودِ Agent برسه.
    return {
        "error": f"اجرای SQL بعد از {MAX_REPAIR_ATTEMPTS + 1} تلاش شکست خورد: {last_error}",
        "last_attempted_sql": last_sql,
    }
