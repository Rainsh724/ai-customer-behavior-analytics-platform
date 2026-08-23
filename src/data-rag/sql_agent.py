## PATH: app/graph/sql_agent.py
"""
پیاده‌سازی واقعی sql_agent -- جایگزین استاب فعلی در nodes.py.

مسئولیت "Agent" اینجا (که با "LLM" فرق داره -- توضیحش رو در پاسخ چت بخون):
    1. ساخت prompt با context اسکیمای واقعی دیتابیس
    2. صدا زدن LLM برای گرفتن SQL (این تنها جایی‌ست که LLM درگیره)
    3. VALIDATE کردن خروجی LLM قبل از اجرا (LLM قابل‌اعتماد نیست!)
    4. اجرای SQL روی Postgres واقعی (از طریق db.py)
    5. اگر خطا داد یا رد شد -> یک بار خودش رو با پیام خطا اصلاح کنه (self-repair loop)
    6. برگردوندن نتیجه به فرمت GraphState (sql_query, sql_result)

این تابع همون امضای سایر nodeها رو داره: GraphState -> dict[str, Any]
و باید مثل بقیه با @safe_node("sql_agent") در nodes.py دکوریت بشه.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .db import run_readonly_query
from .llm_client import call_llm_json
from .state import GraphState

logger = logging.getLogger(__name__)

# ============================================================
# SCHEMA CONTEXT -- دقیقاً همون جدول/ستون‌هایی که در
# lod_data_to_database2.py پروژه‌ی اصلی بهشون INSERT می‌شه.
# اینو به‌عنوان single source of truth نگه دار؛ اگه اسکیما عوض شد
# فقط همینجا آپدیت کن.
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
         raw_text_normalized TEXT, created_at TIMESTAMPTZ,
         embedded_comment VECTOR)  -- برای RAG استفاده میشه، توی SQL دستکاریش نکن

comment_aspects(aspect_id PK, comment_id FK->comments.id, term TEXT,
                 sentiment TEXT, negative_pct DOUBLE, neutral_pct DOUBLE, positive_pct DOUBLE)
"""

ALLOWED_TABLES = {
    "products", "brands", "categories", "sellers", "users", "cities",
    "sessions", "user_behavior_logs", "comments", "comment_aspects",
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
    # همه‌ی نام‌های جدول استفاده‌شده باید داخل ALLOWED_TABLES باشن
    used_tables = set(re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", stripped, re.IGNORECASE))
    used_tables = {t for pair in used_tables for t in pair if t}
    unknown = used_tables - ALLOWED_TABLES
    if unknown:
        return f"جدول(های) غیرمجاز استفاده شده: {unknown}"
    if "limit" not in stripped.lower() and "count(" not in stripped.lower() and "sum(" not in stripped.lower():
        return "کوئری باید LIMIT داشته باشه (مگر aggregate باشه)."
    return None


def _generate_sql(question: str, prior_error: str | None = None, prior_sql: str | None = None) -> dict[str, Any]:
    user_prompt = f"سوال کاربر: {question}"
    if prior_error and prior_sql:
        user_prompt += (
            f"\n\nتلاش قبلی رد شد. SQL قبلی:\n{prior_sql}\n"
            f"دلیل رد شدن: {prior_error}\nلطفاً اصلاح‌شده رو برگردون."
        )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)


def sql_agent(state: GraphState) -> dict[str, Any]:
    question = state.get("question", "")
    if not question.strip():
        return {"sql_query": "", "sql_result": {}, "errors": ["sql_agent: empty question"]}

    last_sql = ""
    last_error: str | None = None

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        llm_out = _generate_sql(question, prior_error=last_error, prior_sql=last_sql)
        sql = (llm_out.get("sql") or "").strip()
        last_sql = sql

        validation_error = _validate_sql(sql)
        if validation_error:
            last_error = validation_error
            logger.warning("sql_agent: validation attempt %d failed: %s", attempt, validation_error)
            continue

        try:
            rows = run_readonly_query(sql)
            return {
                "sql_query": sql,
                "sql_result": {
                    "rows": rows,
                    "row_count": len(rows),
                    "explanation": llm_out.get("explanation", ""),
                },
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            logger.warning("sql_agent: execution attempt %d failed: %s", attempt, last_error)
            continue

    # همه‌ی تلاش‌ها شکست خورد -- به‌جای crash، یک نتیجه‌ی خالی و خطای واضح برمی‌گردونیم
    return {
        "sql_query": last_sql,
        "sql_result": {},
        "errors": [f"sql_agent: failed after {MAX_REPAIR_ATTEMPTS + 1} attempts: {last_error}"],
    }
