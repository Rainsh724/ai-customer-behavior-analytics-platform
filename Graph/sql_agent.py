## PATH: app/graph/sql_agent.py
"""
پیاده‌سازی واقعی Tool_SQL.

تغییر مهم نسبت به نسخه‌ی قبلی: قبلاً این تابع یک "data_need" (توضیح
فارسی از نیاز) می‌گرفت و خودش با یک تماس LLM جداگانه SQL واقعی رو
می‌ساخت. طبق تصمیمی که گرفتیم، اون لایه‌ی اضافه حذف شد -- الان خودِ
Agent (همون تماس اول، نه یک زیرایجنت پنهان) مستقیماً متن SQL رو در
آرگومان tool_sql می‌نویسه. اسکیما و قوانین امنیتی دیتابیس
(SCHEMA_CONTEXT پایین همین فایل) دیگه در یک system prompt جدا نیست --
مستقیم در description خود ابزار tool_sql قرار می‌گیره (نگاه کن به
tools.py::TOOL_DEFINITIONS)، دقیقاً همون‌جایی که Agent قبل از نوشتن SQL
می‌بینتش.

مسئولیت این تابع الان فقط:
    1. VALIDATE کردن SQL ای که Agent نوشته -- این گیت امنیتی حذف
       نمی‌شه، چون LLM (حتی اگه همون Agent اصلی باشه) قابل‌اعتماد نیست.
    2. اجرای SQL روی Postgres واقعی (از طریق db.py).

اگه SQL نامعتبر بود یا اجرا خطا داد، این تابع دیگه خودش تلاش مجدد
نمی‌کنه -- خطا به‌صورت پیام "tool" به خودِ Agent برمی‌گرده تا طبق سند
معماری ("خطا به LLM برمی‌گردد تا کوئری خود را اصلاح و دوباره ارسال
کند") خودش با یک tool_call جدید و SQL اصلاح‌شده دوباره تلاش کنه. یعنی
حلقه‌ی self-correction کاملاً در همون حلقه‌ی agent<->tools اصلی اتفاق
می‌افته -- در tool_trace هم قابل‌مشاهده و قابل‌ممیزیه، نه یک ریترای
پنهان که کسی نمی‌بینتش.
"""
from __future__ import annotations

import logging
import re
from typing import Any
from .production_validator import ProductionSQLValidator
from .db import run_readonly_query

logger = logging.getLogger(__name__)


# ============================================================
# اعتبارسنج SQL -- ترجیحاً production_validator (پیشرفته‌تر)، ولی اگه
# در دسترس نبود (مثلاً این فایل جابه‌جا/گم شده)، کل برنامه نباید
# ImportError بده و از کار بیفته -- فقط اعتبارسنجی regex-based داخلی
# (پایین همین فایل) به‌عنوان fallback فعال می‌شه.
# ============================================================
try:
    sql_validator = ProductionSQLValidator(dialect="postgres")
except ImportError:
    logger.warning(
        "production_validator در دسترس نیست -- fallback به اعتبارسنج "
        "داخلی (_validate_sql، پایین همین فایل) شد. برای اعتبارسنجی "
        "کامل‌تر، production_validator.py رو در PYTHONPATH قرار بدید."
    )
    sql_validator = None

# ============================================================
# SCHEMA CONTEXT -- دقیقاً همون جدول/ستون‌هایی که در
# lod_data_to_database2.py پروژه‌ی اصلی بهشون INSERT می‌شه.
#
# این متن مستقیماً در description ابزار tool_sql تزریق می‌شه (نگاه کن
# به tools.py) تا Agent در همون اولین و تنها تماسش بتونه SQL معتبر
# بنویسه -- دیگه هیچ تماس LLM دومی برای "ترجمه‌ی نیاز به SQL" نداریم.
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

comments_embedding(id BIGINT PK/FK->comments.id, embedded_comment VECTOR(768))
                    -- فقط برای RAG/similarity search؛ برای tool_sql ازش استفاده نکن.

comment_aspects(aspect_id PK, comment_id FK->comments.id, term TEXT,
                 sentiment TEXT, negative_pct DOUBLE, neutral_pct DOUBLE, positive_pct DOUBLE)
                 -- برای شمارش/آمار جنبه‌ها قابل‌استفاده‌ست؛ برای *خوندن متن*
                 -- نظرات و جست‌وجوی معنایی، اون کار tool_rag است نه tool_sql.


-- ==========================================
-- جداول تحلیلی و هوشمند (AI & Analytics)
-- ==========================================
analytics.feature_user(user_id BIGINT PK/FK->users.user_id, total_spend BIGINT, total_purchases INT, 
                       total_views INT, active_days INT, category_diversity INT, 
                       avg_session_duration_minutes DOUBLE PRECISION, 
                       night_activity_ratio DOUBLE PRECISION, weekend_activity_ratio DOUBLE PRECISION)

kpi.rfm_segments(user_id BIGINT PK/FK->users.user_id, recency_days INT, frequency INT, 
                 monetary BIGINT, rfm_code TEXT, rfm_label TEXT)
                 -- مقادیر rfm_label شامل: 'vip', 'promising', 'at_risk', 'lost', 'regular'

kpi.ml_user_clusters(user_id BIGINT PK/FK->users.user_id, cluster_id INT, cluster_name TEXT)
                     -- مقادیر cluster_name شامل: 'vip_champions', 'night_weekend_buyers', 'active_loyals', 'low_intent_shoppers', 'churned_customers'
"""

# اضافه کردن جداول تحلیلی و نام اسکیماها به لیست سفید (Whitelist)
ALLOWED_TABLES = {
    "products", "brands", "categories", "sellers", "users", "cities",
    "sessions", "user_behavior_logs", "comments", "comment_aspects",
    "comments_embedding",
    # --- اضافه‌شده‌های جدید ---
    "feature_user", "rfm_segments", "ml_user_clusters",
    "analytics.feature_user", "kpi.rfm_segments", "kpi.ml_user_clusters",
    "analytics", "kpi"  # اضافه کردن نام اسکیماها تا پارسر امنیتی به آن‌ها گیر ندهد
}


FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|COPY|"
    r"CALL|EXECUTE|MERGE|VACUUM|ATTACH|DETACH|--|;.*\S)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> str | None:
    """برمی‌گردونه: پیام خطا اگه SQL مشکل داره، وگرنه None. این تنها گیت
    امنیتی‌ست که بعد از حذف تماس LLM داخلی باقی مونده -- عمداً حذف
    نشده، چون حتی وقتی خودِ Agent اصلی SQL می‌نویسه، نباید بدون بررسی
    مستقیم روی دیتابیس اجرا بشه."""
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



# سقف سخت تعداد ردیف برگشتی -- مستقل از اینکه Agent در متن SQL خودش
# LIMIT گذاشته باشه یا نه. production_validator.py (که در اولویته)
# اصلاً چک LIMIT نداره، پس بدون این سقف کدی، یک SELECT بی‌LIMIT (مثلاً
# ILIKE روی یک کلمه‌ی پرتکرار) می‌تونه صدها ردیف خام رو مستقیم وارد
# state["messages"] کنه و در تماس بعدی LLM با خطای context-length کرش
# کنه -- دقیقاً همون اتفاقی که افتاد. این یک محافظ defense-in-depth
# در سطح کد است، نه فقط یک دستورالعمل متنی به مدل.
HARD_MAX_ROWS = 200


def run_sql_tool(sql: str) -> dict[str, Any]:
    """
    ورودی: متن SQL که خودِ Agent در آرگومان tool_sql نوشته.
    خروجی: dict که مستقیم به‌صورت JSON در پیام "tool" به Agent برمی‌گرده؛
    اگه کلید "error" داشته باشه، Agent خودش با یک tool_call جدید و SQL اصلاح‌شده دوباره تلاش می‌کنه.
    """
    if not sql or not sql.strip():
        return {"error": "sql خالی بود."}

    # اعتبارسنجی دقیق و ساختاری کوئری -- production_validator اگه در
    # دسترس بود، وگرنه fallback به اعتبارسنج داخلی (regex-based).
    if sql_validator is not None:
        is_valid, validation_errors = sql_validator.validate(sql)
        if not is_valid:
            error_message = " | ".join(validation_errors)
            return {
                "error": f"SQL نامعتبر (خطای ساختاری/کاردینالیتی): {error_message}",
                "rejected_sql": sql
            }
    else:
        validation_error = _validate_sql(sql)
        if validation_error:
            return {"error": f"SQL نامعتبر: {validation_error}", "rejected_sql": sql}

    try:
        rows = run_readonly_query(sql)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_sql_tool: execution failed: %s", exc)
        return {"error": f"اجرای SQL شکست خورد: {exc}", "rejected_sql": sql}

    total_row_count = len(rows)
    truncated = total_row_count > HARD_MAX_ROWS
    if truncated:
        logger.warning(
            "run_sql_tool: %d ردیف برگشت، به %d ردیف بریده شد (SQL بدون LIMIT کافی): %s",
            total_row_count, HARD_MAX_ROWS, sql,
        )
        rows = rows[:HARD_MAX_ROWS]

    result: dict[str, Any] = {
        "sql_query": sql,
        "rows": rows,
        "row_count": len(rows),
    }
    if truncated:
        result["truncated"] = True
        result["total_row_count_before_truncation"] = total_row_count
        result["note"] = (
            f"نتیجه {total_row_count} ردیف داشت و به {HARD_MAX_ROWS} ردیف اول بریده شد -- "
            "اگه برای جواب نیاز به کل داده داری، کوئری رو با LIMIT/GROUP BY/aggregate مناسب‌تر دوباره بنویس."
        )
    return result





















