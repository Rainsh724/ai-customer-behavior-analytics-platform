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
from .dataset_time import get_reference_date

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


def refresh_sql_validator_from_db() -> None:
    """
    این رو صریحاً از main.py در startup صدا بزنید.
    schema/join-key ها رو زنده از information_schema می‌خونه و جایگزین sql_validator فعلی می‌کنه.
    """
    global sql_validator
    from .db import get_conn

    with get_conn() as conn:
        sql_validator = ProductionSQLValidator.from_database(conn, dialect="postgres")
    logger.info("sql_validator: schema از information_schema بازخوانی شد.")
# SCHEMA CONTEXT -- دقیقاً همون جدول/ستون‌هایی که در
# lod_data_to_database2.py پروژه‌ی اصلی بهشون INSERT می‌شه.
#
# این متن مستقیماً در description ابزار tool_sql تزریق می‌شه (نگاه کن
# به tools.py) تا Agent در همون اولین و تنها تماسش بتونه SQL معتبر
# بنویسه -- دیگه هیچ تماس LLM دومی برای "ترجمه‌ی نیاز به SQL" نداریم.
# ============================================================

SCHEMA_CONTEXT = """
Allowed tables and columns:

-- 1. Core Catalog & Organization:
products(id BIGINT PK, title_fa TEXT, brand_id INT FK->brands.brand_id,
         category_id INT FK->categories.category_id, seller_id INT FK->sellers.seller_id,
         price BIGINT, min_price_last_month BIGINT, is_fake BOOLEAN,
         rate DOUBLE PRECISION, rate_cnt BIGINT)
brands(brand_id INT PK, name TEXT)
categories(category_id INT PK, category1 TEXT, category2 TEXT, sub_category TEXT)
sellers(seller_id INT PK, seller_title TEXT)
cities(city_id INT PK, name TEXT)
users(user_id BIGINT PK)
sessions(session_id TEXT PK, user_id BIGINT FK->users.user_id, city_id INT FK->cities.city_id)

-- 2. Real-time Events & Dynamic Time-windowed Analytics:
-- USE FOR: Any question asking for dynamic dates/rolling windows (e.g. 'last 7 days', 'last 30 days', trends over time).
-- NOTE: Each record with event_type='purchase' is 1 purchase event. Total sales volume = COUNT(*).
user_behavior_logs(log_id BIGINT PK, session_id TEXT FK->sessions.session_id,
                   product_id BIGINT FK->products.id, event_type TEXT,
                   timestamp TIMESTAMPTZ)
                   -- event_type in ('view', 'add_to_cart', 'purchase', 'remove_from_cart')

-- 3. Customer Reviews & Aspect Sentiment:
comments(id BIGINT PK, product_id BIGINT FK->products.id, is_buyer BOOLEAN,
         rate DOUBLE PRECISION, recommendation_status TEXT, likes INT, dislikes INT,
         created_at TIMESTAMPTZ)
comment_aspects(aspect_id INT PK, comment_id BIGINT FK->comments.id, term TEXT,
                sentiment TEXT, negative_pct DOUBLE PRECISION, neutral_pct DOUBLE PRECISION, positive_pct DOUBLE PRECISION)

-- 4. Pre-computed KPIs & ML Clusters (FAST & HIGH-ACCURACY - Use for general/all-time/segmentation queries):
-- Customer Personas & Segmentation:
kpi.ml_user_clusters(user_id BIGINT PK/FK->users.user_id, cluster_id INT, cluster_name TEXT)
  -- cluster_name values:
  --   'vip_champions' (مشتریان بسیار سودآور و پرخرید / VIP)
  --   'night_weekend_buyers' (خریداران شب و روزهای تعطیل / تخفیف‌محور)
  --   'active_loyals' (مشتریان وفادار با خریدهای منظم و فعالیت مداوم)
  --   'low_intent_shoppers' (بازدیدکنندگان با قصد خرید پایین / چرخ‌زنندگان)
  --   'churned_customers' (مشتریان ریزشی که مدت‌هاست غیرفعال‌اند)

kpi.rfm_segments(user_id BIGINT PK/FK->users.user_id, recency_days NUMERIC, frequency BIGINT, 
                 monetary DOUBLE PRECISION, rfm_code TEXT, rfm_label TEXT)
                 -- rfm_label values: 'vip', 'promising', 'at_risk', 'lost', 'regular'

analytics.feature_user(user_id BIGINT PK/FK->users.user_id, total_spend BIGINT, total_purchases INT, 
                       total_views INT, active_days INT, avg_purchase_value BIGINT,
                       night_activity_ratio DOUBLE PRECISION, weekend_activity_ratio DOUBLE PRECISION, category_diversity INT)

-- Product, Brand & Funnel 360 Aggregates:
product_negative_feedback_summary(product_id BIGINT PK/FK->products.id, avg_negative_pct DOUBLE PRECISION, comment_cnt BIGINT)
  -- Precomputed product-level average negative feedback. Always use this instead of joining comments with comment_aspects for product-level negative %.

kpi.product_360(product_id BIGINT PK/FK->products.id, title_fa TEXT, price BIGINT, 
                total_views BIGINT, total_purchases BIGINT, total_revenue BIGINT, 
                conversion_rate NUMERIC, comment_count BIGINT, star_rating NUMERIC, 
                positive_sentiment_pct NUMERIC, sentiment_score NUMERIC, managerial_action_tag TEXT)
  -- managerial_action_tag values: 'Hero Product (قهرمان)', 'High Traffic, Low Conversion (نیازمند بررسی قیمت)', 
  --                               'High Risk (فروش بالا اما به شدت ناراضی)', 'Needs Reviews (نیازمند کمپین ثبت نظر)', 'Normal'

kpi.brand_diagnostics(brand_id INT PK/FK->brands.brand_id, brand_name TEXT, 
                      total_views BIGINT, total_purchases BIGINT, total_comments BIGINT, 
                      avg_rating NUMERIC, brand_sentiment_score NUMERIC)

kpi.aspect_diagnostics(aspect_name TEXT, total_mentions BIGINT, positive_mentions BIGINT, 
                       negative_mentions BIGINT, negative_impact_pct NUMERIC, aspect_status TEXT)
  -- aspect_status values: 'Critical Weakness (نقطه ضعف بحرانی)', 'Key Strength (نقطه قوت کلیدی)', 'Neutral'

kpi.global_funnel(total_views NUMERIC, total_carts NUMERIC, total_purchases NUMERIC, total_removes NUMERIC, 
                  view_to_cart_pct NUMERIC, cart_to_purchase_pct NUMERIC, overall_conversion_pct NUMERIC, cart_abandonment_pct NUMERIC)

-- ==========================================
-- REALISTIC DATASET BENCHMARKS & SCALE (Sample Environment):
-- ==========================================
-- * Product Views: Max views in this dataset is 33 (avg: 3.8, p90: 7, p99: 12).
--   For "high traffic" products, use total_views >= 8 or use managerial_action_tag.
--   NEVER filter total_views > 20 or > 50 or > 100 -- it will return 0 rows!
-- * Product Purchases: Max purchases is 9 (avg: 1.3, p90: 2).
--   Top sellers are products with total_purchases >= 2 or >= 3. Never filter purchases > 10!
-- * Bundles / Co-purchases: Most product pairs are co-purchased 1 time per session.
--   Always use HAVING COUNT(*) >= 1 (never >= 2).
-- * Active Date Range: 2019-10-01 to 2023-03-01. Reference date is 2023-03-01.

-- ==========================================
-- Important PostgreSQL note: the ROUND function
-- ==========================================
-- ROUND(double precision, integer) does not exist in PostgreSQL -- only
-- ROUND(numeric, integer) is supported. Any column that is DOUBLE PRECISION
-- must be cast to numeric before rounding: ROUND(expr::numeric, 2)
"""


# اضافه کردن جداول تحلیلی و نام اسکیماها به لیست سفید (Whitelist)
ALLOWED_TABLES = {
    "products",
    "brands",
    "categories",
    "sellers",
    "users",
    "cities",
    "sessions",
    "user_behavior_logs",
    "comments",
    "comment_aspects",
    "product_negative_feedback_summary",

    # analytics schema
    "analytics",
    "analytics.feature_user",

    # kpi schema
    "kpi",
    "kpi.rfm_segments",
    "kpi.ml_user_clusters",
    "kpi.product_360",
    "kpi.brand_diagnostics",
    "kpi.aspect_diagnostics",
    "kpi.global_funnel",
    "kpi.user_segments",

    # unqualified table names
    "feature_user",
    "rfm_segments",
    "ml_user_clusters",
    "product_360",
    "brand_diagnostics",
    "aspect_diagnostics",
    "global_funnel",
    "user_segments",
}


FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|COPY|"
    r"CALL|EXECUTE|MERGE|VACUUM|ATTACH|DETACH|--|;.*\S)\b",
    re.IGNORECASE,
)

# اگه تاریخ مرجع لفظی (مثلاً '2023-03-01') مستقیم به‌عنوان کران بالای
# یک شرط "<" استفاده بشه، بدون + INTERVAL بعدش، یعنی خودِ روز مرجع از
# بازه حذف می‌شه (چون ستون timestamp ساعت هم داره). این دقیقاً همون
# چیزیه که باعث می‌شه دو کوئری مختلف برای "همین بازه" (مثلاً tool_sql
# و tool_chart) دو تا کران پایانی متفاوت داشته باشن -- حتی اگه هردو از
# instruction متنی پیروی نکنن. برای این‌که این تناقض دیگه اصلاً وابسته
# به این نباشه که مدل instruction رو دقیق رعایت کنه یا نه، این الگو در
# سطح کد رد می‌شه.
def _reference_date_boundary_pattern() -> re.Pattern[str]:
    ref = re.escape(get_reference_date().isoformat())
    return re.compile(
        r"<\s*'" + ref + r"'(?:\s*::\s*date)?(?!\s*\+\s*INTERVAL)",
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

    # این چک مستقل از این‌که کدوم validator (production_validator یا
    # fallback) فعاله همیشه اجرا می‌شه -- چون production_validator یه
    # AST-validator عمومیه و این قانونِ خاصِ پروژه (تاریخ مرجع) رو
    # نمی‌شناسه.
    boundary_error = None
    if _reference_date_boundary_pattern().search(sql):
        boundary_error = (
            "کران پایانی بازه بدون INTERVAL -- تاریخ مرجع مستقیم به‌عنوان "
            "کران بالای '<' استفاده شده بدون + INTERVAL '1 day'، که یعنی "
            "خودِ روز مرجع کامل از بازه حذف می‌شه. اگه بازه باید تا خودِ "
            "تاریخ مرجع (شامل همون روز) رو بپوشونه، بنویس: "
            "'<تاریخ مرجع>'::date + INTERVAL '1 day'."
        )
    if boundary_error:
        return {"error": f"SQL نامعتبر: {boundary_error}", "rejected_sql": sql}

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