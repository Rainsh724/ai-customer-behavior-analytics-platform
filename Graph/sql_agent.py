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

product_negative_feedback_summary(product_id BIGINT PK/FK->products.id,
                 avg_negative_pct DOUBLE, comment_cnt BIGINT)
                 -- یک جدول خلاصه‌ی از پیش محاسبه‌شده در سطح محصول است.
                 -- هر وقت نیاز به میانگین درصد بازخورد منفی (negative_pct)
                 -- در سطح یک محصول (نه تک‌تک نظرات) داری، همیشه از همین
                 -- جدول بخوان -- هرگز مستقیم comments را با comment_aspects
                 -- JOIN نکن تا این آمار را دوباره از صفر محاسبه کنی؛ آن
                 -- JOIN روی کل دیتاست بسیار کند است (میلیون‌ها ردیف) و این
                 -- جدول همان نتیجه را از پیش محاسبه کرده. توجه: این جدول
                 -- periodic رفرش می‌شود، پس ممکن است چند ساعت/روز قدیمی
                 -- باشد -- برای تحلیل‌های سطح-محصول/گزارش‌گیری کافی است.

-- ==========================================
-- جداول تحلیلی و هوشمند (AI & Analytics)
-- ==========================================
analytics.feature_user(user_id BIGINT PK/FK->users.user_id, total_spend BIGINT, total_purchases INT, 
                       total_views INT, active_days INT, category_diversity INT, 
                       avg_session_duration_minutes DOUBLE PRECISION, 
                       night_activity_ratio DOUBLE PRECISION, weekend_activity_ratio DOUBLE PRECISION,
                       total_events,total_sessions , total_cart_adds , total_removes , avg_session_events , max_session_events,
                       max_session_duration_minutes,morning_activity_ratio , afternoon_activity_ratio , evening_activity_ratio,
                       preferred_hour, preferred_weekday , unique_products_viewed ,unique_products_purchased,
                       cities_visited,avg_purchase_value , min_purchase_price , max_purchase_price , purchase_frequency , 
                       purchase_days , brand_diversity)
analytics.feature_behavior(log_id, hour, day, month, weekday, is_weekend, is_view, is_cart, is_remove, is_purchase)                       
analytics.feature_product(product_id, total_events, total_views, total_cart_adds, total_removes, total_purchases,unique_viewers, unique_carters, unique_buyers, total_sessions, price_drop_ratio)
analytics.feature_city(city_id, total_users, total_sessions, total_events, total_views, total_cart_adds, total_purchases, total_removes, unique_products_viewed, unique_products_purchased)
analytics.feature_category(category_id, total_events, total_views, total_cart_adds, total_purchases, total_removes, unique_viewers, unique_buyers, avg_product_price)
analytics.feature_brand(brand_id, total_events, total_views, total_cart_adds, total_purchases, total_removes, unique_viewers, unique_buyers)
analytics.feature_user_product(user_id, product_id, total_events, view_count, cart_count, remove_count, purchase_count, active_days, session_count)
analytics.feature_user_category(user_id, category_id, total_events, view_count, cart_count, remove_count, purchase_count, category_spend, view_share, purchase_share, spend_share)
analytics.feature_product_sentiment(product_id, comment_count, avg_rate, avg_like_ratio, total_likes, total_dislikes, total_aspect_mentions, positive_aspect_mentions, negative_aspect_mentions, neutral_aspect_mentions, avg_positive_pct, avg_negative_pct, avg_neutral_pct, positive_aspect_ratio, negative_aspect_ratio, neutral_aspect_ratio)
analytics.feature_product_aspect(product_id, term, total_mentions, positive_mentions, negative_mentions, neutral_mentions, avg_negative_pct, avg_neutral_pct, avg_positive_pct)
analytics.feature_brand_sentiment(brand_id, total_comments, total_aspect_mentions, positive_aspect_mentions, negative_aspect_mentions, neutral_aspect_mentions, avg_comment_rating, total_likes, total_dislikes)
analytics.feature_category_sentiment(category_id, total_comments, total_aspect_mentions, positive_aspect_mentions, negative_aspect_mentions, neutral_aspect_mentions, avg_comment_rating, total_likes, total_dislikes)
analytics.feature_aspect(term, total_mentions, positive_mentions, negative_mentions, neutral_mentions, avg_negative_pct, avg_neutral_pct, avg_positive_pct)
analytics.feature_time(hour, iso_weekday, total_events, total_views, total_cart_adds, total_purchases, total_removes)

kpi.rfm_segments(user_id BIGINT PK/FK->users.user_id, recency_days INT, frequency INT, 
                 monetary BIGINT, rfm_code TEXT, rfm_label TEXT)
                 -- مقادیر rfm_label شامل: 'vip', 'promising', 'at_risk', 'lost', 'regular'

kpi.ml_user_clusters(user_id BIGINT PK/FK->users.user_id, cluster_id INT, cluster_name TEXT)
                     -- مقادیر cluster_name شامل: 'vip_champions', 'night_weekend_buyers', 'active_loyals', 'low_intent_shoppers', 'churned_customers'
kpi.global_funnel(view_to_cart_pct, cart_to_purchase_pct, overall_conversion_pct, cart_abandonment_pct)`
kpi.product_360( conversion_rate, comment_count, star_rating, positive_sentiment_pct, sentiment_score, managerial_action_tag)`
kpi.brand_diagnostics( total_comments, avg_rating, brand_sentiment_score)`
kpi.aspect_diagnostics(aspect_name, total_mentions, positive_mentions, negative_mentions, negative_impact_pct, aspect_status)`


-- ==========================================
-- نکته‌ی مهم PostgreSQL: تابع ROUND
-- ==========================================
-- ROUND(double precision, integer) در PostgreSQL وجود ندارد -- فقط
-- ROUND(numeric, integer) پشتیبانی می‌شود. هر ستونی که DOUBLE PRECISION
-- است (مثل conversion_rate یا هر مقداری که با ::DOUBLE PRECISION ساخته
-- شده) قبل از ROUND کردن با تعداد رقم اعشار، باید اول به numeric کست شود:
--     ROUND(some_double_precision_expr::numeric, 4)
-- در غیر این صورت خطای «function round(double precision, integer) does
-- not exist» می‌گیری.
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
    "comments_embedding",

    # analytics schema
    "analytics",
    "analytics.feature_user",

    # kpi schema
    "kpi",
    "kpi.rfm_segments",
    "kpi.ml_user_clusters",

    # unqualified table names
    "feature_user",
    "rfm_segments",
    "ml_user_clusters",
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

    # ORDER BY + LIMIT بدون tie-breaker قطعی (ستون شبه‌کلید) -- همون
    # چک production_validator، نسخه‌ی regex‌ای برای fallback.
    # ORDER BY + LIMIT بدون tie-breaker قطعی (ستون شبه‌کلید) -- همون
    # چک production_validator، نسخه‌ی regex‌ای برای fallback. باید هر
    # جفت ORDER BY...LIMIT رو جدا چک کنه (finditer، نه فقط اولین)، چون
    # ممکنه چند تا CTE هر کدوم ORDER BY+LIMIT خودشون رو داشته باشن و
    # فقط یکیشون بدون tie-breaker باشه.
    id_like = (
        "product_id", "user_id", "comment_id", "session_id", "city_id",
        "brand_id", "category_id", "seller_id", "log_id", "aspect_id",
    )
    for order_match in re.finditer(
        r"\bORDER\s+BY\s+(.+?)\bLIMIT\s+\d+",
        stripped,
        re.IGNORECASE | re.DOTALL,
    ):
        order_clause = order_match.group(1).lower()
        has_id_col = any(col in order_clause for col in id_like) or re.search(
            r"\bid\b", order_clause
        )
        if not has_id_col:
            return (
                "ORDER BY+LIMIT بدون tie-breaker قطعی -- (احتمالاً داخل یک CTE) "
                "یک ستون شبه‌کلید (مثل product_id) رو به‌عنوان معیار دوم به این "
                "ORDER BY اضافه کن، وگرنه در معیارهای هم‌امتیاز (tie) هر اجرا "
                "می‌تونه ست متفاوتی برگردونه. بند مشکل‌دار: "
                f"ORDER BY {order_match.group(1).strip()[:150]}"
            )

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