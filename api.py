## PATH: app/api.py
"""
FastAPI لایه‌ی بک‌اند برای دستیار هوشمند.

فقط یک wrapper نازک روی main.run() است -- منطق agent (Graph/*),
حافظه‌ی مکالمه (memory_store.py) دست‌نخورده باقی می‌مانند.

اجرا (development):
    uvicorn api:app --reload --port 8000

فرانت‌اند (rahin_front_ai) از قبل انتظار همین قرارداد را دارد:
    POST /api/chat   body: {"message": "...", "session_id": "..."}
    ->   {"answer": "..."}
"""
from __future__ import annotations
import json
from main import run as run_agent, run_stream as run_agent_stream
import logging
import os
import threading
import time
from io import BytesIO
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import memory_store
import auth_store
from Graph.dataset_time import get_reference_date
from Graph.db import run_readonly_query
from Graph.llm_client import preload_embedding_model
from main import run as run_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="Rahin AI Assistant API")

# ------------------------------------------------------------------
# کش ساده‌ی درون‌حافظه‌ای برای صفحات داشبورد/هوش برند/تحلیل دسته‌بندی.
# این صفحات real-time نیستن (دیتاست تاریخ مرجع ثابت داره)، پس نیازی
# نیست هر بار که کاربر صفحه رو باز می‌کنه SQL دوباره اجرا بشه.
#
# مکانیزم: هر نتیجه با یک "کلید" و زمان محاسبه‌اش نگه داشته می‌شه.
# تا وقتی سن کش از CACHE_TTL_SECONDS کمتره، مستقیم از کش برگردونده
# می‌شه؛ وگرنه دوباره محاسبه و کش می‌شه. موقع بالا اومدن سرور هم
# on_startup این‌ها رو یک‌بار پیش‌بارگذاری می‌کنه تا اولین کاربر هم
# منتظر SQL نمونه.
# ------------------------------------------------------------------
CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL_SECONDS", "300"))
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def get_cached(key: str, compute: Callable[[], Any]) -> Any:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and (now - hit[0]) < CACHE_TTL_SECONDS:
            return hit[1]

    value = compute()

    with _cache_lock:
        _cache[key] = (now, value)
    return value

# ------------------------------------------------------------------
# CORS -- فرانت (vite dev server) روی origin جدا اجرا می‌شود.
# در production این لیست را به دامنه‌ی واقعی محدود کن.
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: همه؛ prod: دامنه‌ی دقیق فرانت را بگذار
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Startup: مدل embedding را زودتر لود کن (نه وسط اولین سوال کاربر) و
# وجود جدول chat_memory را چک کن.
# ------------------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    try:
        preload_embedding_model()
    except Exception:
        logger.exception("preload_embedding_model شکست خورد")

    try:
        memory_store.ensure_schema()
    except Exception:
        logger.exception(
            "ensure_schema شکست خورد -- حافظه‌ی چت کار نخواهد کرد تا رفعش کنی"
        )

    try:
        memory_store.ensure_eval_schema()
    except Exception:
        logger.exception("ensure_eval_schema شکست خورد")
    
    try:
        auth_store.ensure_users_schema()
    except Exception:
        logger.exception("ensure_users_schema شکست خورد")
    # پیش‌بارگذاری کش صفحات داشبوردی -- تو یک ترد جدا تا startup سرور
    # رو معطل نکنه، ولی قبل از اینکه کاربری کلیک کنه شروع می‌شه.
    threading.Thread(target=_warm_all_caches, daemon=True).start()
    threading.Thread(target=_background_refresh_loop, daemon=True).start()


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    chart: dict | None = None
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    ok: bool
    display_name: str | None = None

# ------------------------------------------------------------------
# Endpoint اصلی چت -- دقیقاً همان مسیر/قراردادی که main.jsx صدا می‌زند.
# ------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="message خالی است")

    try:
        result = run_agent(question=question, chat_id=payload.session_id)
    except Exception:
        logger.exception("اجرای agent برای session_id=%s شکست خورد", payload.session_id)
        raise HTTPException(status_code=500, detail="خطا در پردازش درخواست توسط دستیار هوشمند")

    answer = (result or {}).get("final_answer") or ""

    chart = None
    for entry in reversed((result or {}).get("tool_trace", []) or []):
        if entry.get("tool") == "tool_chart" and entry.get("ok") and entry.get("chart_data"):
            chart = entry["chart_data"]
            break

    return ChatResponse(answer=answer, chart=chart)

@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest):
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="message خالی است")

    def event_generator():
        try:
            for event in run_agent_stream(question=question, chat_id=payload.session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("stream failed for session_id=%s", payload.session_id)
            yield f"data: {json.dumps({'type': 'error', 'message': 'خطا در پردازش'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/api/chat/{session_id}")
def delete_chat(session_id: str) -> dict:
    try:
        memory_store.delete_messages(session_id)
    except Exception:
        logger.exception("حذف چت session_id=%s شکست خورد", session_id)
        raise HTTPException(status_code=500, detail="خطا در حذف چت")
    return {"deleted": True}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    try:
        user = auth_store.verify_login(payload.username, payload.password)
    except Exception:
        logger.exception("بررسی لاگین برای username=%s شکست خورد", payload.username)
        raise HTTPException(status_code=500, detail="خطا در بررسی اطلاعات ورود")

    if not user:
        raise HTTPException(status_code=401, detail="نام کاربری یا رمز عبور اشتباه است")

    return LoginResponse(ok=True, display_name=user.get("display_name"))
# ------------------------------------------------------------------
# هوش برند -- کوئری مستقیم SQL (بدون عبور از LLM)، چون ساختار صفحه
# ثابته و نیازی به تفسیر زبان طبیعی نداره.
# بازه‌ی "۳۰ روز اخیر" نسبت به تاریخ مرجعِ دیتاست حساب می‌شه، نه
# NOW() واقعی Postgres، چون این دیتاست real-time نیست.
# ------------------------------------------------------------------
BRAND_INTELLIGENCE_SQL = """
WITH ref AS (
    SELECT %s::date AS d
),
base AS (
    SELECT p.brand_id, p.id AS product_id, p.price, ubl.event_type
    FROM user_behavior_logs ubl
    JOIN products p ON p.id = ubl.product_id
    CROSS JOIN ref
    WHERE ubl.timestamp >= ref.d - INTERVAL '30 days'
      AND ubl.timestamp < ref.d + INTERVAL '1 day'
),
agg AS (
    SELECT
        brand_id,
        COUNT(*) FILTER (WHERE event_type = 'view')     AS total_views_30d,
        COUNT(*) FILTER (WHERE event_type = 'purchase')  AS total_purchases_30d
    FROM base
    GROUP BY brand_id
),
purchase_products AS (
    SELECT brand_id, product_id, price, COUNT(*) AS purchase_cnt
    FROM base
    WHERE event_type = 'purchase'
    GROUP BY brand_id, product_id, price
),
revenue AS (
    SELECT brand_id, SUM(purchase_cnt * price) AS total_revenue_30d
    FROM purchase_products
    GROUP BY brand_id
),
sentiment AS (
    SELECT p.brand_id, AVG(c.rate) AS avg_rate
    FROM comments c
    JOIN products p ON p.id = c.product_id
    GROUP BY p.brand_id
)
SELECT
    b.name AS brand_name,
    COALESCE(agg.total_views_30d, 0)      AS total_views_30d,
    COALESCE(agg.total_purchases_30d, 0)  AS total_purchases_30d,
    COALESCE(revenue.total_revenue_30d, 0) AS total_revenue_30d,
    CASE WHEN COALESCE(agg.total_views_30d, 0) = 0 THEN 0
         ELSE ROUND((agg.total_purchases_30d::numeric / agg.total_views_30d) * 100, 2)
    END AS conversion_rate_30d,
    ROUND(sentiment.avg_rate::numeric, 2) AS brand_sentiment_score
FROM brands b
LEFT JOIN agg      ON agg.brand_id = b.brand_id
LEFT JOIN revenue  ON revenue.brand_id = b.brand_id
LEFT JOIN sentiment ON sentiment.brand_id = b.brand_id
ORDER BY total_revenue_30d DESC NULLS LAST
LIMIT 50;
"""


def _compute_brand_intelligence() -> dict:
    ref_date = get_reference_date()
    rows = run_readonly_query(BRAND_INTELLIGENCE_SQL, (ref_date,))
    return {"items": rows}


@app.get("/api/brand-intelligence")
def brand_intelligence() -> dict:
    try:
        return get_cached("brand_intelligence", _compute_brand_intelligence)
    except Exception:
        logger.exception("کوئری هوش برند شکست خورد")
        raise HTTPException(status_code=500, detail="خطا در دریافت اطلاعات هوش برند")


# ------------------------------------------------------------------
# تحلیل دسته‌بندی‌ها -- شش دسته‌ی اصلی (category1) از منظر فروش،
# نرخ تبدیل و رضایت، نسبت به هم رتبه‌بندی و به سطح کیفی
# (قوی/متوسط/ضعیف و خوب/متوسط/پایین) نگاشت می‌شن -- دقیقاً همون
# چیزی که کامپوننت CategoryIntelligence در فرانت نمایش می‌ده.
# ------------------------------------------------------------------
CATEGORY_INTELLIGENCE_SQL = """
WITH ref AS (
    SELECT %s::date AS d
),
base AS (
    SELECT c.sub_category, ubl.event_type
    FROM user_behavior_logs ubl
    JOIN products p ON p.id = ubl.product_id
    JOIN categories c ON c.category_id = p.category_id
    CROSS JOIN ref
    WHERE ubl.timestamp >= ref.d - INTERVAL '30 days'
      AND ubl.timestamp < ref.d + INTERVAL '1 day'
),
agg AS (
    SELECT
        sub_category,
        COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases_30d,
        COUNT(*) FILTER (WHERE event_type = 'view')      AS views_30d
    FROM base
    GROUP BY sub_category
),
conv AS (
    SELECT
        sub_category,
        purchases_30d,
        CASE WHEN views_30d = 0 THEN 0
             ELSE purchases_30d::numeric / views_30d
        END AS conversion_rate
    FROM agg
),
satisfaction AS (
    SELECT c.sub_category, AVG(cm.rate) AS avg_rate
    FROM comments cm
    JOIN products p ON p.id = cm.product_id
    JOIN categories c ON c.category_id = p.category_id
    GROUP BY c.sub_category
),
combined AS (
    SELECT conv.sub_category, conv.purchases_30d, conv.conversion_rate, satisfaction.avg_rate
    FROM conv
    LEFT JOIN satisfaction ON satisfaction.sub_category = conv.sub_category
),
ranked AS (
    SELECT
        *,
        NTILE(3) OVER (ORDER BY purchases_30d)     AS sales_tier,
        NTILE(3) OVER (ORDER BY conversion_rate)   AS conversion_tier,
        NTILE(3) OVER (ORDER BY avg_rate)          AS satisfaction_tier
    FROM combined
)
SELECT
    sub_category AS name,
    purchases_30d,
    ROUND(conversion_rate::numeric * 100, 2) AS conversion_rate_pct,
    ROUND(avg_rate::numeric, 2)              AS satisfaction_score,
    CASE sales_tier
        WHEN 1 THEN 'weak' WHEN 2 THEN 'medium' ELSE 'strong'
    END AS sales_status,
    CASE conversion_tier
        WHEN 1 THEN 'weak' WHEN 2 THEN 'medium' ELSE 'strong'
    END AS conversion_status,
    CASE satisfaction_tier
        WHEN 1 THEN 'low' WHEN 2 THEN 'medium' ELSE 'good'
    END AS satisfaction_status
FROM ranked
ORDER BY sub_category;
"""


def _compute_category_intelligence() -> dict:
    ref_date = get_reference_date()
    rows = run_readonly_query(CATEGORY_INTELLIGENCE_SQL, (ref_date,))
    return {"items": rows}


@app.get("/api/category-intelligence")
def category_intelligence() -> dict:
    try:
        return get_cached("category_intelligence", _compute_category_intelligence)
    except Exception:
        logger.exception("کوئری تحلیل دسته‌بندی‌ها شکست خورد")
        raise HTTPException(status_code=500, detail="خطا در دریافت اطلاعات تحلیل دسته‌بندی‌ها")


# ------------------------------------------------------------------
# داشبورد -- ۴ بخش: روند ۳۰ روزه، پرفروش‌ترین برندها، پرفروش‌ترین
# محصولات، خوشه‌بندی رفتاری مشتریان. نوع نمودار (خطی/میله‌ای) از قبل
# توی خود React کامپوننت مشخص شده -- اینجا فقط داده رو با شکل درست
# برمی‌گردونیم.
#
# ⚠️ TODO: مقدار دقیق event_type برای "حذف از سبد" هنوز تایید نشده --
# فعلاً 'remove_from_cart' گذاشتم. بعد از اجرای check_event_types.sql
# و گرفتن مقادیر واقعی، اگه فرق داشت این‌جا اصلاح می‌کنیم.
# ------------------------------------------------------------------
DASHBOARD_TREND_SQL = """
WITH ref AS (
    SELECT %s::date AS d
),
days AS (
    SELECT generate_series(
        (SELECT d - INTERVAL '29 days' FROM ref)::date,
        (SELECT d FROM ref)::date,
        INTERVAL '1 day'
    )::date AS day
)
SELECT
    to_char(days.day, 'YYYY-MM-DD') AS date,
    COUNT(*) FILTER (WHERE ubl.event_type = 'view')             AS views,
    COUNT(*) FILTER (WHERE ubl.event_type = 'add_to_cart')      AS carts,
    COUNT(*) FILTER (WHERE ubl.event_type = 'purchase')         AS purchases,
    COUNT(*) FILTER (WHERE ubl.event_type = 'remove_from_cart') AS removes
FROM days
LEFT JOIN user_behavior_logs ubl ON ubl.timestamp::date = days.day
GROUP BY days.day
ORDER BY days.day;
"""

DASHBOARD_TOP_BRANDS_SQL = """
WITH ref AS (
    SELECT %s::date AS d
),
base AS (
    SELECT p.brand_id
    FROM user_behavior_logs ubl
    JOIN products p ON p.id = ubl.product_id
    CROSS JOIN ref
    WHERE ubl.event_type = 'purchase'
      AND ubl.timestamp >= ref.d - INTERVAL '30 days'
      AND ubl.timestamp < ref.d + INTERVAL '1 day'
)
SELECT b.name AS brand_name, COUNT(*) AS total_purchases_30d
FROM base
JOIN brands b ON b.brand_id = base.brand_id
GROUP BY b.name
ORDER BY total_purchases_30d DESC
LIMIT 10;
"""

DASHBOARD_TOP_PRODUCTS_SQL = """
WITH ref AS (
    SELECT %s::date AS d
)
SELECT p.title_fa, COUNT(*) AS total_purchases_30d
FROM user_behavior_logs ubl
JOIN products p ON p.id = ubl.product_id
CROSS JOIN ref
WHERE ubl.event_type = 'purchase'
  AND ubl.timestamp >= ref.d - INTERVAL '30 days'
  AND ubl.timestamp < ref.d + INTERVAL '1 day'
GROUP BY p.title_fa
ORDER BY total_purchases_30d DESC
LIMIT 10;
"""

DASHBOARD_SEGMENTS_SQL = """
SELECT cluster_name AS segment, COUNT(*) AS count
FROM kpi.ml_user_clusters
GROUP BY cluster_name
ORDER BY count DESC;
"""


def _compute_dashboard() -> dict:
    ref_date = get_reference_date()
    trend = run_readonly_query(DASHBOARD_TREND_SQL, (ref_date,))
    brands = run_readonly_query(DASHBOARD_TOP_BRANDS_SQL, (ref_date,))
    products = run_readonly_query(DASHBOARD_TOP_PRODUCTS_SQL, (ref_date,))
    segments = run_readonly_query(DASHBOARD_SEGMENTS_SQL)
    return {"trend": trend, "brands": brands, "products": products, "segments": segments}


@app.get("/api/dashboard")
def dashboard() -> dict:
    try:
        return get_cached("dashboard", _compute_dashboard)
    except Exception:
        logger.exception("کوئری‌های داشبورد شکست خوردن")
        raise HTTPException(status_code=500, detail="خطا در دریافت اطلاعات داشبورد")


# ------------------------------------------------------------------
# هوش مشتریان -- تعداد و سهم هر بخش، از جدول خوشه‌بندی
# kpi.ml_user_clusters. نگاشت cluster_name <-> برچسب انگلیسی segment
# دقیقاً همونیه که خود فرانت (main.jsx -> downloadSegment) از قبل
# داره، تا با translateSegment و دکمه‌ی دانلود هماهنگ بمونه.
# ------------------------------------------------------------------
CUSTOMER_SEGMENTS_SQL = """
SELECT
    cluster_name,
    COUNT(*) AS count,
    ROUND(COUNT(*)::numeric * 100.0 / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS share
FROM kpi.ml_user_clusters
GROUP BY cluster_name;
"""

CLUSTER_TO_SEGMENT_LABEL = {
    "vip_champions": "VIP Customer",
    "active_loyals": "Returning Customer",
    "night_weekend_buyers": "One-Time Buyer",
    "low_intent_shoppers": "Low Engagement",
    "churned_customers": "Window Shopper",
}


def _compute_customer_segments() -> dict:
    rows = run_readonly_query(CUSTOMER_SEGMENTS_SQL)
    items = [
        {
            "segment": CLUSTER_TO_SEGMENT_LABEL.get(r["cluster_name"], r["cluster_name"]),
            "count": r["count"],
            "share": float(r["share"]) if r["share"] is not None else None,
        }
        for r in rows
    ]
    return {"items": items}


@app.get("/api/customer-segments")
def customer_segments() -> dict:
    try:
        return get_cached("customer_segments", _compute_customer_segments)
    except Exception:
        logger.exception("کوئری بخش‌های مشتریان شکست خورد")
        raise HTTPException(status_code=500, detail="خطا در دریافت بخش‌های مشتریان")


# ------------------------------------------------------------------
# خروجی اکسل شناسه‌ی مشتریان هر بخش -- همون منطق فایل تیمیت
# (bakend/routers/customers.py + services/excel_service.py)، فقط
# اینجا از همون run_readonly_query و pool موجود استفاده می‌شه به‌جای
# اتصال جدا.
# ------------------------------------------------------------------
CUSTOMER_IDS_BY_CLUSTER_SQL = """
SELECT user_id
FROM kpi.ml_user_clusters
WHERE cluster_name = %s
ORDER BY user_id;
"""


@app.get("/api/customer-segments/{cluster_name}/export")
def export_customer_segment(cluster_name: str) -> StreamingResponse:
    if cluster_name not in CLUSTER_TO_SEGMENT_LABEL:
        raise HTTPException(status_code=400, detail="cluster_name نامعتبر است")

    try:
        rows = run_readonly_query(CUSTOMER_IDS_BY_CLUSTER_SQL, (cluster_name,))
    except Exception:
        logger.exception("خواندن شناسه‌ی مشتریان بخش '%s' شکست خورد", cluster_name)
        raise HTTPException(status_code=500, detail="خطا در دریافت لیست مشتریان")

    user_ids = [r["user_id"] for r in rows]
    df = pd.DataFrame({"user_id": user_ids})
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{cluster_name}.xlsx"'},
    )


# ------------------------------------------------------------------
# پیش‌بارگذاری + ریفرش پس‌زمینه‌ای کش -- در startup هر سه صفحه یک‌بار
# محاسبه می‌شن (پس اولین کاربر هم منتظر SQL نمی‌مونه)، و بعد هر
# CACHE_TTL_SECONDS یک‌بار توی یک ترد جدا خودکار تازه می‌شن. کاربر
# هیچ‌وقت منتظر اجرای کوئری نمی‌مونه -- همیشه از کش می‌خونه.
# ------------------------------------------------------------------
_CACHE_BUILDERS: dict[str, Callable[[], Any]] = {
    "brand_intelligence": _compute_brand_intelligence,
    "category_intelligence": _compute_category_intelligence,
    "dashboard": _compute_dashboard,
    "customer_segments": _compute_customer_segments,
}


def _warm_all_caches() -> None:
    for key, builder in _CACHE_BUILDERS.items():
        try:
            value = builder()
            with _cache_lock:
                _cache[key] = (time.time(), value)
            logger.info("کش '%s' با موفقیت تازه‌سازی شد", key)
        except Exception:
            logger.exception("تازه‌سازی کش '%s' شکست خورد", key)


def _background_refresh_loop() -> None:
    while True:
        time.sleep(CACHE_TTL_SECONDS)
        _warm_all_caches()