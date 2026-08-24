## PATH: app/graph/bi_agent.py
"""
پیاده‌سازی واقعی Tool_BI.

تغییر مهم نسبت به نسخه‌ی قبلی: قبلاً bi_builder خودش از داده‌ی خام
(sql_result/aspect_statistics) یک chart_spec/dashboard JSON می‌ساخت که
قرار بود فرانت‌اند خودمون رندرش کنه. طبق سند معماری جدید، مصورسازی دیگه
مسئولیت ما نیست -- Power BI از قبل این کار رو انجام می‌ده. کاری که این
ابزار می‌کنه فقط ساختن یک **لینک فیلترشده و آماده‌کلیک** به یک گزارش/
داشبورد از پیش ساخته‌شده در Power BI هست (بر اساس متریک/بعد/فیلترهایی که
Agent از سوال کاربر استخراج کرده)، نه اجرای هیچ کوئری یا ساخت هیچ نموداری
اینجا.

پیش‌نیاز پیکربندی (مثل db.py که به یک role/رمزعبور واقعی نیاز داشت):
    POWERBI_BASE_EMBED_URL  -- پیش‌فرض: endpoint استاندارد reportEmbed
    POWERBI_REPORT_ID       -- شناسه‌ی گزارش Power BI (باید از workspace
                               واقعی گرفته بشه)
    POWERBI_WORKSPACE_ID    -- شناسه‌ی workspace/group

نگاشت metric/dimension به نام جدول/ستون در مدل داده‌ی Power BI
(METRIC_FILTER_MAP) فعلاً placeholder با بهترین حدس از روی اسکیمای
Postgres پروژه است -- باید با اسکیمای واقعی دیتاست Power BI (که معمولاً
یک semantic model جداست، نه لزوماً هم‌نام با جداول Postgres) هماهنگ بشه.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)

POWERBI_BASE_EMBED_URL = os.getenv("POWERBI_BASE_EMBED_URL", "https://app.powerbi.com/reportEmbed")
POWERBI_REPORT_ID = os.getenv("POWERBI_REPORT_ID", "")
POWERBI_WORKSPACE_ID = os.getenv("POWERBI_WORKSPACE_ID", "")

# TODO: این نگاشت رو با اسکیمای واقعی دیتاست Power BI هماهنگ کن.
METRIC_LABELS = {
    "sales": "فروش",
    "rating": "امتیاز محصول",
    "complaints": "شکایات/نظرات منفی",
    "price": "قیمت",
    "views": "بازدید",
}


def _build_filters(product_id: int | None, brand_id: int | None, days_back: int | None) -> list[str]:
    filters: list[str] = []

    if product_id:
        filters.append(f"Products/ProductId eq {int(product_id)}")

    if brand_id:
        filters.append(f"Products/BrandId eq {int(brand_id)}")

    if days_back:
        cutoff = (datetime.utcnow() - timedelta(days=int(days_back))).strftime("%Y-%m-%d")
        filters.append(f"Sales/Date ge {cutoff}")

    return filters


def run_bi_tool(
    metric: str,
    dimension: str | None = None,
    product_id: int | None = None,
    brand_id: int | None = None,
    days_back: int | None = None,
) -> dict[str, Any]:
    """
    ورودی: metric (اجباری)، دیگر آرگومان‌ها اختیاری -- همه از آرگومان‌های
    tool_bi که Agent صدا زده میان (نگاه کن به tools.py).
    خروجی: dict شامل دقیقاً یک لینک قابل‌کلیک؛ Agent این لینک رو مستقیم
    در جواب نهایی به کاربر منتقل می‌کنه (طبق سند: "زمانی که کاربر نمودار
    می‌خواهد، لینک قابل کلیک برایش ارسال می‌شود").
    """
    if not metric or not metric.strip():
        return {"error": "metric مشخص نشده."}

    if not POWERBI_REPORT_ID or not POWERBI_WORKSPACE_ID:
        return {
            "error": (
                "POWERBI_REPORT_ID / POWERBI_WORKSPACE_ID تنظیم نشده. "
                "این‌ها باید در متغیرهای محیطی سرویس مقداردهی بشن."
            )
        }

    filters = _build_filters(product_id, brand_id, days_back)

    base_params = {
        "reportId": POWERBI_REPORT_ID,
        "groupId": POWERBI_WORKSPACE_ID,
    }
    link = f"{POWERBI_BASE_EMBED_URL}?{urlencode(base_params)}"
    if filters:
        filter_qs = "&".join(f"filter={quote(f)}" for f in filters)
        link = f"{link}&{filter_qs}"

    return {
        "dashboard_link": link,
        "metric": metric,
        "metric_label": METRIC_LABELS.get(metric, metric),
        "dimension": dimension,
        "applied_filters": filters,
        "note": "کاربر باید دسترسی مجاز به این گزارش Power BI رو داشته باشه تا لینک باز بشه.",
    }
