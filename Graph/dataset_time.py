## PATH: app/graph/dataset_time.py
"""
محاسبه‌ی "زمان حال" منطقی برای این دیتاست -- نه زمان واقعی امروز.

مشکل:
------
دیتاست ما real-time نیست؛ یک اسنپ‌شات تاریخی/استاتیک از یک بازه‌ی
مشخصه. ولی description ابزار tool_sql قبلاً به Agent می‌گفت از
NOW()/CURRENT_DATE واقعیِ Postgres استفاده کنه. چون NOW() واقعی (مثلاً
تاریخ امروزِ سرور) معمولاً خیلی جلوتر از آخرین رکورد موجود در دیتاسته،
هر فیلتر نسبی مثل "۳۰ روز اخیر" همیشه صفر ردیف برمی‌گردوند -- دقیقاً
همون چیزی که در نتیجه‌ی "دفترچه‌های یادداشت" (فروش=۰ در هر دو بازه)
دیدیم.

راه‌حل:
--------
"الان" رو به‌جای NOW() واقعی، جدیدترین تاریخ/زمانِ موجود در خودِ دیتاست
تعریف می‌کنیم. این محاسبه یک کوئری واقعی به دیتابیس می‌زنه، پس نباید هر
بار (هر تماس Agent، هر پیام کاربر) دوباره انجام بشه -- فقط یک‌بار (اولین
باری که لازم بشه) محاسبه و در حافظه‌ی پردازه cache می‌شه؛ دفعات بعد
مستقیم از cache خونده می‌شه، بدون هیچ query جدیدی.

main.py::run() این تاریخ رو یک‌بار در ابتدای هر مکالمه‌ی *جدید* (نه هر
پیام) در system prompt تزریق می‌کنه -- Agent باید محاسبات زمانی نسبی رو
نسبت به همین تاریخ انجام بده، نه CURRENT_DATE/NOW() واقعی Postgres.

نکته برای شما: CANDIDATE_TIME_SOURCES پایین رو با ستون‌های زمان‌دار
واقعیِ اسکیمای خودتون (از skima.text) تطبیق بدید -- هرکدوم که واقعاً
نشون‌دهنده‌ی "جدیدترین رویداد ثبت‌شده در سیستم" باشه (مثلاً آخرین کامنت،
آخرین لاگ رفتار کاربر، یا اگه kpi.daily_funnel ستون تاریخ داره، اونم
اضافه کنید). چند منبع می‌ذاریم و MAX کل‌شون رو برمی‌داریم چون معمولاً
جدول‌های مختلف تا تاریخ‌های کمی متفاوت پر شدن.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, timezone

from .db import run_readonly_query

logger = logging.getLogger(__name__)

# TODO: با ستون‌های زمان‌دار واقعی اسکیمای خودتون هماهنگ کنید.
CANDIDATE_TIME_QUERIES: list[str] = [
    "SELECT MAX(created_at) AS latest FROM comments",
    "SELECT MAX(timestamp) AS latest FROM user_behavior_logs",
]

# اگه هیچ‌کدوم از کوئری‌های بالا جواب ندادن (مثلاً جدول خالیه یا دیتابیس
# در دسترس نیست)، به‌جای کرش کردن، از این متغیر محیطی (اگه ست شده باشه)
# یا در نهایت از تاریخ واقعی امروز استفاده می‌کنیم -- fail-safe، نه بی‌جواب.
FALLBACK_REFERENCE_DATE = os.getenv("DATASET_FALLBACK_REFERENCE_DATE")  # فرمت: YYYY-MM-DD

_cached_reference_date: date | None = None
_lock = threading.Lock()


def get_reference_date(force_refresh: bool = False) -> date:
    """
    "امروزِ" این دیتاست رو برمی‌گردونه. فقط بار اول (یا وقتی
    force_refresh=True باشه) واقعاً به دیتابیس query می‌زنه؛ دفعات بعد
    از cache حافظه‌ای می‌خونه -- صفر هزینه‌ی اضافه به ازای هر پیام کاربر.
    """
    global _cached_reference_date

    if _cached_reference_date is not None and not force_refresh:
        return _cached_reference_date

    with _lock:
        if _cached_reference_date is not None and not force_refresh:
            return _cached_reference_date

        latest: date | None = None
        for query in CANDIDATE_TIME_QUERIES:
            try:
                rows = run_readonly_query(query)
                value = rows[0].get("latest") if rows else None
                if value is None:
                    continue
                value_date = value.date() if isinstance(value, datetime) else value
                if latest is None or value_date > latest:
                    latest = value_date
            except Exception as exc:  # noqa: BLE001
                logger.warning("get_reference_date: کوئری '%s' شکست خورد: %s", query, exc)

        if latest is None and FALLBACK_REFERENCE_DATE:
            try:
                latest = date.fromisoformat(FALLBACK_REFERENCE_DATE)
            except ValueError:
                logger.warning("get_reference_date: DATASET_FALLBACK_REFERENCE_DATE نامعتبره: %s", FALLBACK_REFERENCE_DATE)

        if latest is None:
            logger.warning(
                "get_reference_date: هیچ تاریخی از دیتابیس/فال‌بک پیدا نشد -- "
                "موقتاً از تاریخ واقعی امروز استفاده می‌شه (احتمالاً باعث "
                "می‌شه فیلترهای نسبی زمانی صفر ردیف برگردونن، دقیقاً همون "
                "مشکلی که این ماژول قراره حلش کنه -- لطفاً CANDIDATE_TIME_QUERIES "
                "یا DATASET_FALLBACK_REFERENCE_DATE رو تنظیم کنید)."
            )
            latest = datetime.now(timezone.utc).date()

        _cached_reference_date = latest
        logger.info("get_reference_date: 'امروزِ' دیتاست روی %s قفل شد (تا وقتی پردازه دوباره استارت بشه یا force_refresh بگیره).", latest)
        return _cached_reference_date


def reset_reference_date_cache() -> None:
    """برای تست، یا وقتی دیتاست به‌روزرسانی شد و می‌خواید دوباره محاسبه بشه."""
    global _cached_reference_date
    _cached_reference_date = None
