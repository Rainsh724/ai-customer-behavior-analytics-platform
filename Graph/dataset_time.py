# PATH: Graph/dataset_time.py

"""
تاریخ مرجع ثابت دیتاست.

این پروژه real-time نیست و تمام محاسبات زمانی نسبی باید
بر اساس تاریخ مرجع ثابت دیتاست انجام شوند.
"""

from __future__ import annotations

import os
from datetime import date


# تاریخ مرجع پیش‌فرض دیتاست
DEFAULT_REFERENCE_DATE = date(2023, 3, 1)


# امکان تغییر از طریق .env
_env_reference_date = os.getenv(
    "DATASET_REFERENCE_DATE",
    ""
).strip()

if _env_reference_date:
    try:
        REFERENCE_DATE = date.fromisoformat(
            _env_reference_date
        )
    except ValueError:
        REFERENCE_DATE = DEFAULT_REFERENCE_DATE
else:
    REFERENCE_DATE = DEFAULT_REFERENCE_DATE


def get_reference_date(
    force_refresh: bool = False,
) -> date:
    """
    تاریخ مرجع ثابت دیتاست را برمی‌گرداند.

    force_refresh برای سازگاری با API قبلی نگه داشته شده،
    اما چون تاریخ ثابت است، عملاً کاری انجام نمی‌دهد.
    """
    return REFERENCE_DATE


def reset_reference_date_cache() -> None:
    """
    برای سازگاری با نسخه‌های قبلی.
    در نسخه‌ی فعلی cacheای وجود ندارد.
    """
    return None