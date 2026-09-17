"""
احراز هویت کاربران.

این ماژول اتصال مستقل خودش را به PostgreSQL دارد
و هیچ وابستگی‌ای به memory_store.py ندارد.

Expected table:

    public.app_users
        id             BIGSERIAL PRIMARY KEY
        username       TEXT UNIQUE NOT NULL
        password_hash  TEXT NOT NULL
        display_name   TEXT
        created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
"""

from __future__ import annotations

import logging
from typing import Any

import bcrypt
import psycopg2
import psycopg2.extras


logger = logging.getLogger(__name__)


def get_auth_conn():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="ai_project",
        user="postgres",
        password="",
    )


def ensure_users_schema() -> None:
    try:
        with get_auth_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.app_users');")
                row = cur.fetchone()

                if not row or row[0] is None:
                    raise RuntimeError(
                        "جدول public.app_users وجود ندارد. "
                        "آن را یک‌بار در PostgreSQL ایجاد کنید."
                    )

                logger.info(
                    "جدول public.app_users با موفقیت پیدا شد."
                )

    except Exception:
        logger.exception(
            "بررسی schema app_users شکست خورد."
        )
        raise


def verify_login(
    username: str,
    password: str
) -> dict[str, Any] | None:
    """
    اگر username/password درست باشند،
    اطلاعات کاربر را برمی‌گرداند؛
    در غیر این صورت None.
    """

    username = (username or "").strip()

    if not username or not password:
        return None

    with get_auth_conn() as conn:
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:

            cur.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    display_name
                FROM public.app_users
                WHERE username = %s
                """,
                (username,),
            )

            row = cur.fetchone()

    if not row:
        return None

    stored_hash = row["password_hash"].encode("utf-8")

    if not bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash
    ):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
    }