"""
Persistent conversation memory backed by PostgreSQL.

Responsibilities
----------------
- Load/save chat history by chat_id
- Maintain a dedicated write-capable PostgreSQL connection pool
- Read database configuration from .env / environment variables
- Verify that chat_memory exists without creating/modifying database objects
- Compact old conversation turns when the configured threshold is exceeded

Important
---------
This module intentionally uses a separate database role from the analytics
connection used by app/graph/db.py.

Analytics DB:
    read-only role

Chat memory DB:
    write-capable role
    CHAT_DB_USER / CHAT_DB_PASSWORD / CHAT_DB_* settings

The runtime application must NOT create database tables.
The chat_memory table should be created once during database setup/migration.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool

from Graph.llm_client import call_llm_json


# ============================================================
# Environment
# ============================================================

# Load .env before reading any environment variables.
# override=False means explicitly exported environment variables
# have priority over values inside .env.
load_dotenv(override=False)


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

class ChatDBConfig:
    """
    PostgreSQL configuration dedicated to conversation memory.

    IMPORTANT:
    These values are intentionally separate from DB_* used by the
    analytics/read-only database connection.

    Expected .env:

        CHAT_DB_NAME=postgres
        CHAT_DB_USER=app_chat_writer
        CHAT_DB_PASSWORD=ChatWriter123
        CHAT_DB_HOST=localhost
        CHAT_DB_PORT=5432

    """

    DB_NAME = os.getenv("CHAT_DB_NAME", "postgres")
    DB_USER = os.getenv("CHAT_DB_USER", "app_chat_writer")
    DB_PASSWORD = os.getenv("CHAT_DB_PASSWORD", "HiddenPatern")
    DB_HOST = os.getenv("CHAT_DB_HOST", "localhost")
    DB_PORT = os.getenv("CHAT_DB_PORT", "5432")

    POOL_MIN_CONN = int(os.getenv("CHAT_PG_POOL_MIN", "1"))
    POOL_MAX_CONN = int(os.getenv("CHAT_PG_POOL_MAX", "5"))

    # Number of most recent conversation turns that remain raw.
    MAX_RAW_TURNS = int(
        os.getenv("CHAT_MEMORY_MAX_RAW_TURNS", "6")
    )

    # PostgreSQL connection timeout in seconds.
    CONNECT_TIMEOUT = int(
        os.getenv("CHAT_DB_CONNECT_TIMEOUT", "10")
    )


SUMMARY_MARKER = "[خلاصه‌ی مکالمات قبلی]"


# ============================================================
# Global connection pool
# ============================================================

_pool: ThreadedConnectionPool | None = None


# ============================================================
# Helpers
# ============================================================

def _validate_config() -> None:
    """
    Validate the minimum required configuration before opening
    a database connection.
    """

    required = {
        "CHAT_DB_NAME": ChatDBConfig.DB_NAME,
        "CHAT_DB_USER": ChatDBConfig.DB_USER,
        "CHAT_DB_HOST": ChatDBConfig.DB_HOST,
        "CHAT_DB_PORT": ChatDBConfig.DB_PORT,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "تنظیمات دیتابیس حافظه ناقص است: "
            + ", ".join(missing)
        )

    if not ChatDBConfig.DB_PASSWORD:
        logger.warning(
            "CHAT_DB_PASSWORD تنظیم نشده است."
        )

    if ChatDBConfig.POOL_MIN_CONN < 1:
        raise RuntimeError(
            "CHAT_PG_POOL_MIN باید حداقل 1 باشد."
        )

    if ChatDBConfig.POOL_MAX_CONN < ChatDBConfig.POOL_MIN_CONN:
        raise RuntimeError(
            "CHAT_PG_POOL_MAX باید بزرگ‌تر یا مساوی "
            "CHAT_PG_POOL_MIN باشد."
        )

    if ChatDBConfig.MAX_RAW_TURNS < 1:
        raise RuntimeError(
            "CHAT_MEMORY_MAX_RAW_TURNS باید حداقل 1 باشد."
        )


def _log_config() -> None:
    """
    Log effective runtime configuration.

    Password is intentionally never logged.
    """

    logger.info(
        "Chat memory DB config: "
        "host=%s port=%s db=%s user=%s "
        "pool=%s-%s max_raw_turns=%s password_set=%s",
        ChatDBConfig.DB_HOST,
        ChatDBConfig.DB_PORT,
        ChatDBConfig.DB_NAME,
        ChatDBConfig.DB_USER,
        ChatDBConfig.POOL_MIN_CONN,
        ChatDBConfig.POOL_MAX_CONN,
        ChatDBConfig.MAX_RAW_TURNS,
        bool(ChatDBConfig.DB_PASSWORD),
    )


# ============================================================
# Connection pool
# ============================================================

def get_pool() -> ThreadedConnectionPool:
    """
    Lazily create the dedicated write-capable connection pool.
    """

    global _pool

    if _pool is None:
        _validate_config()
        _log_config()

        logger.info(
            "در حال ساخت connection pool حافظه‌ی چت..."
        )

        try:
            _pool = ThreadedConnectionPool(
                minconn=ChatDBConfig.POOL_MIN_CONN,
                maxconn=ChatDBConfig.POOL_MAX_CONN,
                dbname=ChatDBConfig.DB_NAME,
                user=ChatDBConfig.DB_USER,
                password=ChatDBConfig.DB_PASSWORD,
                host=ChatDBConfig.DB_HOST,
                port=ChatDBConfig.DB_PORT,
                connect_timeout=ChatDBConfig.CONNECT_TIMEOUT,
                application_name="ai_customer_behavior_chat_memory",
            )

        except Exception:
            logger.exception(
                "ساخت connection pool حافظه‌ی چت شکست خورد."
            )
            raise

    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    """
    Borrow a connection from the pool.

    On success:
        commit

    On failure:
        rollback

    If the connection itself becomes unusable, remove it from
    the pool instead of returning a broken connection.
    """

    pool = get_pool()
    conn = pool.getconn()

    try:
        yield conn
        conn.commit()

    except Exception:
        try:
            conn.rollback()
        except Exception:
            logger.exception(
                "rollback اتصال حافظه‌ی چت شکست خورد."
            )

        # A connection can become unusable after network/database
        # failures. Test whether PostgreSQL still considers it usable.
        try:
            if conn.closed:
                pool.putconn(conn, close=True)
            else:
                pool.putconn(conn)
        except Exception:
            logger.exception(
                "بازگرداندن connection به pool شکست خورد."
            )

        raise

    else:
        pool.putconn(conn)


# ============================================================
# Connection diagnostics
# ============================================================

def check_connection() -> dict[str, Any]:
    """
    Verify the actual database/user used by the running application.

    This is intentionally read-only and safe to call for diagnostics.
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    current_schema(),
                    inet_server_addr()::text,
                    inet_server_port()
                """
            )

            row = cur.fetchone()

            return {
                "database": row[0],
                "user": row[1],
                "schema": row[2],
                "server": row[3],
                "port": row[4],
            }


# ============================================================
# Schema verification
# ============================================================

def ensure_schema() -> None:
    """
    Verify that public.chat_memory already exists.

    IMPORTANT:
    This function DOES NOT execute CREATE TABLE.

    The application role should not need CREATE privileges on schema public.
    Database schema creation belongs to a one-time migration/setup step.

    Expected table:

        public.chat_memory
            chat_id      TEXT PRIMARY KEY
            messages     JSONB NOT NULL
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT to_regclass('public.chat_memory');
                    """
                )

                row = cur.fetchone()

                if not row or row[0] is None:
                    raise RuntimeError(
                        "جدول public.chat_memory وجود ندارد. "
                        "آن را یک‌بار در PostgreSQL ایجاد کنید."
                    )

                logger.info(
                    "جدول public.chat_memory با موفقیت پیدا شد."
                )

    except Exception:
        logger.exception(
            "بررسی schema حافظه‌ی چت شکست خورد."
        )
        raise


def ensure_eval_schema() -> None:
    """
    Verify that public.eval_log already exists.

    این جدول جدا از chat_memory است -- برای لاگ کردن معیارهای ارزیابیِ
    هر پاسخ (faithfulness_score / relevance_score / confidence_score --
    نگاه کن به audit.py) استفاده می‌شه، تا بعداً بشه calibration رو
    به‌صورت تجمعی/آفلاین از روش حساب کرد (نگاه کن به
    compute_calibration پایین همین فایل).

    IMPORTANT:
    درست مثل ensure_schema بالا، این تابع هم CREATE TABLE اجرا نمی‌کنه؛
    فقط وجودش رو verify می‌کنه. ساخت جدول جزو migration/setup یک‌باره‌ست.

    Expected table:

        public.eval_log
            id                  BIGSERIAL PRIMARY KEY
            chat_id             TEXT
            question            TEXT
            faithfulness_score  INTEGER
            relevance_score     INTEGER
            confidence_score    INTEGER
            grounded            BOOLEAN
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT to_regclass('public.eval_log');
                    """
                )

                row = cur.fetchone()

                if not row or row[0] is None:
                    raise RuntimeError(
                        "جدول public.eval_log وجود ندارد. "
                        "آن را یک‌بار در PostgreSQL ایجاد کنید."
                    )

                logger.info(
                    "جدول public.eval_log با موفقیت پیدا شد."
                )

    except Exception:
        logger.exception(
            "بررسی schema eval_log شکست خورد."
        )
        raise


# ============================================================
# Input validation
# ============================================================

def _validate_chat_id(chat_id: str) -> str:
    """
    Validate and normalize chat_id.
    """

    if not isinstance(chat_id, str):
        raise TypeError(
            "chat_id باید از نوع str باشد."
        )

    chat_id = chat_id.strip()

    if not chat_id:
        raise ValueError(
            "chat_id نمی‌تواند خالی باشد."
        )

    if len(chat_id) > 500:
        raise ValueError(
            "chat_id بیش از حد طولانی است."
        )

    return chat_id


def _validate_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Validate the conversation message structure.

    We intentionally keep this validation lightweight because
    messages can contain user/assistant/tool/system fields.
    """

    if not isinstance(messages, list):
        raise TypeError(
            "messages باید از نوع list باشد."
        )

    validated: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"پیام شماره {index} باید dict باشد."
            )

        if "role" not in message:
            raise ValueError(
                f"پیام شماره {index} فاقد role است."
            )

        validated.append(dict(message))

    return validated


# ============================================================
# Load memory
# ============================================================

def load_messages(
    chat_id: str,
) -> list[dict[str, Any]] | None:
    """
    Load the complete conversation history for a chat.

    Returns:
        list[dict] -> existing conversation
        None       -> chat does not exist

    Database errors are raised instead of silently converting them
    to None. This is intentional: otherwise a database failure looks
    exactly like a new conversation and causes follow-up questions
    such as "چرا؟" to mysteriously lose their context.
    """

    chat_id = _validate_chat_id(chat_id)

    with get_conn() as conn:
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:

            cur.execute(
                """
                SELECT messages
                FROM public.chat_memory
                WHERE chat_id = %s;
                """,
                (chat_id,),
            )

            row = cur.fetchone()

            if row is None:
                logger.info(
                    "حافظه‌ای برای chat_id='%s' پیدا نشد.",
                    chat_id,
                )
                return None

            messages = row["messages"]

            if messages is None:
                return []

            if not isinstance(messages, list):
                logger.warning(
                    "messages ذخیره‌شده برای chat_id='%s "
                    "ساختار list ندارد.",
                    chat_id,
                )
                return []

            logger.info(
                "حافظه‌ی چت '%s' بارگذاری شد: %d پیام.",
                chat_id,
                len(messages),
            )

            return messages


# ============================================================
# Save memory
# ============================================================

def save_messages(
    chat_id: str,
    messages: list[dict[str, Any]],
) -> None:
    """
    Insert or update the complete conversation history.

    Uses PostgreSQL UPSERT so one chat_id corresponds to exactly
    one memory record.
    """

    chat_id = _validate_chat_id(chat_id)
    messages = _validate_messages(messages)

    with get_conn() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO public.chat_memory (
                    chat_id,
                    messages,
                    updated_at
                )
                VALUES (
                    %s,
                    %s,
                    now()
                )
                ON CONFLICT (chat_id)
                DO UPDATE SET
                    messages = EXCLUDED.messages,
                    updated_at = now();
                """,
                (
                    chat_id,
                    psycopg2.extras.Json(messages),
                ),
            )

    logger.info(
        "حافظه‌ی چت '%s' ذخیره شد: %d پیام.",
        chat_id,
        len(messages),
    )


# ============================================================
# Delete memory
# ============================================================

def delete_messages(chat_id: str) -> None:
    """
    Delete one conversation memory.

    Useful for starting a completely fresh conversation.
    """

    chat_id = _validate_chat_id(chat_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM public.chat_memory
                WHERE chat_id = %s;
                """,
                (chat_id,),
            )

    logger.info(
        "حافظه‌ی چت '%s' حذف شد.",
        chat_id,
    )


# ============================================================
# Evaluation logging (faithfulness / relevance / confidence)
# ============================================================
#
# نکته‌ی مهم درباره‌ی calibration:
# calibration یک معیار per-response نیست -- از روی یک جواب تنها نمی‌شه
# گفت مدل "calibrated" هست یا نه. این معیار فقط با جمع‌آوریِ
# (confidence_score, faithfulness_score) در طول زمان و مقایسه‌ی آماری‌شون
# معنی پیدا می‌کنه: آیا جواب‌هایی که مدل بهشون مثلاً ۸۰٪ اطمینان داده،
# واقعاً حدود ۸۰٪‌شون faithful/درست از آب در اومدن؟
#
# پس اینجا دو تابع جداست:
#   log_evaluation      -- بعد از هر turn (چه تک‌سوالی چه چندبخشی) یک
#                          سطر در public.eval_log ثبت می‌کنه. Best-effort:
#                          اگه جدول نبود یا insert شکست خورد، فقط لاگ
#                          می‌شه و کل درخواست کاربر رو خراب نمی‌کنه.
#   compute_calibration -- یک تابع تجمعی/آفلاین که از روی لاگ‌های ثبت‌شده
#                          calibration رو حساب می‌کنه. این تابع در مسیر
#                          داغِ هر درخواست صدا زده نمی‌شه -- جایی جدا
#                          (مثلاً یک اسکریپت گزارش‌گیری دوره‌ای، یا یک
#                          endpoint ادمین) صداش بزن.


def log_evaluation(
    chat_id: str | None,
    question: str,
    validation: dict[str, Any],
) -> None:
    """
    یک سطر ارزیابی برای یک turn ثبت می‌کنه.

    اگه validation خاموش بوده (validation.get("skipped")) یا امتیازها
    None بودن (خودِ تماس ممیزی شکست خورده -- نگاه کن به
    audit.py::validate_answer)، چیزی ثبت نمی‌شه -- چون داده‌ی معناداری
    برای لاگ کردن وجود نداره.

    Best-effort: هر خطایی (جدول نبودن، اتصال قطع بودن، ...) فقط لاگ
    می‌شه؛ لاگ کردن ارزیابی هیچ‌وقت نباید جواب کاربر رو خراب کنه.
    """

    if not validation or validation.get("skipped"):
        return

    faithfulness_score = validation.get("faithfulness_score")
    relevance_score = validation.get("relevance_score")
    confidence_score = validation.get("confidence_score")

    if faithfulness_score is None and confidence_score is None:
        # یعنی خودِ تماس ممیزی شکست خورده (validate_answer فیل-سیف
        # None برگردونده) -- چیزی برای لاگ کردن نداریم.
        return

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.eval_log (
                        chat_id,
                        question,
                        faithfulness_score,
                        relevance_score,
                        confidence_score,
                        grounded
                    )
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        chat_id,
                        question,
                        faithfulness_score,
                        relevance_score,
                        confidence_score,
                        validation.get("grounded"),
                    ),
                )

        logger.info(
            "ارزیابی برای chat_id='%s' ثبت شد "
            "(faithfulness=%s, relevance=%s, confidence=%s).",
            chat_id,
            faithfulness_score,
            relevance_score,
            confidence_score,
        )

    except Exception:  # noqa: BLE001 - لاگ کردن ارزیابی نباید درخواست رو خراب کنه
        logger.exception(
            "ثبت ارزیابی برای chat_id='%s' شکست خورد.",
            chat_id,
        )


def compute_calibration(
    chat_id: str | None = None,
    bucket_size: int = 10,
) -> dict[str, Any]:
    """
    calibration رو به‌صورت تجمعی از روی سطرهای ثبت‌شده در
    public.eval_log حساب می‌کنه.

    چون هیچ برچسب "درست/غلط"ی از انسان نداریم، از faithfulness_score
    به‌عنوان proxy برای "درستیِ واقعی" استفاده می‌کنیم (چون خودش قبلاً
    توسط یک ممیز مستقل -- نه خودِ مدلی که جواب داده -- حساب شده).
    calibration خوب یعنی: میانگین confidence_score هر بازه‌ی (bucket)
    نزدیک به میانگین faithfulness_score همون بازه باشه.

    خروجی شامل:
        sample_count             -- تعداد کل سطرهای استفاده‌شده
        mean_confidence          -- میانگین کلی confidence_score
        mean_faithfulness        -- میانگین کلی faithfulness_score
        mean_absolute_calibration_error
                                  -- میانگین |confidence - faithfulness|
                                     به ازای هر بازه (عدد کوچیک‌تر = بهتر)
        buckets                  -- لیست {range, count, mean_confidence,
                                     mean_faithfulness} برای هر بازه --
                                     برای رسم یک reliability diagram

    اگه chat_id داده بشه، فقط همون مکالمه؛ وگرنه کل تاریخچه.
    این تابع در مسیر داغِ هیچ درخواستی صدا زده نمی‌شه -- جدا (مثلاً یک
    اسکریپت گزارش‌گیری دوره‌ای) صداش بزن.
    """

    query = """
        SELECT confidence_score, faithfulness_score
        FROM public.eval_log
        WHERE confidence_score IS NOT NULL
          AND faithfulness_score IS NOT NULL
    """
    params: tuple = ()

    if chat_id:
        query += " AND chat_id = %s"
        params = (chat_id,)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        return {
            "sample_count": 0,
            "mean_confidence": None,
            "mean_faithfulness": None,
            "mean_absolute_calibration_error": None,
            "buckets": [],
        }

    bucket_map: dict[int, list[tuple[int, int]]] = {}

    for confidence, faithfulness in rows:
        bucket_start = (int(confidence) // bucket_size) * bucket_size
        bucket_map.setdefault(bucket_start, []).append(
            (int(confidence), int(faithfulness))
        )

    buckets = []
    abs_errors = []

    for bucket_start in sorted(bucket_map):
        pairs = bucket_map[bucket_start]
        mean_conf = sum(c for c, _ in pairs) / len(pairs)
        mean_faith = sum(f for _, f in pairs) / len(pairs)
        abs_errors.append(abs(mean_conf - mean_faith))

        buckets.append(
            {
                "range": f"{bucket_start}-{bucket_start + bucket_size - 1}",
                "count": len(pairs),
                "mean_confidence": round(mean_conf, 1),
                "mean_faithfulness": round(mean_faith, 1),
            }
        )

    all_confidence = [c for c, _ in rows]
    all_faithfulness = [f for _, f in rows]

    return {
        "sample_count": len(rows),
        "mean_confidence": round(sum(all_confidence) / len(rows), 1),
        "mean_faithfulness": round(sum(all_faithfulness) / len(rows), 1),
        "mean_absolute_calibration_error": round(
            sum(abs_errors) / len(abs_errors), 1
        ),
        "buckets": buckets,
    }


# ============================================================
# Conversation compaction
# ============================================================

def _split_into_turns(
    messages: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
]:
    """
    Split conversation into:

        leading system messages

        turns:
            user
            assistant/tool...
            user
            assistant/tool...
            ...

    Every turn starts with a user message.
    """

    leading_system: list[dict[str, Any]] = []

    i = 0

    while (
        i < len(messages)
        and messages[i].get("role") == "system"
    ):
        leading_system.append(messages[i])
        i += 1

    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for message in messages[i:]:

        role = message.get("role")

        if role == "user":

            if current:
                turns.append(current)

            current = [message]

        else:
            current.append(message)

    if current:
        turns.append(current)

    return leading_system, turns


def _render_turn(
    turn: list[dict[str, Any]],
) -> str:
    """
    یک turn را برای compaction به متن فشرده ولی
    context-preserving تبدیل می‌کند.

    برخلاف نسخه‌ی قبلی، اطلاعات ساختاری مهم tool result
    عمداً حفظ می‌شوند:
        product_id
        title
        metric
        period
        result
        comparison
    """

    lines: list[str] = []

    for msg in turn:

        role = msg.get("role")

        # -----------------------------------------------------
        # User
        # -----------------------------------------------------

        if role == "user":

            content = str(
                msg.get("content") or ""
            )

            lines.append(
                f"کاربر: {content[:1000]}"
            )

        # -----------------------------------------------------
        # Assistant
        # -----------------------------------------------------

        elif role == "assistant":

            tool_calls = msg.get(
                "tool_calls"
            )

            if tool_calls:

                calls = ", ".join(
                    tc.get("function", {}).get(
                        "name",
                        "unknown",
                    )
                    for tc in tool_calls
                )

                lines.append(
                    f"دستیار: ابزارهای {calls} را فراخوانی کرد."
                )

            else:

                content = str(
                    msg.get("content") or ""
                )

                lines.append(
                    f"دستیار: {content[:1200]}"
                )

        # -----------------------------------------------------
        # Tool
        # -----------------------------------------------------

        elif role == "tool":

            content = str(
                msg.get("content") or ""
            )

            # JSON را تا حد امکان parse کن
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None

            if isinstance(parsed, dict):

                important: dict[str, Any] = {}

                important_keys = [
                    "product_id",
                    "id",
                    "product_title",
                    "title_fa",
                    "metric",
                    "metric_label",
                    "units_sold",
                    "purchase_cnt",
                    "purchase_count",
                    "revenue",
                    "sales",
                    "period_start",
                    "period_end",
                    "start_date",
                    "end_date",
                    "comparison",
                    "change",
                    "change_pct",
                    "hit_count",
                    "summary",
                    "error",
                ]

                for key in important_keys:

                    if key in parsed:
                        important[key] = parsed[key]

                rows = parsed.get("rows")

                if isinstance(rows, list):

                    important["row_count"] = len(
                        rows
                    )

                    important["rows"] = rows[:5]

                compact_content = json.dumps(
                    important,
                    ensure_ascii=False,
                    default=str,
                )

            else:

                compact_content = content[:1200]

            lines.append(
                f"نتیجه‌ی ابزار "
                f"{msg.get('name')}: "
                f"{compact_content}"
            )

    return "\n".join(lines)


# ============================================================
# Summary prompt
# ============================================================

SUMMARY_SYSTEM_PROMPT = """
تو مسئول فشرده‌سازی حافظه‌ی یک مکالمه‌ی تحلیل کسب‌وکار هستی.

فقط یک JSON معتبر با این فرمت برگردان:

{
  "summary": "..."
}

هدف summary این است که اطلاعات لازم برای ادامه‌ی دقیق
مکالمه حفظ شود، نه اینکه صرفاً مکالمه کوتاه شود.

حتماً اطلاعات زیر را اگر در مکالمه وجود دارند حفظ کن:

1. محصول:
   - نام محصول
   - product_id

2. metric:
   - نام metric
   - مقدار metric
   - واحد metric

3. زمان:
   - period_start
   - period_end
   - نوع بازه مثل «شش ماه اخیر»، «ماه اخیر» یا تاریخ دقیق

4. نتیجه:
   - ranking
   - مقدار فروش / تعداد خرید
   - درصد تغییر
   - مقایسه با دوره‌ی قبل

5. evidence:
   - نتیجه‌ی SQL
   - شواهد مستقیم RAG
   - hit_count در صورت وجود

6. Follow-up context:
   اگر یک سؤال کوتاه مثل «چرا؟»، «دلیلش؟»،
   «همون محصول؟» بر اساس یک نتیجه‌ی قبلی قابل پاسخ است،
   context لازم برای ادامه‌ی آن را حفظ کن.

قوانین مهم:

- product_id را هرگز حذف نکن.
- تاریخ شروع و پایان را حذف نکن.
- metric و مقدار metric را حذف نکن.
- نتیجه‌ی قطعی ابزارها را از حدس یا تفسیر جدا نگه دار.
- علت قطعی تولید نکن مگر اینکه evidence مستقیم وجود داشته باشد.
- یک نظر مشتری را به‌عنوان علت قطعی تغییر فروش معرفی نکن.
- اگر RAG فقط یک complaint نشان داده، آن را فقط به‌عنوان یک complaint گزارش کن.
- خلاصه باید فشرده باشد، اما اطلاعات ساختاری بالا نباید قربانی کوتاه‌سازی شوند.

اگر خلاصه‌ی قبلی وجود دارد، آن را با اطلاعات جدید ادغام کن.
اطلاعات معتبر قبلی را بدون دلیل حذف نکن.
"""


# ============================================================
# LLM summarization
# ============================================================

def _summarize_turns(
    prior_summary: str | None,
    turns: list[list[dict[str, Any]]],
) -> str:
    """
    Summarize old conversation turns.

    If summarization fails, preserve the previous summary instead
    of destroying existing memory.
    """

    if not turns:
        return prior_summary or ""

    rendered = "\n\n".join(
        _render_turn(turn)
        for turn in turns
    )

    if prior_summary:
        user_prompt = (
            "خلاصه‌ی قبلی:\n"
            f"{prior_summary}\n\n"
            "مکالمه‌ی جدیدی که باید با خلاصه‌ی قبلی ادغام شود:\n"
            f"{rendered}"
        )
    else:
        user_prompt = rendered

    try:

        result = call_llm_json(
            SUMMARY_SYSTEM_PROMPT,
            user_prompt,
        )

        if not isinstance(result, dict):
            raise ValueError(
                "خروجی LLM برای خلاصه‌سازی dict نیست."
            )

        summary = result.get("summary", "")

        if not isinstance(summary, str):
            summary = str(summary)

        summary = summary.strip()

        if summary:
            return summary

        return prior_summary or ""

    except Exception as exc:
        logger.warning(
            "فشرده‌سازی حافظه شکست خورد؛ "
            "خلاصه‌ی قبلی حفظ می‌شود: %s",
            exc,
        )

        return prior_summary or ""


# ============================================================
# Maybe compact
# ============================================================

def maybe_compact(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compact old conversation turns when their number exceeds
    CHAT_MEMORY_MAX_RAW_TURNS.

    Example with MAX_RAW_TURNS=6:

        old turns:
            1
            2
            3
            4

        recent turns:
            5
            6
            7
            8
            9
            10

    Turns 1-4 are summarized.
    Turns 5-10 remain raw.

    If the threshold is not exceeded, the exact original list
    is returned without an LLM call.
    """

    messages = _validate_messages(messages)

    leading_system, turns = _split_into_turns(messages)

    max_raw_turns = ChatDBConfig.MAX_RAW_TURNS

    if len(turns) <= max_raw_turns:
        return messages

    # --------------------------------------------------------
    # Find an existing summary.
    # --------------------------------------------------------

    prior_summary_text: str | None = None
    other_system_messages: list[dict[str, Any]] = []

    for message in leading_system:

        content = str(
            message.get("content") or ""
        )

        if content.startswith(SUMMARY_MARKER):

            prior_summary_text = (
                content[
                    len(SUMMARY_MARKER):
                ].strip()
            )

        else:
            other_system_messages.append(
                message
            )

    # --------------------------------------------------------
    # Split old and recent turns.
    # --------------------------------------------------------

    turns_to_compact = turns[:-max_raw_turns]
    turns_to_keep = turns[-max_raw_turns:]

    # --------------------------------------------------------
    # Summarize old turns.
    # --------------------------------------------------------

    new_summary_text = _summarize_turns(
        prior_summary=prior_summary_text,
        turns=turns_to_compact,
    )

    summary_message = {
        "role": "system",
        "content": (
            f"{SUMMARY_MARKER} "
            f"{new_summary_text}"
        ),
    }

    # --------------------------------------------------------
    # Keep recent turns completely intact.
    # --------------------------------------------------------

    kept_messages = [
        message
        for turn in turns_to_keep
        for message in turn
    ]

    compacted = (
        other_system_messages
        + [summary_message]
        + kept_messages
    )

    logger.info(
        "حافظه compact شد: "
        "%d turn قدیمی خلاصه شد، "
        "%d turn اخیر حفظ شد.",
        len(turns_to_compact),
        len(turns_to_keep),
    )

    return compacted


# ============================================================
# Pool shutdown
# ============================================================

def close_pool() -> None:
    """
    Close the chat-memory connection pool.

    Useful for application shutdown/tests.
    """

    global _pool

    if _pool is not None:

        try:
            _pool.closeall()

        finally:
            _pool = None

        logger.info(
            "connection pool حافظه‌ی چت بسته شد."
        )