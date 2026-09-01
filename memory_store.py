## PATH: app/memory_store.py
"""
ذخیره‌سازی و فشرده‌سازی حافظه‌ی مکالمه (چت) در Postgres.

چرا اینجا (app/) و نه app/graph/؟
-----------------------------------
app/graph/ فقط منطق استدلال (agent/tools/finalize/validate) رو داره و
state["messages"] براش فقط یک لیست ساده‌ست که از بیرون تزریق می‌شه -- هیچ
ایده‌ای درباره‌ی "این پیام‌ها مال کدوم چته" یا "کجا ذخیره می‌شن" نداره.
این جداسازی عمدیه: گراف بدون نیاز به دیتابیس چت قابل‌تسته (همون تست‌های
mock‌شده‌ای که قبلاً زدیم). "کدوم چت کدوم تاریخچه رو داره" یک concern
زیرساختی/لایه‌ی اپلیکیشنه، نه بخشی از گراف استدلال.

چرا یک connection/pool کاملاً جدا از app/graph/db.py؟
-------------------------------------------------------
db.py عمداً روی هر کانکشن `default_transaction_read_only = on` می‌زنه
(تا حتی اگه validation در sql_agent.py هم رد بشه، tool_sql فیزیکاً
نتونه چیزی بنویسه). ذخیره‌ی حافظه‌ی چت برعکسِ این نیازه -- INSERT/UPDATE
واقعی می‌خواد. پس این فایل یک pool و یک role کاملاً مجزا (write-capable)
استفاده می‌کنه؛ هرگز نباید از role فقط-خواندنیِ tool_sql براش استفاده
بشه.

استراتژی فشرده‌سازی (وقتی تاریخچه زیاد شد):
----------------------------------------------
مکالمه به "دور" (turn) تقسیم می‌شه: هر دور از یک پیام user شروع می‌شه و
تمام assistant/tool های بعدش رو تا قبل از user بعدی شامل می‌شه. وقتی
تعداد دورها از CHAT_MEMORY_MAX_RAW_TURNS بیشتر بشه، دورهای قدیمی‌تر (همه
جز آخرین N تا) با یک تماس LLM به یک "خلاصه‌ی تجمعی" تبدیل می‌شن -- اگه
از قبل هم یک خلاصه موجود بود، جدید و قدیمی با هم ادغام می‌شن (نه این‌که
هر بار یک پیام خلاصه‌ی جدید روی هم انباشته بشه). این تنها جایی‌ست که
حافظه‌ی مکالمه از یک تماس LLM استفاده می‌کنه -- چون فشرده‌سازی مکالمه
(برخلاف خلاصه‌سازی نظرات در text_summary.py) به فهم معنایی نیاز داره، نه
فقط فرکانس کلمات، و این فقط وقتی سقف رد بشه اتفاق می‌افته، نه هر پیام.

نکته‌ی امنیتی مهم: چون خودِ run() در main.py تضمین می‌کنه هر چیزی که در
DB ذخیره می‌شه، خروجی نهایی و کامل یک اجرای گراف (بدون tool_call معلق)
است، هیچ‌وقت وسط یک تبادل ابزار قطع نمی‌شه -- پس تقسیم به "دور" همیشه
ایمنه.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from graph.llm_client import call_llm_json

logger = logging.getLogger(__name__)


class ChatDBConfig:
    # عمداً از یک namespace متغیر محیطی کاملاً جدا از DB_* (در app/graph/db.py)
    # استفاده می‌شه تا هیچ‌وقت به‌اشتباه با همون role فقط-خواندنی قاطی نشه.
    DB_NAME = os.getenv("CHAT_DB_NAME", os.getenv("DB_NAME", "ai_project"))
    DB_USER = os.getenv("CHAT_DB_USER", "app_chat_writer")  # باید GRANT INSERT/UPDATE داشته باشه
    DB_PASSWORD = os.getenv("CHAT_DB_PASSWORD", "")
    DB_HOST = os.getenv("CHAT_DB_HOST", os.getenv("DB_HOST", "localhost"))
    DB_PORT = os.getenv("CHAT_DB_PORT", os.getenv("DB_PORT", "5432"))

    POOL_MIN_CONN = int(os.getenv("CHAT_PG_POOL_MIN", "1"))
    POOL_MAX_CONN = int(os.getenv("CHAT_PG_POOL_MAX", "5"))

    # بعد از این تعداد "دور" مکالمه، دورهای قدیمی‌تر (به‌جز همین‌قدر آخر)
    # خلاصه می‌شن.
    MAX_RAW_TURNS = int(os.getenv("CHAT_MEMORY_MAX_RAW_TURNS", "6"))


SUMMARY_MARKER = "[خلاصه‌ی مکالمات قبلی]"

_pool: ThreadedConnectionPool | None = None


def get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info("در حال ساخت connection pool نوشتنی برای حافظه‌ی چت...")
        _pool = ThreadedConnectionPool(
            ChatDBConfig.POOL_MIN_CONN,
            ChatDBConfig.POOL_MAX_CONN,
            dbname=ChatDBConfig.DB_NAME,
            user=ChatDBConfig.DB_USER,
            password=ChatDBConfig.DB_PASSWORD,
            host=ChatDBConfig.DB_HOST,
            port=ChatDBConfig.DB_PORT,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ============================================================
# راه‌اندازی جدول -- یک‌بار در استقرار/migration صدا بزن، نه در مسیر
# داغ هر درخواست (CREATE TABLE معمولاً دسترسی بالاتری از role نوشتنِ
# معمولی می‌خواد).
# ============================================================

def ensure_schema() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_memory (
                    chat_id     TEXT PRIMARY KEY,
                    messages    JSONB NOT NULL,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )


# ============================================================
# خواندن / نوشتن
# ============================================================

def load_messages(chat_id: str) -> list[dict[str, Any]] | None:
    """اگه چتی با این chat_id قبلاً ذخیره نشده باشه، None برمی‌گردونه --
    یعنی مکالمه‌ی تازه. خطای اتصال هم به None تبدیل می‌شه (نه crash) تا
    یک مشکل موقت DB، کل درخواست کاربر رو نندازه."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT messages FROM chat_memory WHERE chat_id = %s;", (chat_id,))
                row = cur.fetchone()
                return row["messages"] if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_messages: خواندن حافظه‌ی چت '%s' شکست خورد: %s", chat_id, exc)
        return None


def save_messages(chat_id: str, messages: list[dict[str, Any]]) -> None:
    """شکست در ذخیره‌سازی فقط لاگ می‌شه -- نباید جواب آماده‌ی کاربر رو
    به‌خاطر یک خطای زیرساختی از بین ببره (فراخوان مسئول تصمیم‌گیری
    درباره‌ی retry/alert است)."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_memory (chat_id, messages, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (chat_id)
                    DO UPDATE SET messages = EXCLUDED.messages, updated_at = now();
                    """,
                    (chat_id, psycopg2.extras.Json(messages)),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_messages: ذخیره‌ی حافظه‌ی چت '%s' شکست خورد: %s", chat_id, exc)


# ============================================================
# فشرده‌سازی (compaction)
# ============================================================

def _split_into_turns(
    messages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """پیام‌های system ابتدایی (system prompt + خلاصه‌ی قبلی اگه باشه) رو
    از "دور"های user/assistant/tool جدا می‌کنه."""
    leading_system: list[dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        leading_system.append(messages[i])
        i += 1

    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in messages[i:]:
        if msg.get("role") == "user":
            if current:
                turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)

    return leading_system, turns


def _render_turn(turn: list[dict[str, Any]]) -> str:
    """یک دور رو به متن خوانا برای پرامپت خلاصه‌سازی تبدیل می‌کنه."""
    lines: list[str] = []
    for msg in turn:
        role = msg.get("role")
        if role == "user":
            lines.append(f"کاربر: {msg.get('content')}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                lines.append(f"دستیار: (ابزار(های) {calls} را فراخوانی کرد)")
            else:
                lines.append(f"دستیار: {msg.get('content')}")
        elif role == "tool":
            content = str(msg.get("content", ""))[:300]
            lines.append(f"نتیجه‌ی ابزار {msg.get('name')}: {content}")
    return "\n".join(lines)


SUMMARY_SYSTEM_PROMPT = """
تو داری تاریخچه‌ی یک مکالمه‌ی تحلیل کسب‌وکار رو خلاصه می‌کنی تا در
حافظه‌ی محدود نگه داشته بشه. فقط یک JSON با این فرمت برگردون:

{"summary": "<خلاصه‌ی فشرده به فارسی>"}

نکات مهم:
- فقط واقعیت‌ها/نتیجه‌گیری‌های قطعی رو نگه دار (مثلاً "فروش محصول X در
  اسفند ۱۲٪ کاهش داشت"، "علت اصلی شکایات: باتری")، نه جزئیات گفت‌وگو.
- اگه خلاصه‌ی قبلی داده شده، اون رو با اطلاعات جدید ادغام کن -- چیزی که
  در خلاصه‌ی قبلی بوده رو حذف نکن مگر مکالمه‌ی جدید نقضش کرده باشه.
- خلاصه باید کوتاه بمونه (حداکثر چند جمله به ازای هر موضوع).
"""


def _summarize_turns(prior_summary: str | None, turns: list[list[dict[str, Any]]]) -> str:
    rendered = "\n\n".join(_render_turn(t) for t in turns)
    user_prompt = rendered
    if prior_summary:
        user_prompt = f"خلاصه‌ی قبلی:\n{prior_summary}\n\nمکالمه‌ی جدیدی که باید باهاش ادغام بشه:\n{rendered}"

    try:
        result = call_llm_json(SUMMARY_SYSTEM_PROMPT, user_prompt)
        return result.get("summary", "") or (prior_summary or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("_summarize_turns: تماس LLM شکست خورد، خلاصه‌ی قبلی حفظ می‌شه: %s", exc)
        # بدترین حالت: خلاصه رو گم نمی‌کنیم، فقط آپدیت نمی‌شه.
        return prior_summary or "(خلاصه‌سازی این بخش موقتاً ناموفق بود)"


def maybe_compact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    اگه تعداد "دور"های مکالمه از CHAT_MEMORY_MAX_RAW_TURNS بیشتر باشه،
    همه‌ی دورهای قدیمی‌تر (به‌جز آخرین N تا) رو در یک پیام system خلاصه
    می‌کنه و بقیه رو دست‌نخورده نگه می‌داره. اگه از سقف رد نشده باشه،
    همون messages بدون تغییر برمی‌گرده (هزینه‌ی این تابع صفره).
    """
    leading_system, turns = _split_into_turns(messages)

    if len(turns) <= ChatDBConfig.MAX_RAW_TURNS:
        return messages

    # اگه از قبل یک خلاصه در پیام‌های system بوده، پیداش کن و جداش کن --
    # قراره جایگزین بشه با نسخه‌ی ادغام‌شده‌ی جدید، نه این‌که کنارش انباشته بشه.
    prior_summary_text: str | None = None
    other_system_msgs: list[dict[str, Any]] = []
    for msg in leading_system:
        content = msg.get("content") or ""
        if content.startswith(SUMMARY_MARKER):
            prior_summary_text = content[len(SUMMARY_MARKER):].strip()
        else:
            other_system_msgs.append(msg)

    turns_to_compact = turns[: -ChatDBConfig.MAX_RAW_TURNS]
    turns_to_keep = turns[-ChatDBConfig.MAX_RAW_TURNS:]

    new_summary_text = _summarize_turns(prior_summary_text, turns_to_compact)
    summary_msg = {"role": "system", "content": f"{SUMMARY_MARKER} {new_summary_text}"}

    kept_messages = [m for turn in turns_to_keep for m in turn]
    return other_system_msgs + [summary_msg] + kept_messages
