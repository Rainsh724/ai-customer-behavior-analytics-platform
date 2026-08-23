## PATH: app/graph/db.py
"""
اتصال واقعی به Postgres (همون دیتابیسی که lod_data_to_database2.py توش دیتا
می‌ریزه: users, cities, brands, categories, sellers, sessions, products,
user_behavior_logs, comments (+ embedded_comment با pgvector), comment_aspects)

نکات امنیتی/عملیاتی که باید رعایت بشه:
- کاربر دیتابیسی که sql_agent باهاش وصل میشه باید فقط GRANT SELECT داشته باشه
  (یک role جدا بساز، نه همون postgres). این از "SQL Injection از طریق LLM"
  (یعنی LLM یه DROP/DELETE بنویسه) جلوگیری می‌کنه حتی اگه validation ما رد بشه.
- statement_timeout ست میشه که یک کوئری بد کل سرویس رو قفل نکنه.
- Connection pooling چون هر request یک conn جدید نمی‌سازه.
"""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)


class DBConfig:
    DB_NAME = os.getenv("DB_NAME", "ai_project")
    # کاربر read-only مخصوص لایه LLM/آنالیتیکس -- نه superuser
    DB_USER = os.getenv("DB_READONLY_USER", "app_readonly")
    DB_PASSWORD = os.getenv("DB_READONLY_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    STATEMENT_TIMEOUT_MS = int(os.getenv("SQL_STATEMENT_TIMEOUT_MS", "8000"))
    POOL_MIN_CONN = int(os.getenv("PG_POOL_MIN", "1"))
    POOL_MAX_CONN = int(os.getenv("PG_POOL_MAX", "10"))

    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # matches comments_embedding.embedded_comment VECTOR(768)


_pool: ThreadedConnectionPool | None = None


def get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info("در حال ساخت connection pool به Postgres...")
        _pool = ThreadedConnectionPool(
            DBConfig.POOL_MIN_CONN,
            DBConfig.POOL_MAX_CONN,
            dbname=DBConfig.DB_NAME,
            user=DBConfig.DB_USER,
            password=DBConfig.DB_PASSWORD,
            host=DBConfig.DB_HOST,
            port=DBConfig.DB_PORT,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {DBConfig.STATEMENT_TIMEOUT_MS};")
            # فقط SELECT مجازه -- دفاع دوم بعد از validation در sql_agent
            cur.execute("SET default_transaction_read_only = on;")
        yield conn
    finally:
        pool.putconn(conn)


def run_readonly_query(sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
    """
    اجرای یک کوئری SELECT و برگردوندن نتیجه به‌صورت لیستی از dict.
    فقط برای کوئری‌های read-only استفاده بشه (validation قبلش انجام میشه).
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def vector_similarity_search(
    query_embedding: list[float],
    where_sql: str,
    where_params: tuple,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    """
    جست‌وجوی شباهت برداری روی comments.embedded_comment با pgvector.
    از عملگر <=> (cosine distance) استفاده می‌کنه که همون ایندکس hnsw
    ساخته‌شده در index.py رو به کار می‌گیره:
        CREATE INDEX ... ON comments USING hnsw (embedded_comment vector_cosine_ops);

    where_sql: شرط‌های اضافی (متادیتا فیلتر) که از retrieval_planner میاد،
               مثلاً "c.product_id = %s AND c.rate <= %s"
    """
    embedding_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    sql = f"""
        SELECT
            c.id                      AS comment_id,
            c.product_id,
            c.rate,
            c.recommendation_status,
            c.likes,
            c.dislikes,
            c.raw_text_normalized,
            c.created_at,
            p.title_fa                AS product_title,
            (ce.embedded_comment <=> %s::vector) AS distance
        FROM comments c
        JOIN comments_embedding ce ON ce.id = c.id
        JOIN products p ON p.id = c.product_id
        WHERE ce.embedded_comment IS NOT NULL
        {("AND " + where_sql) if where_sql else ""}
        ORDER BY ce.embedded_comment <=> %s::vector
        LIMIT %s;
    """
    params = (embedding_literal, *where_params, embedding_literal, top_k)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
