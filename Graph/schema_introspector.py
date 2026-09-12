## PATH: app/graph/schema_introspector.py
"""
جایگزین sync دستیِ schema/allowed_join_keys در production_validator.py.

دو تا چیز رو مستقیم از خودِ Postgres می‌خونه:
  1. ستون‌های هر جدول (information_schema.columns) -> جای self.schema رو می‌گیره.
  2. FK های واقعاً تعریف‌شده روی دیتابیس (information_schema.table_constraints) ->
     پایه‌ی one_to_many_map و allowed_join_keys رو می‌سازه.

نکته‌ی مهم: اگه جداول analytics/kpi شما (که معمولاً خروجی ETL/feature-store
هستن) روی دیتابیس FK constraint واقعی نداشته باشن -- که در این‌جور جدول‌ها
خیلی رایجه -- این تابع برای اون‌ها چیزی برنمی‌گردونه. در اون صورت باید اون
join pair خاص رو در MANUAL_EXTRA_* (در production_validator.py) نگه دارید.
این طبیعیه: تفاوتش با وضعیت الان اینه که دیگه لازم نیست *ستون‌ها* رو دستی
sync کنید -- فقط *join pair*های بدون FK واقعی رو، که تعدادشون خیلی کمتره و
با هر ستون جدید عوض نمی‌شه.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SCHEMAS = ("public", "analytics", "kpi")


def load_columns_schema(
    conn, schemas: Tuple[str, ...] = DEFAULT_SCHEMAS
) -> Dict[str, Dict[str, List[str]]]:
    """{db: {table: [col1, col2, ...]}} مستقیم از information_schema."""
    query = """
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = ANY(%s)
        ORDER BY table_schema, table_name, ordinal_position;
    """
    schema: Dict[str, Dict[str, List[str]]] = {}
    with conn.cursor() as cur:
        cur.execute(query, (list(schemas),))
        for table_schema, table_name, column_name in cur.fetchall():
            schema.setdefault(table_schema, {}).setdefault(table_name, []).append(
                column_name
            )
    return schema


def load_fk_joins(
    conn, schemas: Tuple[str, ...] = DEFAULT_SCHEMAS
) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Set[Tuple[str, str]]]]:
    """
    از روی FK constraint های واقعی، (one_to_many_map, allowed_join_keys) می‌سازه.
    جهت parent/child از خودِ FK مشخصه: جدولی که ستونش REFERENCES می‌کنه = child
    (سمت many)، جدولی که ارجاع بهش داده می‌شه = parent (سمت one).
    """
    query = """
        SELECT
            tc.table_schema  AS child_schema,
            tc.table_name    AS child_table,
            kcu.column_name  AS child_column,
            ccu.table_schema AS parent_schema,
            ccu.table_name   AS parent_table,
            ccu.column_name  AS parent_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
         AND tc.table_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = ANY(%s);
    """
    one_to_many: Set[Tuple[str, str]] = set()
    allowed_keys: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    with conn.cursor() as cur:
        cur.execute(query, (list(schemas),))
        rows = cur.fetchall()
        for child_schema, child_table, child_col, parent_schema, parent_table, parent_col in rows:
            parent = f"{parent_schema}.{parent_table}"
            child = f"{child_schema}.{child_table}"
            one_to_many.add((parent, child))
            allowed_keys.setdefault((parent, child), set()).add(
                (parent_col, child_col)
            )
    if not rows:
        logger.warning(
            "هیچ FK constraint واقعی‌ای در schemas=%s پیدا نشد -- اگه انتظار "
            "داشتید (مثلاً روی public) دارید، یعنی این جداول FK ندارن و باید "
            "join هاشون در MANUAL_EXTRA_JOIN_KEYS دستی بمونه.",
            schemas,
        )
    return one_to_many, allowed_keys


def introspect_all(conn, schemas: Tuple[str, ...] = DEFAULT_SCHEMAS):
    """یک‌جا هر سه تا رو برمی‌گردونه: (schema, one_to_many_map, allowed_join_keys)."""
    schema = load_columns_schema(conn, schemas)
    one_to_many, allowed_keys = load_fk_joins(conn, schemas)
    return schema, one_to_many, allowed_keys