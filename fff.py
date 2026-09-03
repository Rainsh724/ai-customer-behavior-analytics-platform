


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope


# ======================================================================
# 1. تعریف schema فیزیکی (منطبق بر DDL واقعی شما)
# ======================================================================

SCHEMA: Dict[str, Dict[str, List[str]]] = {
    "public": {
        "cities": ["city_id", "name"],
        "users": ["user_id"],
        "sessions": ["session_id", "user_id", "city_id"],
        "brands": ["brand_id", "name"],
        "categories": ["category_id", "category1", "category2", "sub_category"],
        "sellers": ["seller_id", "seller_title"],
        "products": [
            "id", "title_fa", "brand_id", "category_id", "seller_id",
            "price", "min_price_last_month", "is_fake", "rate", "rate_cnt",
        ],
        "user_behavior_logs": [
            "log_id", "session_id", "product_id", "event_type", "timestamp",
        ],
        "comments": [
            "id", "product_id", "is_buyer", "rate", "recommendation_status",
            "likes", "dislikes", "raw_text_normalized", "created_at",
        ],
        "comments_embedding": ["id", "embedded_comment"],
        "comment_aspects": [
            "aspect_id", "comment_id", "term", "sentiment",
            "negative_pct", "neutral_pct", "positive_pct",
        ],
    },
    "analytics": {
        "feature_behavior": [
            "log_id", "hour", "day", "month", "weekday", "is_weekend",
            "is_view", "is_cart", "is_remove", "is_purchase",
        ],
        "feature_user": [
            "user_id", "total_events", "total_sessions", "active_days",
            "total_views", "total_cart_adds", "total_removes", "total_purchases",
            "avg_session_events", "max_session_events",
            "avg_session_duration_minutes", "max_session_duration_minutes",
            "weekend_activity_ratio", "morning_activity_ratio",
            "afternoon_activity_ratio", "evening_activity_ratio",
            "night_activity_ratio", "preferred_hour", "preferred_weekday",
            "unique_products_viewed", "unique_products_purchased",
            "cities_visited", "total_spend", "avg_purchase_value",
            "min_purchase_price", "max_purchase_price", "purchase_frequency",
            "purchase_days", "brand_diversity", "category_diversity",
        ],
        "feature_product": [
            "product_id", "total_events", "total_views", "total_cart_adds",
            "total_removes", "total_purchases", "unique_viewers",
            "unique_carters", "unique_buyers", "total_sessions", "price_drop_ratio",
        ],
        "feature_city": [
            "city_id", "total_users", "total_sessions", "total_events",
            "total_views", "total_cart_adds", "total_purchases", "total_removes",
            "unique_products_viewed", "unique_products_purchased",
        ],
        "feature_category": [
            "category_id", "total_events", "total_views", "total_cart_adds",
            "total_purchases", "total_removes", "unique_viewers",
            "unique_buyers", "avg_product_price",
        ],
        "feature_brand": [
            "brand_id", "total_events", "total_views", "total_cart_adds",
            "total_purchases", "total_removes", "unique_viewers", "unique_buyers",
        ],
        "feature_user_product": [
            "user_id", "product_id", "total_events", "view_count", "cart_count",
            "remove_count", "purchase_count", "active_days", "session_count",
        ],
        "feature_user_category": [
            "user_id", "category_id", "total_events", "view_count", "cart_count",
            "remove_count", "purchase_count", "category_spend",
            "view_share", "purchase_share", "spend_share",
        ],
        "feature_product_sentiment": [
            "product_id", "comment_count", "avg_rate", "avg_like_ratio",
            "total_likes", "total_dislikes", "total_aspect_mentions",
            "positive_aspect_mentions", "negative_aspect_mentions",
            "neutral_aspect_mentions", "avg_positive_pct", "avg_negative_pct",
            "avg_neutral_pct", "positive_aspect_ratio", "negative_aspect_ratio",
            "neutral_aspect_ratio",
        ],
        "feature_product_aspect": [
            "product_id", "term", "total_mentions", "positive_mentions",
            "negative_mentions", "neutral_mentions", "avg_negative_pct",
            "avg_neutral_pct", "avg_positive_pct",
        ],
        "feature_brand_sentiment": [
            "brand_id", "total_comments", "total_aspect_mentions",
            "positive_aspect_mentions", "negative_aspect_mentions",
            "neutral_aspect_mentions", "avg_comment_rating",
            "total_likes", "total_dislikes",
        ],
        "feature_category_sentiment": [
            "category_id", "total_comments", "total_aspect_mentions",
            "positive_aspect_mentions", "negative_aspect_mentions",
            "neutral_aspect_mentions", "avg_comment_rating",
            "total_likes", "total_dislikes",
        ],
        "feature_aspect": [
            "term", "total_mentions", "positive_mentions", "negative_mentions",
            "neutral_mentions", "avg_negative_pct", "avg_neutral_pct", "avg_positive_pct",
        ],
        "feature_time": [
            "hour", "iso_weekday", "total_events", "total_views",
            "total_cart_adds", "total_purchases", "total_removes",
        ],
    },
    "kpi": {
        "global_funnel": [
            "total_views", "total_carts", "total_purchases", "total_removes",
            "view_to_cart_pct", "cart_to_purchase_pct", "overall_conversion_pct",
            "cart_abandonment_pct",
        ],
        "product_360": [
            "product_id", "title_fa", "price", "total_views", "total_purchases",
            "total_revenue", "conversion_rate", "comment_count", "star_rating",
            "positive_sentiment_pct", "sentiment_score", "managerial_action_tag",
        ],
        "user_segments": [
            "user_id", "active_days", "total_views", "total_purchases",
            "total_spend", "user_segment", "user_conversion_pct",
        ],
        "brand_diagnostics": [
            "brand_id", "brand_name", "total_views", "total_purchases",
            "total_comments", "avg_rating", "brand_sentiment_score",
        ],
        "aspect_diagnostics": [
            "aspect_name", "total_mentions", "positive_mentions",
            "negative_mentions", "negative_impact_pct", "aspect_status",
        ],
        "rfm_segments": [
            "user_id", "recency_days", "frequency", "monetary",
            "rfm_code", "rfm_label",
        ],
        "ml_user_clusters": ["user_id", "cluster_id", "cluster_name"],
        "daily_funnel": [
            "event_date", "total_views", "total_carts", "total_purchases",
            "total_removes", "view_to_cart_pct", "cart_to_purchase_pct",
            "overall_conversion_pct", "cart_abandonment_pct",
        ],
        "top_products_30d": [
            "product_id", "product_name", "price", "total_views_30d",
            "total_carts_30d", "total_purchases_30d", "total_removes_30d",
            "total_revenue_30d", "conversion_rate_30d",
        ],
        "top_brands_30d": [
            "brand_id", "brand_name", "total_views_30d", "total_carts_30d",
            "total_purchases_30d", "total_removes_30d", "total_revenue_30d",
            "conversion_rate_30d",
        ],
    },
}


# ======================================================================
# 2. منبع واحد حقیقت: همه‌ی رابطه‌های واقعی بین جدول‌ها
#    (parent_table, parent_col, child_table, child_col, cardinality)
#    cardinality:
#       "one_to_many"  -> ریسک fan-out واقعی دارد (باید محافظت شود)
#       "one_to_one"   -> هم‌گرَن هستند (مثلاً feature_product با
#                         products)، join آزاد و بی‌خطر است.
# ======================================================================

Cardinality = str  # "one_to_many" | "one_to_one"

RELATIONSHIPS: List[Tuple[str, str, str, str, Cardinality]] = [
    # ---- روابط اصلی OLTP (طبق DDL) ----
    ("public.cities", "city_id", "public.sessions", "city_id", "one_to_many"),
    ("public.users", "user_id", "public.sessions", "user_id", "one_to_many"),
    ("public.sessions", "session_id", "public.user_behavior_logs", "session_id", "one_to_many"),
    ("public.products", "id", "public.user_behavior_logs", "product_id", "one_to_many"),
    ("public.brands", "brand_id", "public.products", "brand_id", "one_to_many"),
    ("public.categories", "category_id", "public.products", "category_id", "one_to_many"),
    ("public.sellers", "seller_id", "public.products", "seller_id", "one_to_many"),
    ("public.products", "id", "public.comments", "product_id", "one_to_many"),
    ("public.comments", "id", "public.comment_aspects", "comment_id", "one_to_many"),
    ("public.comments", "id", "public.comments_embedding", "id", "one_to_one"),

    # ---- جدول‌های analytics هم‌گرن با موجودیت اصلی (1:1) ----
    ("public.user_behavior_logs", "log_id", "analytics.feature_behavior", "log_id", "one_to_one"),
    ("public.users", "user_id", "analytics.feature_user", "user_id", "one_to_one"),
    ("public.products", "id", "analytics.feature_product", "product_id", "one_to_one"),
    ("public.cities", "city_id", "analytics.feature_city", "city_id", "one_to_one"),
    ("public.categories", "category_id", "analytics.feature_category", "category_id", "one_to_one"),
    ("public.brands", "brand_id", "analytics.feature_brand", "brand_id", "one_to_one"),
    ("public.products", "id", "analytics.feature_product_sentiment", "product_id", "one_to_one"),
    ("public.brands", "brand_id", "analytics.feature_brand_sentiment", "brand_id", "one_to_one"),
    ("public.categories", "category_id", "analytics.feature_category_sentiment", "category_id", "one_to_one"),

    # ---- جدول‌های bridge / چندبه‌یک analytics (ریسک fan-out دارند) ----
    ("public.users", "user_id", "analytics.feature_user_product", "user_id", "one_to_many"),
    ("public.products", "id", "analytics.feature_user_product", "product_id", "one_to_many"),
    ("public.users", "user_id", "analytics.feature_user_category", "user_id", "one_to_many"),
    ("public.categories", "category_id", "analytics.feature_user_category", "category_id", "one_to_many"),
    ("public.products", "id", "analytics.feature_product_aspect", "product_id", "one_to_many"),

    # ---- ارتباط مستقیم بین جدول‌های analytics با یکدیگر (بدون واسطه‌ی جدول اصلی) ----
    ("analytics.feature_product", "product_id", "analytics.feature_product_sentiment", "product_id", "one_to_one"),
    ("analytics.feature_product", "product_id", "analytics.feature_product_aspect", "product_id", "one_to_many"),
    ("analytics.feature_user", "user_id", "analytics.feature_user_product", "user_id", "one_to_many"),
    ("analytics.feature_user", "user_id", "analytics.feature_user_category", "user_id", "one_to_many"),
    ("analytics.feature_category", "category_id", "analytics.feature_user_category", "category_id", "one_to_many"),
    ("analytics.feature_category", "category_id", "analytics.feature_category_sentiment", "category_id", "one_to_one"),
    ("analytics.feature_brand", "brand_id", "analytics.feature_brand_sentiment", "brand_id", "one_to_one"),

    # ---- ویوهای kpi (از پیش تجمیع‌شده؛ هم‌گرن با موجودیت پایه) ----
    ("public.products", "id", "kpi.product_360", "product_id", "one_to_one"),
    ("public.users", "user_id", "kpi.user_segments", "user_id", "one_to_one"),
    ("public.users", "user_id", "kpi.rfm_segments", "user_id", "one_to_one"),
    ("public.users", "user_id", "kpi.ml_user_clusters", "user_id", "one_to_one"),
    ("public.brands", "brand_id", "kpi.brand_diagnostics", "brand_id", "one_to_one"),
    ("public.products", "id", "kpi.top_products_30d", "product_id", "one_to_one"),
    ("public.brands", "brand_id", "kpi.top_brands_30d", "brand_id", "one_to_one"),
    ("analytics.feature_product", "product_id", "kpi.product_360", "product_id", "one_to_one"),
    ("analytics.feature_product", "product_id", "kpi.top_products_30d", "product_id", "one_to_one"),
    ("analytics.feature_brand", "brand_id", "kpi.brand_diagnostics", "brand_id", "one_to_one"),
    ("analytics.feature_brand", "brand_id", "kpi.top_brands_30d", "brand_id", "one_to_one"),
    ("analytics.feature_user", "user_id", "kpi.user_segments", "user_id", "one_to_one"),
    ("analytics.feature_user", "user_id", "kpi.rfm_segments", "user_id", "one_to_one"),
    ("analytics.feature_aspect", "term", "kpi.aspect_diagnostics", "aspect_name", "one_to_one"),
]


# ======================================================================
# 3. ستون‌های "حساس" هر جدول والد (اگر با join به یک child تکثیر شوند
#    و SUM/AVG رویشان زده شود، عدد نهایی نادرست می‌شود)
# ======================================================================

PARENT_SENSITIVE_COLUMNS: Dict[str, Set[str]] = {
    "public.products": {"price", "min_price_last_month", "rate", "rate_cnt"},
    "public.comments": {"likes", "dislikes", "rate"},
    "analytics.feature_product": {
        "total_events", "total_views", "total_cart_adds", "total_removes",
        "total_purchases", "unique_viewers", "unique_carters", "unique_buyers",
        "total_sessions", "price_drop_ratio",
    },
    "analytics.feature_user": {
        "total_events", "total_sessions", "active_days", "total_views",
        "total_cart_adds", "total_removes", "total_purchases", "total_spend",
        "avg_purchase_value", "min_purchase_price", "max_purchase_price",
        "purchase_frequency", "purchase_days", "brand_diversity", "category_diversity",
    },
    "analytics.feature_category": {
        "total_events", "total_views", "total_cart_adds", "total_purchases",
        "total_removes", "unique_viewers", "unique_buyers", "avg_product_price",
    },
    "analytics.feature_brand": {
        "total_events", "total_views", "total_cart_adds", "total_purchases",
        "total_removes", "unique_viewers", "unique_buyers",
    },
    "kpi.product_360": {
        "total_views", "total_purchases", "total_revenue", "conversion_rate",
        "comment_count", "star_rating", "positive_sentiment_pct", "sentiment_score",
    },
    "kpi.user_segments": {
        "active_days", "total_views", "total_purchases", "total_spend", "user_conversion_pct",
    },
    "kpi.brand_diagnostics": {
        "total_views", "total_purchases", "total_comments", "avg_rating", "brand_sentiment_score",
    },
    "kpi.top_products_30d": {
        "total_views_30d", "total_carts_30d", "total_purchases_30d",
        "total_removes_30d", "total_revenue_30d", "conversion_rate_30d",
    },
    "kpi.top_brands_30d": {
        "total_views_30d", "total_carts_30d", "total_purchases_30d",
        "total_removes_30d", "total_revenue_30d", "conversion_rate_30d",
    },
}

# ستون‌های نسبتی/درصدی: SUM رویشون همیشه بی‌معنیه (بلاک می‌شود).
# AVG رویشون بلاک نمی‌شود چون سوال متداول مدیریتیه (با وجود ناخالصی آماری جزئی).
RATIO_COLUMNS: Set[str] = {
    "conversion_rate", "view_to_cart_pct", "cart_to_purchase_pct",
    "overall_conversion_pct", "cart_abandonment_pct", "positive_sentiment_pct",
    "sentiment_score", "star_rating", "positive_aspect_ratio",
    "negative_aspect_ratio", "neutral_aspect_ratio", "price_drop_ratio",
    "view_share", "purchase_share", "spend_share", "avg_positive_pct",
    "avg_negative_pct", "avg_neutral_pct", "weekend_activity_ratio",
    "morning_activity_ratio", "afternoon_activity_ratio",
    "evening_activity_ratio", "night_activity_ratio",
    "conversion_rate_30d", "user_conversion_pct", "brand_sentiment_score",
    "negative_impact_pct", "avg_like_ratio",
}


# ======================================================================
# 4. بلاک‌لیست امنیتی: توابع خطرناک پستگرس
# ======================================================================

FORBIDDEN_FUNCTIONS: Set[str] = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "lo_read", "lo_write",
    "dblink", "dblink_connect", "dblink_exec",
    "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_rotate_logfile",
    "set_config", "current_setting",
    "copy_from", "copy_to", "pg_read_server_files",
}

# اسکیمای مجاز برای ارجاع جدول (هر چیز خارج از این‌ها رد می‌شود؛
# مثلاً information_schema یا pg_catalog)
ALLOWED_SCHEMAS: Set[str] = {"public", "analytics", "kpi"}


# ======================================================================
# 5. خود کلاس Validator
# ======================================================================

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SQLGuardrail:
    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

        # --- schema فیزیکی برای qualify() ---
        self.schema = SCHEMA
        self.formatted_schema = {
            db: {table: {col: "TEXT" for col in cols} for table, cols in tables.items()}
            for db, tables in SCHEMA.items()
        }
        self.all_physical_tables: Set[str] = {
            f"{db}.{table}" for db, tables in SCHEMA.items() for table in tables
        }

        # --- ساخت خودکار allowed_join_keys و one_to_many_map از RELATIONSHIPS ---
        self.allowed_join_keys: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        self.one_to_many_map: Set[Tuple[str, str]] = set()

        for parent, parent_col, child, child_col, cardinality in RELATIONSHIPS:
            pair = (parent, child)
            self.allowed_join_keys.setdefault(pair, set()).add((parent_col, child_col))
            if cardinality == "one_to_many":
                self.one_to_many_map.add(pair)

        self.parent_sensitive_columns = PARENT_SENSITIVE_COLUMNS
        self.ratio_columns = RATIO_COLUMNS
        self.forbidden_functions = {f.lower() for f in FORBIDDEN_FUNCTIONS}
        self.allowed_schemas = ALLOWED_SCHEMAS

    # ------------------------------------------------------------------
    # کمک‌تابع‌ها
    # ------------------------------------------------------------------

    def _normalize_table_name(self, table_expr: exp.Table) -> str:
        db = table_expr.db or "public"
        return f"{db.lower()}.{table_expr.name.lower()}"

    def _is_safe_division(self, div_node: exp.Div) -> bool:
        if div_node.right.find(exp.Nullif) or div_node.right.find(exp.Case):
            return True
        curr = div_node.parent
        while curr:
            if isinstance(curr, exp.Case):
                return True
            curr = curr.parent
        return False

    def _resolve_column_table(self, col_expr: exp.Expression, tables_in_scope: Dict[str, str]) -> str:
        if not isinstance(col_expr, exp.Column):
            return ""
        col_table = col_expr.table.lower() if col_expr.table else ""
        col_db = col_expr.db.lower() if col_expr.db else ""

        if col_table in tables_in_scope:
            return tables_in_scope[col_table]
        if col_db and col_table:
            full = f"{col_db}.{col_table}"
            if full in tables_in_scope:
                return tables_in_scope[full]
        if not col_table and len(set(tables_in_scope.values())) == 1:
            return list(tables_in_scope.values())[0]
        return ""

    def _lookup_join_keys(self, t1: str, t2: str) -> Tuple[Set[Tuple[str, str]], bool]:
        """
        کلیدهای مجاز join بین دو جدول را برمی‌گرداند.
        خروجی دوم نشان می‌دهد آیا باید (left,right) یا (right,left) خوانده شود.
        """
        pair = (t1, t2)
        if pair in self.allowed_join_keys:
            return self.allowed_join_keys[pair], False
        reverse = (t2, t1)
        if reverse in self.allowed_join_keys:
            return self.allowed_join_keys[reverse], True
        return set(), False

    # ------------------------------------------------------------------
    # اعتبارسنجی امنیتی سطح‌بالا (DML/DDL/توابع خطرناک/جدول‌های خارج از whitelist)
    # ------------------------------------------------------------------

    _FORBIDDEN_STATEMENT_TYPES = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
         exp.Command, exp.Merge, exp.TruncateTable,
        exp.Grant,
    )

    def _security_scan(self, root: exp.Expression) -> List[str]:
        errors: List[str] = []

        # DML/DDL هر جایی در درخت (از جمله داخل یک CTE نویسنده)
        for node in root.find_all(self._FORBIDDEN_STATEMENT_TYPES):
            errors.append(
                f"SECURITY ERROR: دستور غیرمجاز '{type(node).__name__}' در کوئری شناسایی شد. "
                f"فقط SELECT مجاز است (حتی داخل CTE)."
            )

        # SELECT ... INTO (ساخت جدول جدید)
        for select_node in root.find_all(exp.Select):
            if select_node.args.get("into"):
                errors.append(
                    "SECURITY ERROR: استفاده از 'SELECT ... INTO' برای ساخت جدول جدید مجاز نیست."
                )

        # توابع خطرناک
        for func in root.find_all((exp.Anonymous, exp.Func)):
            fname = str(func.name or getattr(func, "this", "") or "").lower()
            if isinstance(fname, str) and fname in self.forbidden_functions:
                errors.append(f"SECURITY ERROR: فراخوانی تابع غیرمجاز '{fname}' مجاز نیست.")

        return list(dict.fromkeys(errors))

    def _check_table_whitelist(self, root: exp.Expression) -> List[str]:
        errors = []
        for table in root.find_all(exp.Table):
            db = (table.db or "public").lower()
            name = table.name.lower()
            full = f"{db}.{name}"
            if db not in self.allowed_schemas or full not in self.all_physical_tables:
                errors.append(
                    f"UNKNOWN TABLE ERROR: جدول/ویوی '{full}' در schema شناخته‌شده تعریف نشده "
                    f"و مجاز به کوئری‌گیری نیست."
                )
        return list(dict.fromkeys(errors))

    # ------------------------------------------------------------------
    # اعتبارسنجی اصلی
    # ------------------------------------------------------------------

    def validate(self, sql_query: str) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        # ---- ۱. پارس اولیه ----
        try:
            parsed_statements = sqlglot.parse(sql_query, read=self.dialect)
        except Exception as e:
            return ValidationResult(False, [f"Syntax Error: {e}"])

        if not parsed_statements or parsed_statements[0] is None:
            return ValidationResult(False, ["EMPTY QUERY ERROR: کوئری معتبری یافت نشد."])

        if len(parsed_statements) > 1:
            return ValidationResult(False, ["SECURITY ERROR: چند دستور SQL در یک درخواست مجاز نیست."])

        parsed = parsed_statements[0]

        if not isinstance(parsed, exp.Select):
            stmt_type = type(parsed).__name__.upper()
            return ValidationResult(
                False,
                [f"SECURITY ERROR: فقط SELECT مجاز است. دستور دریافتی: '{stmt_type}'."],
            )

        # ---- ۲. اسکن امنیتی روی کل درخت (قبل از qualify، تا چیزی حذف/تغییر نکند) ----
        errors.extend(self._security_scan(parsed))
        errors.extend(self._check_table_whitelist(parsed))
        if errors:
            # اگر مشکل امنیتی پیدا شد نیازی به ادامه‌ی تحلیل ساختاری نیست
            return ValidationResult(False, list(dict.fromkeys(errors)), warnings)

        # ---- ۳. qualify (رفع ابهام نام ستون‌ها/جدول‌ها) ----
        try:
            qualified = qualify(parsed, dialect=self.dialect, schema=self.formatted_schema)
        except Exception as e:
            return ValidationResult(False, [f"Schema/Qualify Error: {e}"])

        # ---- ۴. تقسیم بر صفر ----
        for div in qualified.find_all(exp.Div):
            if not self._is_safe_division(div):
                errors.append(
                    f"DIVIDE BY ZERO RISK: تقسیم '{div.sql()}' بدون محافظت NULLIF/CASE روی مخرج."
                )

        # ---- ۵. SUM روی ستون نسبتی (AVG دیگر بلاک نمی‌شود) ----
        for agg_node in qualified.find_all(exp.Sum):
            for col in agg_node.find_all(exp.Column):
                if col.name.lower() in self.ratio_columns:
                    errors.append(
                        f"INVALID AGGREGATION: نمی‌توان روی ستون نسبتی/درصدیِ از-پیش-محاسبه‌شده‌ی "
                        f"'{col.name}' عملیات SUM انجام داد."
                    )

        # ---- ۶. بررسی هر scope (شامل CTE ها و ساب‌کوئری‌ها) ----
        for scope in traverse_scope(qualified):
            if not isinstance(scope.expression, exp.Select):
                continue

            tables_in_scope: Dict[str, str] = {}
            for table in scope.tables:
                full_name = self._normalize_table_name(table)
                tables_in_scope[table.alias_or_name.lower()] = full_name
                tables_in_scope[table.name.lower()] = full_name
                tables_in_scope[full_name] = full_name

            present_tables = set(tables_in_scope.values())
            scope_joins = scope.expression.args.get("joins", [])

            # کاما-جوین ضمنی ممنوع
            from_tables = [t for t in scope.tables if t.find_ancestor(exp.From)]
            if len(from_tables) > len(scope_joins) + 1:
                errors.append(
                    "SYNTAX POLICY: جوین ضمنی با کاما (FROM t1, t2) مجاز نیست؛ "
                    "همیشه از JOIN صریح با ON استفاده کنید."
                )

            # اعتبارسنجی کلید join
            for join in scope_joins:
                on_clause = join.args.get("on")
                if not on_clause:
                    continue
                for eq in on_clause.find_all(exp.EQ):
                    if not isinstance(eq.left, exp.Column) or not isinstance(eq.right, exp.Column):
                        continue
                    left_col = eq.left.name.lower()
                    right_col = eq.right.name.lower()
                    left_table = self._resolve_column_table(eq.left, tables_in_scope)
                    right_table = self._resolve_column_table(eq.right, tables_in_scope)

                    if not left_table or not right_table or left_table == right_table:
                        continue
                    if left_table not in self.all_physical_tables or right_table not in self.all_physical_tables:
                        continue

                    valid_keys, swapped = self._lookup_join_keys(left_table, right_table)
                    if not valid_keys:
                        # این جفت جدول در نقشه‌ی رابطه‌های شناخته‌شده نیست. اینجا عمداً
                        # فقط warning می‌دهیم نه error: اگر این join واقعاً غلط باشد
                        # (کلید مشترک واقعی ندارند)، خود دیتابیس روی اجرای واقعی خطا
                        # می‌دهد و لایه‌ی retry آن را می‌گیرد. بلاک کردنش در این لایه
                        # فقط ریسک رد کردن یک رابطه‌ی درست ولی پیش‌بینی‌نشده را دارد.
                        warnings.append(
                            f"UNKNOWN JOIN PAIR: جوین بین '{left_table}' و '{right_table}' در نقشه‌ی "
                            f"رابطه‌های شناخته‌شده تعریف نشده. اگر عمداً و درست است، آن را به "
                            f"RELATIONSHIPS اضافه کنید تا این warning دیگر ظاهر نشود؛ اگر اشتباه بود، "
                            f"اجرای واقعی روی دیتابیس آن را مشخص می‌کند."
                        )
                        continue

                    # اینجا برخلاف حالت بالا، رابطه‌ی این دو جدول را می‌شناسیم و کلید صحیحش
                    # را هم می‌دانیم؛ پس اگر LLM کلید اشتباهی زده باشد، برخلاف حالت بالا،
                    # دیتابیس هیچ خطایی نمی‌دهد (چون هر دو ستون از نظر تایپ معتبرند) و فقط
                    # یک عدد کاملاً غلط و بی‌صدا برمی‌گرداند. این حالت باید قطعاً بلاک شود.
                    key_pair = (right_col, left_col) if swapped else (left_col, right_col)
                    if key_pair not in valid_keys:
                        errors.append(
                            f"INVALID JOIN CONDITION: جوین '{left_table}' و '{right_table}' "
                            f"روی '{left_col} = {right_col}' مجاز نیست."
                        )

            # fan-out: SUM/AVG/COUNT روی ستون‌های والد وقتی با child جوین شده
            for parent, child in self.one_to_many_map:
                if parent not in present_tables or child not in present_tables:
                    continue
                sensitive_cols = self.parent_sensitive_columns.get(parent, set())

                for agg_func in scope.expression.find_all((exp.Sum, exp.Avg, exp.Count)):
                    func_name = (
                        "sum" if isinstance(agg_func, exp.Sum)
                        else "avg" if isinstance(agg_func, exp.Avg)
                        else "count"
                    )
                    is_distinct = bool(agg_func.find(exp.Distinct))

                    for col in agg_func.find_all(exp.Column):
                        col_name = col.name.lower()
                        resolved_table = self._resolve_column_table(col, tables_in_scope)
                        if resolved_table != parent:
                            continue

                        if func_name == "count":
                            if not is_distinct:
                                errors.append(
                                    f"CRITICAL FAN-OUT BUG: 'COUNT({col.sql()})' روی جدول والد '{parent}' "
                                    f"بدون DISTINCT، ردیف‌های تکراریِ ناشی از جوین با '{child}' را می‌شمارد. "
                                    f"از 'COUNT(DISTINCT ...)' استفاده کنید."
                                )
                            # COUNT(DISTINCT ...) امن است، حتی روی ستون حساس.
                        elif func_name in ("sum", "avg"):
                            errors.append(
                                f"CRITICAL FAN-OUT BUG: تجمیع '{agg_func.sql()}' روی جدول والد '{parent}' "
                                f"در حالی که با child '{child}' جوین شده، مقدار را به‌اشتباه تکثیر می‌کند."
                            )
                        elif col_name in sensitive_cols and not is_distinct:
                            # حالت نظری: تابع تجمیعی دیگری (غیر از count/sum/avg بالا)
                            # مستقیماً روی ستون حساس بدون DISTINCT اعمال شده.
                            errors.append(
                                f"CRITICAL FAN-OUT BUG: تجمیع '{agg_func.sql()}' روی ستون حساس "
                                f"'{parent}.{col_name}' بدون DISTINCT، هنگام جوین با child '{child}' نادرست است."
                            )

            # چند child همزمان از یک parent -> فقط warning، نه خطا
            children_for_parent: Dict[str, Set[str]] = {}
            for parent, child in self.one_to_many_map:
                if parent in present_tables and child in present_tables:
                    children_for_parent.setdefault(parent, set()).add(child)

            for parent, children in children_for_parent.items():
                if len(children) > 1:
                    warnings.append(
                        f"MULTI-CHILD FAN-OUT WARNING: چند جدول child {sorted(children)} همزمان با "
                        f"parent '{parent}' جوین شده‌اند. اگر تجمیعی در کار است، بهتر است هر child را "
                        f"در یک CTE جدا pre-aggregate کنید."
                    )

        unique_errors = list(dict.fromkeys(errors))
        unique_warnings = list(dict.fromkeys(warnings))
        return ValidationResult(len(unique_errors) == 0, unique_errors, unique_warnings)


# ======================================================================
# 6. تست سریع / نمونه استفاده
# ======================================================================

if __name__ == "__main__":
    guard = SQLGuardrail()

    good_queries = [
        # ساده روی یک ویوی از‌پیش‌تجمیع‌شده
        "SELECT * FROM kpi.product_360 ORDER BY total_revenue DESC LIMIT 10;",
        # جوین یک‌به‌یک feature_product با products (که در نسخه‌ی قبلی رد می‌شد)
        """
        SELECT p.title_fa, fp.total_views, fp.total_purchases
        FROM public.products p
        JOIN analytics.feature_product fp ON p.id = fp.product_id
        ORDER BY fp.total_views DESC LIMIT 20;
        """,
        # COUNT(DISTINCT ستون حساس) باید مجاز باشد
        """
        SELECT p.id, COUNT(DISTINCT c.id) AS comment_cnt
        FROM public.products p
        JOIN public.comments c ON p.id = c.product_id
        GROUP BY p.id;
        """,
        # AVG روی ستون نسبتی مجاز است
        "SELECT AVG(star_rating) FROM kpi.product_360;",
    ]

    bad_queries = [
        # DML قایم‌شده داخل CTE
        """
        WITH d AS (DELETE FROM public.products RETURNING *)
        SELECT * FROM d;
        """,
        # SUM روی ستون قیمت والد بعد از جوین با comments (fan-out)
        """
        SELECT p.id, SUM(p.price) AS total_price
        FROM public.products p
        JOIN public.comments c ON p.id = c.product_id
        GROUP BY p.id;
        """,
        # جوین با کلید اشتباه
        "SELECT * FROM public.products p JOIN public.brands b ON p.id = b.brand_id;",
        # تابع خطرناک
        "SELECT pg_sleep(5);",
    ]

    for q in good_queries:
        result = guard.validate(q)
        print("OK query ->", result.ok, result.errors, result.warnings)

    for q in bad_queries:
        result = guard.validate(q)
        print("BAD query ->", result.ok, result.errors)