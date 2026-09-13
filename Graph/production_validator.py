from typing import Dict, List, Set, Tuple
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope


class ProductionSQLValidator:
    """
    اعتبارسنج SQL برای LLM SQL agent.

    schema / one_to_many_map / allowed_join_keys رو می‌شه از بیرون تزریق
    کرد (پیشنهادی: از schema_introspector.introspect_all() که مستقیم از
    information_schema می‌خونه -- ببین from_database() پایین). اگه هیچ‌کدوم
    پاس داده نشن، از _FALLBACK_* (همون مقادیر قدیمیِ دستی) استفاده می‌شه تا
    کد قدیمی که ProductionSQLValidator() رو بدون آرگومان صدا می‌زنه نشکنه.

    parent_sensitive_columns / ratio_columns / id_like_columns از
    information_schema قابل استخراج نیستن (این‌ها دانشِ معنایی درباره‌ی
    این‌که کدوم ستون‌ها fan-out-unsafe یا pre-aggregated هستن‌اند، نه
    ساختار جدول) -- این‌ها همیشه دستی می‌مونن، ولی چون فقط وقتی عوض می‌شن
    که یه *نوع* جدید متریک اضافه بشه (نه هر ستون)، فشار sync خیلی کمتره.
    """

    # ------------------------------------------------------------------
    # Fallback ها -- فقط وقتی از __init__ چیزی پاس داده نشه استفاده می‌شن.
    # مسیر اصلیِ توصیه‌شده اینه که سرویس در startup این‌ها رو زنده از
    # دیتابیس بخونه (schema_introspector.py) و به __init__/from_database
    # بده، نه این‌که این‌ها رو دستی به‌روز نگه دارید.
    # ------------------------------------------------------------------
    _FALLBACK_ONE_TO_MANY: Set[Tuple[str, str]] = {
        ("public.products", "public.comments"),
        ("public.products", "public.user_behavior_logs"),
        ("public.comments", "public.comment_aspects"),
        ("public.comments", "public.comments_embedding"),
        ("public.sessions", "public.user_behavior_logs"),
        ("public.users", "public.sessions"),
        ("public.brands", "public.products"),
        ("public.categories", "public.products"),
        ("public.sellers", "public.products"),
        ("public.cities", "public.sessions"),
        ("analytics.feature_product", "analytics.feature_product_aspect"),
        ("analytics.feature_user", "analytics.feature_user_product"),
        ("analytics.feature_user", "analytics.feature_user_category"),
        ("analytics.feature_category", "analytics.feature_user_category"),
        ("analytics.feature_category", "public.products"),
        ("analytics.feature_brand", "public.products"),
        ("public.products", "analytics.feature_user_product"),
        ("public.products", "analytics.feature_product_aspect"),
    }

    _FALLBACK_ALLOWED_JOIN_KEYS: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {
        ("public.sessions", "public.users"): {("user_id", "user_id")},
        ("public.users", "analytics.feature_user"): {("user_id", "user_id")},
        ("public.products", "public.comments"): {("id", "product_id")},
        ("public.products", "public.user_behavior_logs"): {("id", "product_id")},
        ("public.sessions", "public.user_behavior_logs"): {("session_id", "session_id")},
        ("public.brands", "public.products"): {("brand_id", "brand_id")},
        ("public.categories", "public.products"): {("category_id", "category_id")},
        ("public.sellers", "public.products"): {("seller_id", "seller_id")},
        ("public.cities", "public.sessions"): {("city_id", "city_id")},
        ("public.comments", "public.comment_aspects"): {("id", "comment_id")},
        ("public.comments", "public.comments_embedding"): {("id", "id")},

        # -- Product-level analytics/kpi joins --
        ("public.products", "analytics.feature_product"): {("id", "product_id")},
        ("public.products", "analytics.feature_product_sentiment"): {("id", "product_id")},
        ("public.products", "analytics.feature_product_aspect"): {("id", "product_id")},
        ("public.products", "kpi.product_360"): {("id", "product_id")},
        ("analytics.feature_product", "kpi.product_360"): {("product_id", "product_id")},
        ("analytics.feature_product", "analytics.feature_product_sentiment"): {("product_id", "product_id")},
        ("analytics.feature_product", "analytics.feature_product_aspect"): {("product_id", "product_id")},
        ("public.product_negative_feedback_summary", "public.products"): {("product_id", "id")},
        ("public.product_negative_feedback_summary", "analytics.feature_product"): {("product_id", "product_id")},
        ("public.product_negative_feedback_summary", "kpi.product_360"): {("product_id", "product_id")},

        # -- Brand-level analytics/kpi joins --
        ("public.brands", "analytics.feature_brand"): {("brand_id", "brand_id")},
        ("public.brands", "analytics.feature_brand_sentiment"): {("brand_id", "brand_id")},
        ("public.brands", "kpi.brand_diagnostics"): {("brand_id", "brand_id")},
        ("analytics.feature_brand", "analytics.feature_brand_sentiment"): {("brand_id", "brand_id")},
        ("analytics.feature_brand", "kpi.brand_diagnostics"): {("brand_id", "brand_id")},

        # -- Category-level analytics joins --
        ("public.categories", "analytics.feature_category"): {("category_id", "category_id")},
        ("public.categories", "analytics.feature_category_sentiment"): {("category_id", "category_id")},
        ("analytics.feature_category", "analytics.feature_category_sentiment"): {("category_id", "category_id")},

        # -- City-level analytics joins --
        ("public.cities", "analytics.feature_city"): {("city_id", "city_id")},

        # -- User-level analytics/kpi joins --
        ("public.users", "analytics.feature_user_product"): {("user_id", "user_id")},
        ("public.users", "analytics.feature_user_category"): {("user_id", "user_id")},
        ("public.users", "kpi.user_segments"): {("user_id", "user_id")},
        ("public.users", "kpi.rfm_segments"): {("user_id", "user_id")},
        ("public.users", "kpi.ml_user_clusters"): {("user_id", "user_id")},
        ("analytics.feature_user", "kpi.rfm_segments"): {("user_id", "user_id")},
        ("analytics.feature_user", "kpi.ml_user_clusters"): {("user_id", "user_id")},
        ("kpi.rfm_segments", "kpi.ml_user_clusters"): {("user_id", "user_id")},

        # -- User<->Product bridge tables --
        ("public.products", "analytics.feature_user_product"): {("id", "product_id")},
        ("public.categories", "analytics.feature_user_category"): {("category_id", "category_id")},
    }

    _FALLBACK_SCHEMA: Dict[str, Dict[str, List[str]]] = {
        "public": {
            "cities": ["city_id", "name"],
            "users": ["user_id"],
            "sessions": ["session_id", "user_id", "city_id"],
            "brands": ["brand_id", "name"],
            "categories": ["category_id", "category1", "category2", "sub_category"],
            "sellers": ["seller_id", "seller_title"],
            "products": ["id", "title_fa", "brand_id", "category_id", "seller_id", "price", "min_price_last_month", "is_fake", "rate", "rate_cnt"],
            "user_behavior_logs": ["log_id", "session_id", "product_id", "event_type", "timestamp"],
            "comments": ["id", "product_id", "is_buyer", "rate", "recommendation_status", "likes", "dislikes", "raw_text_normalized", "created_at"],
            "comments_embedding": ["id", "embedded_comment"],
            "comment_aspects": ["aspect_id", "comment_id", "term", "sentiment", "negative_pct", "neutral_pct", "positive_pct"],
            "product_negative_feedback_summary": ["product_id", "avg_negative_pct", "comment_cnt"]
        },
        "analytics": {
            "feature_behavior": ["log_id", "hour", "day", "month", "weekday", "is_weekend", "is_view", "is_cart", "is_remove", "is_purchase"],
            "feature_user": ["user_id", "total_events", "total_sessions", "active_days", "total_views", "total_cart_adds", "total_removes", "total_purchases", "avg_session_events", "max_session_events", "avg_session_duration_minutes", "max_session_duration_minutes", "weekend_activity_ratio", "morning_activity_ratio", "afternoon_activity_ratio", "evening_activity_ratio", "night_activity_ratio", "preferred_hour", "preferred_weekday", "unique_products_viewed", "unique_products_purchased", "cities_visited", "total_spend", "avg_purchase_value", "min_purchase_price", "max_purchase_price", "purchase_frequency", "purchase_days", "brand_diversity", "category_diversity"],
            "feature_product": ["product_id", "total_events", "total_views", "total_cart_adds", "total_removes", "total_purchases", "unique_viewers", "unique_carters", "unique_buyers", "total_sessions", "price_drop_ratio"],
            "feature_city": ["city_id", "total_users", "total_sessions", "total_events", "total_views", "total_cart_adds", "total_purchases", "total_removes", "unique_products_viewed", "unique_products_purchased"],
            "feature_category": ["category_id", "total_events", "total_views", "total_cart_adds", "total_purchases", "total_removes", "unique_viewers", "unique_buyers", "avg_product_price"],
            "feature_brand": ["brand_id", "total_events", "total_views", "total_cart_adds", "total_purchases", "total_removes", "unique_viewers", "unique_buyers"],
            "feature_user_product": ["user_id", "product_id", "total_events", "view_count", "cart_count", "remove_count", "purchase_count", "active_days", "session_count"],
            "feature_user_category": ["user_id", "category_id", "total_events", "view_count", "cart_count", "remove_count", "purchase_count", "category_spend", "view_share", "purchase_share", "spend_share"],
            "feature_product_sentiment": ["product_id", "comment_count", "avg_rate", "avg_like_ratio", "total_likes", "total_dislikes", "total_aspect_mentions", "positive_aspect_mentions", "negative_aspect_mentions", "neutral_aspect_mentions", "avg_positive_pct", "avg_negative_pct", "avg_neutral_pct", "positive_aspect_ratio", "negative_aspect_ratio", "neutral_aspect_ratio"],
            "feature_product_aspect": ["product_id", "term", "total_mentions", "positive_mentions", "negative_mentions", "neutral_mentions", "avg_negative_pct", "avg_neutral_pct", "avg_positive_pct"],
            "feature_brand_sentiment": ["brand_id", "total_comments", "total_aspect_mentions", "positive_aspect_mentions", "negative_aspect_mentions", "neutral_aspect_mentions", "avg_comment_rating", "total_likes", "total_dislikes"],
            "feature_category_sentiment": ["category_id", "total_comments", "total_aspect_mentions", "positive_aspect_mentions", "negative_aspect_mentions", "neutral_aspect_mentions", "avg_comment_rating", "total_likes", "total_dislikes"],
            "feature_aspect": ["term", "total_mentions", "positive_mentions", "negative_mentions", "neutral_mentions", "avg_negative_pct", "avg_neutral_pct", "avg_positive_pct"],
            "feature_time": ["hour", "iso_weekday", "total_events", "total_views", "total_cart_adds", "total_purchases", "total_removes"]
        },
        "kpi": {
            "product_360": ["product_id", "title_fa", "price", "total_views", "total_purchases", "total_revenue", "conversion_rate", "comment_count", "star_rating", "positive_sentiment_pct", "sentiment_score", "managerial_action_tag"],
            "global_funnel": ["total_views", "total_carts", "total_purchases", "total_removes", "view_to_cart_pct", "cart_to_purchase_pct", "overall_conversion_pct", "cart_abandonment_pct"],
            "user_segments": ["user_id", "active_days", "total_views", "total_purchases", "total_spend", "user_segment", "user_conversion_pct"],
            "rfm_segments": ["user_id", "recency_days", "frequency", "monetary", "rfm_code", "rfm_label"],
            "ml_user_clusters": ["user_id", "cluster_id", "cluster_name"],
            "brand_diagnostics": ["brand_id", "brand_name", "total_views", "total_purchases", "total_comments", "avg_rating", "brand_sentiment_score"],
            "aspect_diagnostics": ["aspect_name", "total_mentions", "positive_mentions", "negative_mentions", "negative_impact_pct", "aspect_status"]
        }
    }

    # ------------------------------------------------------------------
    # Join هایی که به‌خاطر معنایی‌بودنشون (نه FK) هیچ‌وقت از
    # information_schema درنمیان -- مثل join روی "term" (نه یه PK/FK واقعی).
    # این لیست همیشه، چه از fallback و چه از introspection استفاده بشه،
    # merge می‌شه. تنها وقتی این‌جا چیزی اضافه می‌کنید که یه *نوع join*
    # کاملاً جدیدِ بدون FK پیدا بشه -- نه با هر ستون جدید.
    # ------------------------------------------------------------------
    MANUAL_EXTRA_JOIN_KEYS: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {
        ("public.comment_aspects", "analytics.feature_aspect"): {("term", "term")},
        ("analytics.feature_product_aspect", "analytics.feature_aspect"): {("term", "term")},
        ("public.comment_aspects", "kpi.aspect_diagnostics"): {("term", "aspect_name")},
    }

    @classmethod
    def from_database(cls, conn, dialect: str = "postgres") -> "ProductionSQLValidator":
        """
        نسخه‌ی توصیه‌شده برای startup: schema/one_to_many_map/allowed_join_keys
        رو زنده از خودِ Postgres (information_schema) می‌خونه، نه از دیکشنری
        دستی. اگه introspection شکست بخوره (مثلاً دیتابیس در دسترس نیست)،
        عمداً exception رو بالا می‌ده -- بهتره سرویس بالا نیاد تا این‌که با
        یه schema خالی/ناقص سکوت‌آمیز اجرا بشه.
        """
        from .schema_introspector import introspect_all

        schema, one_to_many_map, allowed_join_keys = introspect_all(conn)
        return cls(
            dialect=dialect,
            schema=schema,
            one_to_many_map=one_to_many_map,
            allowed_join_keys=allowed_join_keys,
        )

    def __init__(
        self,
        dialect: str = "postgres",
        schema: Dict[str, Dict[str, List[str]]] | None = None,
        one_to_many_map: Set[Tuple[str, str]] | None = None,
        allowed_join_keys: Dict[Tuple[str, str], Set[Tuple[str, str]]] | None = None,
    ):
        self.dialect = dialect

        self.schema = schema if schema is not None else self._FALLBACK_SCHEMA

        self.one_to_many_map: Set[Tuple[str, str]] = set(
            one_to_many_map if one_to_many_map is not None else self._FALLBACK_ONE_TO_MANY
        )

        base_allowed = (
            allowed_join_keys if allowed_join_keys is not None else self._FALLBACK_ALLOWED_JOIN_KEYS
        )
        self.allowed_join_keys: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {
            pair: set(keys) for pair, keys in base_allowed.items()
        }
        # join های بدون FK واقعی همیشه اضافه می‌شن، صرف‌نظر از منبع بالا
        for pair, keys in self.MANUAL_EXTRA_JOIN_KEYS.items():
            self.allowed_join_keys.setdefault(pair, set()).update(keys)

        self.parent_sensitive_columns: Dict[str, Set[str]] = {
            "public.products": {"price", "min_price_last_month", "rate", "rate_cnt"},
            "public.comments": {"likes", "dislikes", "rate"},
            "analytics.feature_product": {"total_events", "total_views", "total_purchases", "unique_viewers", "unique_buyers"},
            "analytics.feature_user": {"total_spend", "avg_purchase_value", "total_events", "total_purchases"},
            "analytics.feature_category": {"total_events", "total_views", "total_purchases", "avg_product_price"},
            "analytics.feature_brand": {"total_events", "total_views", "total_purchases"},
            "analytics.feature_city": {"total_users", "total_sessions", "total_events", "total_views", "total_purchases"},
        }

        self.ratio_columns: Set[str] = {
            "conversion_rate", "view_to_cart_pct", "cart_to_purchase_pct",
            "overall_conversion_pct", "cart_abandonment_pct", "positive_sentiment_pct",
            "sentiment_score", "star_rating", "positive_aspect_ratio",
            "negative_aspect_ratio", "neutral_aspect_ratio", "price_drop_ratio",
            "view_share", "purchase_share", "spend_share", "avg_positive_pct",
            "avg_negative_pct", "avg_neutral_pct", "weekend_activity_ratio",
            "morning_activity_ratio", "afternoon_activity_ratio", "evening_activity_ratio", "night_activity_ratio"
        }

        self.all_physical_tables: Set[str] = {
            f"{db}.{table}" for db, tables in self.schema.items() for table in tables
        }

        # ستون‌های شبه‌کلید (PK/FK) که یکتا یا نزدیک به یکتا هستن -- برای
        # چک «ORDER BY بدون tie-breaker قطعی» استفاده می‌شن.
        self.id_like_columns: Set[str] = {
            "id", "user_id", "product_id", "comment_id", "session_id",
            "city_id", "brand_id", "category_id", "seller_id", "log_id",
            "aspect_id",
        }

        self.formatted_schema = {}
        for db, tables in self.schema.items():
            self.formatted_schema[db] = {}
            for table, cols in tables.items():
                self.formatted_schema[db][table] = {col: "TEXT" for col in cols}

    def _normalize_table_name(self, table_expr: exp.Table) -> str:
        db = table_expr.db or "public"
        return f"{db.lower()}.{table_expr.name.lower()}"

    def _numeric_literal_value(self, expr: exp.Expression):
        """
        اگه expr یه literal عددی باشه (حتی پیچیده‌شده داخل Cast/پرانتز/منفی، مثل
        `-100`، `(100.0)`، `100::numeric`)، مقدارش رو برمی‌گردونه؛ وگرنه None.
        برای تشخیص «مخرج ثابت غیرصفر» در _is_safe_division استفاده می‌شه.
        """
        node = expr
        while isinstance(node, (exp.Cast, exp.Paren)):
            node = node.this
        if isinstance(node, exp.Neg):
            inner = self._numeric_literal_value(node.this)
            return -inner if inner is not None else None
        if isinstance(node, exp.Literal) and node.is_number:
            try:
                return float(node.this)
            except (TypeError, ValueError):
                return None
        return None

    def _is_safe_division(self, div_node: exp.Div) -> bool:
        # تقسیم بر یه عدد ثابتِ غیرصفر (مثل `/ 100.0` یا `/ 2`) هیچ‌وقت به
        # صفر تقسیم نمی‌شه -- این چک فقط باید نگران مخرج‌های *متغیر*
        # (ستون یا نتیجه‌ی یه محاسبه/تجمیع) باشه که ممکنه در runtime صفر
        # بشن. قبلاً این تشخیص وجود نداشت و هر تقسیم بر یه literal رو هم
        # رد می‌کرد.
        literal_denominator = self._numeric_literal_value(div_node.right)
        if literal_denominator is not None:
            return literal_denominator != 0

        if div_node.right.find(exp.Nullif) or div_node.right.find(exp.Case):
            return True
        curr = div_node.parent
        while curr:
            if isinstance(curr, exp.Case):
                return True
            curr = curr.parent
        return False

    def _resolve_column_table(self, col_expr: exp.Expression, tables_in_scope: Dict[str, str]) -> str:
        # محافظت در برابر مقادیر ثابت (Literal) یا عبارات غیرستونی در شرط ON
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

    def validate(self, sql_query: str) -> Tuple[bool, List[str]]:
        errors = []
        try:
            parsed_statements = sqlglot.parse(sql_query, read=self.dialect)
            if not parsed_statements:
                return False, ["EMPTY QUERY ERROR: No valid SQL expression found."]

            if len(parsed_statements) > 1:
                return False, ["SECURITY ERROR: Multi-statement SQL queries are strictly forbidden."]

            parsed = parsed_statements[0]

            if not isinstance(parsed, exp.Select):
                stmt_type = type(parsed).__name__.upper()
                return False, [f"SECURITY ERROR: Only SELECT queries are permitted in analytics engine. Received forbidden statement type: '{stmt_type}'."]

            qualified_parsed = qualify(parsed, dialect=self.dialect, schema=self.formatted_schema)
        except Exception as e:
            return False, [f"Syntax/Schema Error in SQL Query: {str(e)}"]

        for div in qualified_parsed.find_all(exp.Div):
            if not self._is_safe_division(div):
                errors.append(
                    f"DIVIDE BY ZERO RISK: Division '{div.sql()}' missing NULLIF or CASE protection on denominator."
                )

        # ------------------------------------------------------------
        # ROUND(double precision, integer) در Postgres اصلاً تعریف نشده
        # (فقط ROUND(numeric, integer) یا ROUND(double precision) تک‌آرگومانی
        # وجود داره) -- این یه خطای واقعیِ Postgres‌ه که هیچ ربطی به
        # امنیت/کاردینالیتی نداره، ولی چون این کدبیس همه‌جا از
        # `::double precision` برای تقسیم استفاده می‌کنه، بدون این چک هر بار
        # فقط بعد از یه round-trip واقعی به دیتابیس کشف می‌شه. این‌جا زودتر
        # می‌گیریمش تا Agent همون لحظه بفهمه باید ::numeric کست کنه.
        # ------------------------------------------------------------
        for round_call in qualified_parsed.find_all(exp.Round):
            decimals = round_call.args.get("decimals")
            first_arg = round_call.this
            if decimals is None or first_arg is None:
                continue  # ROUND(x) تک‌آرگومانی روی double precision مشکلی نداره
            has_double_cast = any(
                "DOUBLE" in str(cast_node.to).upper()
                for cast_node in first_arg.find_all(exp.Cast)
            )
            if has_double_cast:
                errors.append(
                    "POSTGRES TYPE ERROR: ROUND(double precision, integer) does not exist in "
                    "Postgres -- only ROUND(numeric, integer) is defined. Cast the first "
                    f"argument to ::numeric instead. Offending expression: {round_call.sql()[:200]}"
                )

        for scope in traverse_scope(qualified_parsed):
            if not isinstance(scope.expression, exp.Select):
                continue

            tables_in_scope: Dict[str, str] = {}
            for table in scope.tables:
                full_name = self._normalize_table_name(table)
                alias = table.alias_or_name.lower()
                name = table.name.lower()

                tables_in_scope[alias] = full_name
                tables_in_scope[name] = full_name
                tables_in_scope[full_name] = full_name

            present_tables = set(tables_in_scope.values())
            scope_joins = scope.expression.args.get("joins", [])

            # ------------------------------------------------------------
            # AVG/SUM مستقیم روی یه ستونِ ratio-نام (مثل conversion_rate)
            # فقط وقتی واقعاً خطرناکه که این SELECT خودش JOIN داشته باشه --
            # چون اون‌جاست که ردیف parent می‌تونه به‌خاطر fan-out تکرار بشه
            # و averaging غلط از آب دربیاد. روی یه جدول تنها (بدون JOIN)،
            # هیچ ردیفی تکرار نمی‌شه؛ AVG(conversion_rate) این‌جا فقط یعنی
            # «میانگین نرخ تبدیل بین محصولات» -- یه متریک کاملاً معتبر و
            # رایج، نه یه باگ. قبلاً این چک بدون توجه به join سرتاسری اجرا
            # می‌شد و کوئری‌های تشخیصیِ ساده‌ی تک‌جدولی رو هم رد می‌کرد.
            # ------------------------------------------------------------
            if scope_joins:
                for agg_node in scope.expression.find_all((exp.Sum, exp.Avg)):
                    arg = agg_node.this
                    if not (isinstance(arg, exp.Column) and arg.name.lower() in self.ratio_columns):
                        continue
                    func_name = "SUM" if isinstance(agg_node, exp.Sum) else "AVG"
                    errors.append(
                        f"INVALID AGGREGATION BUG: Cannot '{func_name}' pre-calculated ratio/metric "
                        f"'{arg.name}' directly after a JOIN -- this averages per-row ratios across "
                        f"joined/duplicated rows without weighting by sample size. Use a weighted "
                        f"average instead: SUM({arg.name} * weight_column) / NULLIF(SUM(weight_column), 0)."
                    )

            from_tables = [t for t in scope.tables if t.find_ancestor(exp.From)]
            if len(from_tables) > len(scope_joins) + 1:
                errors.append(
                    "SYNTAX POLICY: Implicit comma joins (e.g. FROM table1, table2) are strictly forbidden. Always use explicit JOIN with ON clause."
                )

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

                    if left_table and right_table and left_table != right_table:
                        pair = (left_table, right_table)
                        reverse_pair = (right_table, left_table)

                        if left_table in self.all_physical_tables and right_table in self.all_physical_tables:
                            if pair in self.allowed_join_keys:
                                valid_keys = self.allowed_join_keys[pair]
                                if (left_col, right_col) not in valid_keys:
                                    errors.append(
                                        f"INVALID JOIN CONDITION: Cannot join '{left_table}' and '{right_table}' on '{left_col} = {right_col}'."
                                    )
                            elif reverse_pair in self.allowed_join_keys:
                                valid_keys = self.allowed_join_keys[reverse_pair]
                                if (right_col, left_col) not in valid_keys:
                                    errors.append(
                                        f"INVALID JOIN CONDITION: Cannot join '{right_table}' and '{left_table}' on '{right_col} = {left_col}'."
                                    )
                            else:
                                errors.append(
                                    f"UNSUPPORTED JOIN PAIR: Direct join between '{left_table}' and '{right_table}' is not permitted by architecture."
                                )

            for parent, child in self.one_to_many_map:
                if parent in present_tables and child in present_tables:
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

                            if resolved_table == parent:
                                if func_name == "count" and not is_distinct:
                                    errors.append(
                                        f"CRITICAL FAN-OUT BUG: 'COUNT({col.sql()})' on parent table '{parent}' "
                                        f"without DISTINCT will count duplicate child rows. Use 'COUNT(DISTINCT ...)'."
                                    )
                                elif func_name in ("sum", "avg") or col_name in sensitive_cols:
                                    errors.append(
                                        f"CRITICAL FAN-OUT BUG: Aggregation '{agg_func.sql()}' on parent table "
                                        f"'{parent}' while joined with child '{child}'."
                                    )

            children_for_parent = {}
            for parent, child in self.one_to_many_map:
                if parent in present_tables and child in present_tables:
                    children_for_parent.setdefault(parent, set()).add(child)

            for parent, children in children_for_parent.items():
                if len(children) > 1:
                    errors.append(
                        f"MULTI-CHILD FAN-OUT WARNING: Multiple child tables {children} joined with parent '{parent}'. "
                        f"Pre-aggregate child tables separately in CTEs before joining."
                    )

        # چک «ORDER BY بدون tie-breaker قطعی» برای پایداری بیشتر و جلوگیری از
        # ریجکت کوئری‌های تحلیلی پیچیده (CTEها، self-joinها و رتبه‌بندی‌ها) برداشته شد.
        unique_errors = list(dict.fromkeys(errors))
        return len(unique_errors) == 0, unique_errors