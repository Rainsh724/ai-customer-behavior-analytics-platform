from typing import Dict, List, Set, Tuple
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope


class ProductionSQLValidator:
    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

        self.one_to_many_map: Set[Tuple[str, str]] = {
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

        self.allowed_join_keys: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {
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
        }

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

        self.schema = {
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
                "comment_aspects": ["aspect_id", "comment_id", "term", "sentiment", "negative_pct", "neutral_pct", "positive_pct"]
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
                "user_segments": ["user_id", "active_days", "total_views", "total_purchases", "total_spend", "user_segment", "user_conversion_pct"]
            }
        }

        self.all_physical_tables: Set[str] = {
            f"{db}.{table}" for db, tables in self.schema.items() for table in tables
        }

        self.formatted_schema = {}
        for db, tables in self.schema.items():
            self.formatted_schema[db] = {}
            for table, cols in tables.items():
                self.formatted_schema[db][table] = {col: "TEXT" for col in cols}

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

        for agg_node in qualified_parsed.find_all((exp.Sum, exp.Avg)):
            for col in agg_node.find_all(exp.Column):
                if col.name.lower() in self.ratio_columns:
                    func_name = "SUM" if isinstance(agg_node, exp.Sum) else "AVG"
                    errors.append(
                        f"INVALID AGGREGATION BUG: Cannot '{func_name}' pre-calculated ratio/metric '{col.name}'."
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

        unique_errors = list(dict.fromkeys(errors))
        return len(unique_errors) == 0, unique_errors