import os
import pandas as pd
from config.feature_map import OUTPUT_DIR


class KPIEngine:

    def __init__(self, tables):
        self.tables = tables
        self.kpis = {}

    # =========================
    # 1. PRODUCT KPIs
    # =========================
    def product_kpis(self):
        if "product_master" not in self.tables:
            print("[WARN] product_master missing – skip product_kpis")
            return

        df = self.tables["product_master"].copy()
        
        print(df.columns) 
        kpis = {}


        if "is_purchase" in df.columns:
            kpis["top_products"] = df.sort_values(
                "is_purchase", ascending=False
            ).head(10)

            if "is_cart" in df.columns and "is_view" in df.columns:
                df["view_to_cart"] = df["is_cart"] / (df["is_view"] + 1e-9)
                df["cart_to_purchase"] = df["is_purchase"] / (df["is_cart"] + 1e-9)
                kpis["best_funnel_products"] = df.sort_values(
                    "cart_to_purchase", ascending=False
                ).head(10)

        # Price ممکنه Price یا price باشد
        price_col = next((c for c in ["price", "Price"] if c in df.columns), None)
        if price_col and "is_purchase" in df.columns:
            df["revenue"] = df[price_col] * df["is_purchase"]
            kpis["top_revenue_products"] = df.sort_values(
                "revenue", ascending=False
            ).head(10)

        if "is_remove" in df.columns:
            kpis["most_removed_products"] = df.sort_values(
                "is_remove", ascending=False
            ).head(10)

        if "cart_abandon_rate" in df.columns:
            kpis["worst_cart_products"] = df.sort_values(
                "cart_abandon_rate", ascending=False
            ).head(10)

        if "remove_rate" in df.columns:
            kpis["high_remove_rate_products"] = df.sort_values(
                "remove_rate", ascending=False
            ).head(10)

        self.kpis["product"] = kpis

    # =========================
    # 2. FUNNEL KPIs
    # =========================
    def funnel_kpis(self):
        if "user_behavior_enriched" not in self.tables:
            print("[WARN] user_behavior_enriched missing – skip funnel_kpis")
            return

        df = self.tables["user_behavior_enriched"]
        total_view = (df["event_type"] == "view").sum()
        total_cart = (df["event_type"] == "add_to_cart").sum()
        total_purchase = (df["event_type"] == "purchase").sum()
        total_remove = (df["event_type"] == "remove_from_cart").sum()

        self.kpis["funnel"] = {
            "view_to_cart_rate": total_cart / total_view if total_view else 0,
            "cart_to_purchase_rate": total_purchase / total_cart if total_cart else 0,
            "overall_conversion": total_purchase / total_view if total_view else 0,
            "cart_abandon_rate": (total_cart - total_purchase) / total_cart if total_cart else 0,
            "remove_rate": total_remove / total_cart if total_cart else 0,
            "remove_to_purchase_ratio": total_remove / (total_purchase + 1),
        }

    # =========================
    # 3. USER SEGMENTATION
    # =========================
    def user_segmentation(self):
        if "user_features" not in self.tables:
            print("[WARN] user_features missing – skip user_segmentation")
            return

        df = self.tables["user_features"]
        kpis = {}

        if "is_purchase" in df.columns:
            kpis["heavy_users"] = df[df["is_purchase"] > 10]

        if "is_view" in df.columns:
            kpis["low_engagement_users"] = df[df["is_view"] < 5]

        if "conversion_rate" in df.columns:
            kpis["high_conversion_users"] = df.sort_values(
                "conversion_rate", ascending=False
            ).head(10)
        if "remove_rate" in df.columns:
            kpis["high_remove_users"] = df[df["remove_rate"] > 0.5]

        self.kpis["user_segment"] = kpis

    # =========================
    # 4. RETENTION KPI
    # =========================
    def retention_kpis(self):
        if "user_behavior_enriched" not in self.tables:
            print("[WARN] user_behavior_enriched missing – skip retention_kpis")
            return

        df = self.tables["user_behavior_enriched"]
        user_visits = df.groupby("user_id")["timestamp"].nunique()

        self.kpis["retention"] = {
            "returning_users": int((user_visits > 1).sum()),
            "one_time_users": int((user_visits == 1).sum()),
        }

    # =========================
    # 5. CATEGORY KPI
    # =========================
    def category_kpis(self):
        if "product_master" not in self.tables:
            return

        df = self.tables["product_master"]
        cat_col = next(
            (c for c in ["category", "Category1", "Category2", "sub_category"] if c in df.columns),
            None,
        )
        if cat_col is None or "is_purchase" not in df.columns:
            return

        self.kpis["category"] = {
            "top_categories": (
                df.groupby(cat_col)["is_purchase"]
                .sum()
                .sort_values(ascending=False)
            )
        }

    # =========================
    # 6. TIME KPI
    # =========================
    def time_kpis(self):
        if "user_behavior_enriched" not in self.tables:
            print("[WARN] user_behavior_enriched missing – skip time_kpis")
            return

        df = self.tables["user_behavior_enriched"]
        kpis = {}

        if "hour" in df.columns:
            kpis["hourly_activity"] = df.groupby("hour")["event_type"].count()
            kpis["purchase_by_hour"] = (
                df[df["event_type"] == "purchase"]
                .groupby("hour")["event_type"]
                .count()
            )

        if "day" in df.columns:
            kpis["daily_trend"] = df.groupby("day")["event_type"].count()

        self.kpis["time"] = kpis

    # =========================
    # SAVE KPIs
    # =========================
    def save_kpis(self, output_dir=None):
        out = output_dir or OUTPUT_DIR
        os.makedirs(out, exist_ok=True)
        print(f"\n=== SAVING KPIs to {out} ===")

        for kpi_name, kpi_value in self.kpis.items():
            if isinstance(kpi_value, dict):
                for sub_name, sub_val in kpi_value.items():
                    if isinstance(sub_val, (pd.DataFrame, pd.Series)):
                        path = os.path.join(out, f"kpi_{kpi_name}_{sub_name}.parquet")
                        if isinstance(sub_val, pd.Series):
                            sub_val.to_frame(name=sub_name).to_parquet(path)
                        else:
                            sub_val.to_parquet(path, index=False)
                        print(f"  Saved: {path}")
                    else:
                        path = os.path.join(out, f"kpi_{kpi_name}_{sub_name}.txt")
                        with open(path, "w") as f:
                            f.write(str(sub_val))
                        print(f"  Saved: {path} = {sub_val}")
            elif isinstance(kpi_value, (pd.DataFrame, pd.Series)):
                path = os.path.join(out, f"kpi_{kpi_name}.parquet")
                if isinstance(kpi_value, pd.Series):
                    kpi_value.to_frame(name=kpi_name).to_parquet(path)
                else:
                    kpi_value.to_parquet(path, index=False)
                print(f"  Saved: {path}")

        print("=== KPI SAVE COMPLETE ===\n")

    # =========================
    # RUN ALL
    # =========================
    def run(self, save=True):
        self.product_kpis()
        self.funnel_kpis()
        self.user_segmentation()
        self.retention_kpis()
        self.category_kpis()
        self.time_kpis()

        if save:
            self.save_kpis()

        return self.kpis