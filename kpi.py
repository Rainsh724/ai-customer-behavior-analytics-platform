





class KPIEngine:

    def __init__(self, tables):
        self.tables = tables
        self.kpis = {}

    # =========================
    # 1. PRODUCT KPIs (Advanced)
    # =========================
    def product_kpis(self):

        df = self.tables["product_master"].copy()

        kpis = {}

        # Top selling
        kpis["top_products"] = df.sort_values(
            "is_purchase", ascending=False
        ).head(10)

        # Funnel per product
        df["view_to_cart"] = df["is_cart"] / df["is_view"]
        df["cart_to_purchase"] = df["is_purchase"] / df["is_cart"]

        kpis["best_funnel_products"] = df.sort_values(
            "cart_to_purchase", ascending=False
        ).head(10)

        # Revenue (اگر price داشته باشی)
        if "price" in df.columns:
            df["revenue"] = df["price"] * df["is_purchase"]
            kpis["top_revenue_products"] = df.sort_values(
                "revenue", ascending=False
            ).head(10)

        self.kpis["product"] = kpis

    # =========================
    # 2. FUNNEL KPIs (GLOBAL)
    # =========================
    def funnel_kpis(self):

        df = self.tables["user_behavior_enriched"].copy()
        


        total_view = (df["event_type"] == "view").sum()
        total_cart = (df["event_type"] == "add_to_cart").sum()
        total_purchase = (df["event_type"] == "purchase").sum()

        kpis = {}

        kpis["view_to_cart_rate"] = total_cart / total_view
        kpis["cart_to_purchase_rate"] = total_purchase / total_cart
        kpis["overall_conversion"] = total_purchase / total_view

        self.kpis["funnel"] = kpis

    # =========================
    # 3. USER SEGMENTATION
    # =========================
    def user_segmentation(self):

        df = self.tables["user_features"].copy()

        kpis = {}

        # Heavy users
        kpis["heavy_users"] = df[df["is_purchase"] > 10]

        # Low engagement
        kpis["low_engagement_users"] = df[df["is_view"] < 5]

        # High conversion users
        kpis["high_conversion_users"] = df.sort_values(
            "conversion_rate", ascending=False
        ).head(10)

        self.kpis["user_segment"] = kpis

    # =========================
    # 4. RETENTION KPI
    # =========================
    def retention_kpis(self):

        df = self.tables["user_behavior_enriched"].copy()

        kpis = {}

        user_visits = df.groupby("user_id")["timestamp"].nunique()

        kpis["returning_users"] = (user_visits > 1).sum()
        kpis["one_time_users"] = (user_visits == 1).sum()

        self.kpis["retention"] = kpis

    # =========================
    # 5. CATEGORY KPI
    # =========================
    def category_kpis(self):

        df = self.tables["product_master"].copy()

        if "category" not in df.columns:
            return

        kpis = {}

        kpis["top_categories"] = df.groupby("category")[
            "is_purchase"
        ].sum().sort_values(ascending=False)

        self.kpis["category"] = kpis

    # =========================
    # 6. TIME KPI (Improved)
    # =========================
    def time_kpis(self):

        df = self.tables["user_behavior_enriched"].copy()

        kpis = {}

        kpis["hourly_activity"] = df.groupby("hour")[
            "event_type"
        ].count()

        kpis["purchase_by_hour"] = df[df["event_type"] == "purchase"] \
            .groupby("hour")["event_type"].count()

        kpis["daily_trend"] = df.groupby("day")[
            "event_type"
        ].count()

        self.kpis["time"] = kpis

    # =========================
    # RUN ALL
    # =========================
    def run(self):

        self.product_kpis()
        self.funnel_kpis()
        self.user_segmentation()
        self.retention_kpis()
        self.category_kpis()
        self.time_kpis()

        return self.kpis

  