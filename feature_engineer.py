

import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.tables = {}

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self):
        for name, cfg in self.feature_map.items():
            df = pd.read_csv(cfg["file_name"], low_memory=False)
            self.tables[name] = df

    # =========================
    # DATETIME FEATURES
    # =========================
    def create_time_features(self, df, col):

        df[col] = pd.to_datetime(
        df[col],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

        df["hour"] = df[col].dt.hour
        df["day"] = df[col].dt.day
        df["month"] = df[col].dt.month
        df["weekday"] = df[col].dt.weekday
        df["is_weekend"] = df["weekday"].isin([4, 5]).astype(int)

        return df

    # =========================
    # USER BEHAVIOR FEATURES
    # =========================
    def user_behavior_features(self):

        df = self.tables["user_behavior"].copy()

        # time features
        df = self.create_time_features(df, "timestamp")

        # event flags
        df["is_view"] = (df["event_type"] == "view").astype(int)
        df["is_cart"] = (df["event_type"] == "add_to_cart").astype(int)
        df["is_purchase"] = (df["event_type"] == "purchase").astype(int)

        # =====================
        # USER LEVEL
        # =====================
        user_agg = df.groupby("user_id").agg({
            "is_view": "sum",
            "is_cart": "sum",
            "is_purchase": "sum"
        }).reset_index()

        user_agg["conversion_rate"] = user_agg["is_purchase"] / (user_agg["is_view"] + 1)

        # =====================
        # PRODUCT LEVEL
        # =====================
        product_agg = df.groupby("product_id").agg({
            "is_view": "sum",
            "is_cart": "sum",
            "is_purchase": "sum"
        }).reset_index()

        product_agg["conversion_rate"] = product_agg["is_purchase"] / (product_agg["is_view"] + 1)
        product_agg["drop_off_rate"] = 1 - product_agg["conversion_rate"]

        # =====================
        # CITY LEVEL
        # =====================
        city_agg = df.groupby("city").agg({
            "is_view": "sum",
            "is_purchase": "sum"
        }).reset_index()

        self.tables["user_features"] = user_agg
        self.tables["product_behavior"] = product_agg
        self.tables["city_features"] = city_agg
        self.tables["user_behavior_enriched"] = df

    # =========================
    # COMMENT FEATURES
    # =========================
    def comment_features(self):

        df = self.tables["comments"].copy()

        # time features
        df = self.create_time_features(df, "created_at")

        # like ratio
        df["like_ratio"] = df["likes"] / (df["dislikes"] + 1)

        # simple sentiment proxy
        df["sentiment_score"] = df["rate"]

        # =====================
        # PRODUCT LEVEL
        # =====================
        product_comment = df.groupby("product_id").agg({
            "rate": ["mean", "count"],
            "like_ratio": "mean",
            "sentiment_score": "mean"
        })

        product_comment.columns = [
            "avg_rate",
            "comment_count",
            "avg_like_ratio",
            "avg_sentiment"
        ]

        product_comment = product_comment.reset_index()

        self.tables["comment_features"] = product_comment
        self.tables["comments_enriched"] = df

    # =========================
    # PRODUCT FINAL TABLE
    # =========================
    def build_product_master(self):

        products = self.tables["products"].copy()

        # join behavior
        if "product_behavior" in self.tables:
            products = products.merge(
                self.tables["product_behavior"],
                left_on="id",
                right_on="product_id",
                how="left"
            )

        # join comments
        if "comment_features" in self.tables:
            products = products.merge(
                self.tables["comment_features"],
                left_on="id",
                right_on="product_id",
                how="left"
            )

        self.tables["product_master"] = products

    # =========================
    # RUN ALL
    # =========================
    def run(self):

        self.load_data()

        # apply datetime features dynamically
    

        # core features
        self.user_behavior_features()
        self.comment_features()
        self.build_product_master()

        return self.tables



