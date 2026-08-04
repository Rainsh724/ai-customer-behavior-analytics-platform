import os
import glob
import gc
import pandas as pd
import numpy as np

from config.feature_map import FEATURE_MAP, OUTPUT_DIR, get_cleaned_folder_path


class FeatureEngineer:

    def __init__(self, feature_map=None):
        self.feature_map = feature_map or FEATURE_MAP
        self.tables = {}

    # =========================
    # HELPERS
    # =========================
    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        for col in df.select_dtypes(include=["object"]).columns:
            nunique = df[col].nunique(dropna=False)
            if nunique < len(df) * 0.5 and nunique < 10000:
                df[col] = df[col].astype("category")
        return df

    def _load_parquet_folder(self, folder_path: str, downcast: bool = True) -> pd.DataFrame:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {folder_path}")

        print(f"  Loading {len(parquet_files)} parquet files from {folder_path} ...")

        df = None
        for i, f in enumerate(parquet_files, 1):
            part = pd.read_parquet(f)
            if df is None:
                df = part
            else:
                df = pd.concat([df, part], ignore_index=True)
                del part
            if i % 20 == 0 or i == len(parquet_files):
                print(f"    ... {i}/{len(parquet_files)} files, current shape: {df.shape}")
                gc.collect()

        if downcast:
            df = self._downcast(df)
        print(f"  → final shape: {df.shape}")
        return df

    def _aggregate_comments_from_folder(self, folder_path: str) -> pd.DataFrame:
        """کامنت‌ها را فایل‌به‌فایل aggregate می‌کند — کل جدول را در RAM نگه نمی‌دارد."""
        parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {folder_path}")

        print(f"  Aggregating comments from {len(parquet_files)} files (chunked) ...")

        sum_rate = {}
        count_rate = {}
        sum_like = {}
        count_like = {}
        sum_sent = {}
        count_sent = {}

        for i, f in enumerate(parquet_files, 1):
            part = pd.read_parquet(f)

            if "product_id" not in part.columns:
                del part
                continue

            pids = part["product_id"].values

            if "rate" in part.columns:
                rates = pd.to_numeric(part["rate"], errors="coerce")
                for pid, r in zip(pids, rates):
                    if pd.isna(r):
                        continue
                    sum_rate[pid] = sum_rate.get(pid, 0.0) + float(r)
                    count_rate[pid] = count_rate.get(pid, 0) + 1

            if "likes" in part.columns and "dislikes" in part.columns:
                likes = pd.to_numeric(part["likes"], errors="coerce").fillna(0)
                dislikes = pd.to_numeric(part["dislikes"], errors="coerce").fillna(0)
                ratios = likes / (dislikes + 1)
                for pid, lr in zip(pids, ratios):
                    if pd.isna(lr):
                        continue
                    sum_like[pid] = sum_like.get(pid, 0.0) + float(lr)
                    count_like[pid] = count_like.get(pid, 0) + 1

            if "rate" in part.columns:
                sents = pd.to_numeric(part["rate"], errors="coerce")
                for pid, s in zip(pids, sents):
                    if pd.isna(s):
                        continue
                    sum_sent[pid] = sum_sent.get(pid, 0.0) + float(s)
                    count_sent[pid] = count_sent.get(pid, 0) + 1

            del part
            if i % 20 == 0 or i == len(parquet_files):
                print(f"    ... {i}/{len(parquet_files)}")
                gc.collect()

        all_pids = set(sum_rate) | set(sum_like) | set(sum_sent)
        rows = []
        for pid in all_pids:
            row = {"product_id": pid}
            if pid in count_rate and count_rate[pid] > 0:
                row["avg_rate"] = sum_rate[pid] / count_rate[pid]
                row["comment_count"] = count_rate[pid]
            if pid in count_like and count_like[pid] > 0:
                row["avg_like_ratio"] = sum_like[pid] / count_like[pid]
            if pid in count_sent and count_sent[pid] > 0:
                row["avg_sentiment"] = sum_sent[pid] / count_sent[pid]
            rows.append(row)

        out = pd.DataFrame(rows)
        print(f"  → comment_features shape: {out.shape}")
        return out

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self):
        print("=== LOADING PARQUET DATA ===")

        # 1) products (کوچک‌تر)
        name = "digikala-products"
        if name in self.feature_map:
            folder = get_cleaned_folder_path(name)
            alias = self.feature_map[name].get("table_alias", name)
            print(f"\n[{alias}]")
            self.tables[alias] = self._load_parquet_folder(folder, downcast=True)
            gc.collect()

        # 2) user behavior
        name = "user_behavior_logs"
        if name in self.feature_map:
            folder = get_cleaned_folder_path(name)
            alias = self.feature_map[name].get("table_alias", name)
            print(f"\n[{alias}]")
            self.tables[alias] = self._load_parquet_folder(folder, downcast=True)
            gc.collect()

        # 3) comments — فقط aggregate
        name = "digikala-comments"
        if name in self.feature_map:
            folder = get_cleaned_folder_path(name)
            print(f"\n[comment_features]")
            self.tables["comment_features"] = self._aggregate_comments_from_folder(folder)
            gc.collect()

        print("\n=== LOAD COMPLETE ===")
        for k, v in self.tables.items():
            print(f"  {k}: {v.shape}")
        print()

    # =========================
    # DATETIME FEATURES
    # =========================
    def create_time_features(self, df, col):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df["hour"] = df[col].dt.hour.astype("int8")
        df["day"] = df[col].dt.day.astype("int8")
        df["month"] = df[col].dt.month.astype("int8")
        df["weekday"] = df[col].dt.weekday.astype("int8")
        df["is_weekend"] = df["weekday"].isin([4, 5]).astype("int8")
        return df

    # =========================
    # USER BEHAVIOR FEATURES
    # =========================
    def user_behavior_features(self):
        if "user_behavior" not in self.tables:
            print("[WARN] user_behavior missing – skip")
            return

        print("=== USER BEHAVIOR FEATURES ===")
        df = self.tables["user_behavior"]
        df = self.create_time_features(df, "timestamp")

        df["is_view"] = (df["event_type"] == "view").astype("int8")
        df["is_cart"] = (df["event_type"] == "add_to_cart").astype("int8")
        df["is_purchase"] = (df["event_type"] == "purchase").astype("int8")
        df["is_remove"] = (df["event_type"] == "remove_from_cart").astype("int8")

        user_agg = df.groupby("user_id", observed=True).agg({
            "is_view": "sum",
            "is_cart": "sum",
            "is_remove": "sum",
            "is_purchase": "sum",
        }).reset_index()
        user_agg["conversion_rate"] = user_agg["is_purchase"] / (user_agg["is_view"] + 1)
        user_agg["remove_rate"] = user_agg["is_remove"] / (user_agg["is_cart"] + 1)

        product_agg = df.groupby("product_id", observed=True).agg({
            "is_view": "sum",
            "is_cart": "sum",
             "is_remove": "sum",
            "is_purchase": "sum",
        }).reset_index()
        product_agg["conversion_rate"] = product_agg["is_purchase"] / (product_agg["is_view"] + 1)
        product_agg["drop_off_rate"] =(
        (product_agg["is_view"] - product_agg["is_purchase"]) / (product_agg["is_view"] + 1)
)
        product_agg["remove_rate"] = product_agg["is_remove"] / (product_agg["is_cart"] + 1)

        product_agg["cart_abandon_rate"] = (
        (product_agg["is_cart"] - product_agg["is_purchase"]) / (product_agg["is_cart"] + 1)
)

        product_agg["remove_to_cart_ratio"] = (
        product_agg["is_remove"] / (product_agg["is_cart"] + 1)
)

        city_agg = df.groupby("city", observed=True).agg({
            "is_view": "sum",
            "is_purchase": "sum",
        }).reset_index()

        self.tables["user_features"] = user_agg
        self.tables["product_behavior"] = product_agg
        self.tables["city_features"] = city_agg
        self.tables["user_behavior_enriched"] = df

        print(f"  user_features: {user_agg.shape}")
        print(f"  product_behavior: {product_agg.shape}")
        print(f"  city_features: {city_agg.shape}")
        print(f"  user_behavior_enriched: {df.shape}")
        gc.collect()
    # =========================================
    # ADVANCED USER FEATURES (RFM, Sessions & Profile)
    # =========================================
    def advanced_user_features(self):
        if "user_behavior_enriched" not in self.tables or "products" not in self.tables:
            print("[WARN] Required tables missing for advanced_user_features - skip")
            return
            
        print("=== BUILDING ADVANCED USER FEATURES ===")
        df_bh = self.tables["user_behavior_enriched"]
        df_pr = self.tables["products"]
        
        # ۱. ویژگی‌های سطح نشست (Session Dynamics)
        # محاسبه عمق هر نشست و سپس میانگین آن برای هر کاربر
        session_agg = df_bh.groupby('session_id', observed=True).agg(
            session_depth=('event_type', 'count'),
            user_id=('user_id', 'first')
        ).reset_index()
        
        user_session = session_agg.groupby('user_id').agg(
            total_sessions=('session_id', 'nunique'),
            avg_session_depth=('session_depth', 'mean')
        ).reset_index()
        
        # ۲. ویژگی‌های ترجیحاتی (Time Preference)
        # محاسبه نسبت فعالیت در روزهای آخر هفته
        time_pref = df_bh.groupby('user_id', observed=True).agg(
            weekend_activity_ratio=('is_weekend', 'mean')
        ).reset_index()
        
        # ۳. ویژگی‌های ارزش مشتری (RFM & Diversity)
        # ادغام لاگ خریدها با مشخصات کالاها برای استخراج قیمت و برند
        df_purchases = df_bh[df_bh['is_purchase'] == 1].merge(
            df_pr[['id', 'Price', 'Brand', 'Category1']], 
            left_on='product_id', right_on='id', how='inner'
        )
        
        if not df_purchases.empty:
            rfm_agg = df_purchases.groupby('user_id').agg(
                total_spend=('Price', 'sum'),            # مجموع ارزش خرید
                aov=('Price', 'mean'),                   # میانگین ارزش سبد خرید
                brand_diversity=('Brand', 'nunique'),    # تنوع برند
                category_diversity=('Category1', 'nunique') # تنوع دسته‌بندی
            ).reset_index()
        else:
            rfm_agg = pd.DataFrame(columns=['user_id', 'total_spend', 'aov', 'brand_diversity', 'category_diversity'])
            
        # ۴. چسباندن فیچرهای جدید به جدول اصلی user_features
        user_features = self.tables["user_features"]
        user_features = user_features.merge(user_session, on='user_id', how='left')
        user_features = user_features.merge(time_pref, on='user_id', how='left')
        user_features = user_features.merge(rfm_agg, on='user_id', how='left')
        
        # پر کردن مقادیر خالی (کاربرانی که خریدی نداشته‌اند) با صفر
        cols_to_fillna = ['total_spend', 'aov', 'brand_diversity', 'category_diversity']
        user_features[cols_to_fillna] = user_features[cols_to_fillna].fillna(0)
        
        self.tables["user_features"] = user_features
        print(f"  advanced user_features shape: {user_features.shape}")
        import gc
        gc.collect()
    # =========================
    # COMMENT FEATURES
    # =========================
    def comment_features(self):
        if "comment_features" in self.tables:
            print("[INFO] comment_features already built during load – skip")
            return
        print("[WARN] comment_features missing")

    # =========================
    # PRODUCT MASTER
    # =========================
    def build_product_master(self):
        if "products" not in self.tables:
            print("[WARN] products missing – skip build_product_master")
            return

        print("=== BUILD PRODUCT MASTER ===")
        products = self.tables["products"]

        if "product_behavior" in self.tables:
            products = products.merge(
                self.tables["product_behavior"],
                left_on="id",
                right_on="product_id",
                how="left",
            )

        if "comment_features" in self.tables:
            products = products.merge(
                self.tables["comment_features"],
                left_on="id",
                right_on="product_id",
                how="left",
                suffixes=("", "_comment"),
            )
            
        # --- کدهای جدید اضافه شده برای ویژگی‌های پیشرفته محصول ---
        
        # ۱. نرخ درگیری نظرات (Review Engagement Rate)
        if 'comment_count' in products.columns and 'is_view' in products.columns:
            products['review_engagement_rate'] = products['comment_count'] / (products['is_view'] + 1e-5)
            products['review_engagement_rate'] = products['review_engagement_rate'].fillna(0)

        # ۲. حساسیت قیمتی و افت قیمت (Price Drop Ratio)
        price_col = 'Price' if 'Price' in products.columns else 'price'
        if price_col in products.columns and 'min_price_last_month' in products.columns:
            # فرمول: چقدر قیمت فعلی نسبت به کمترین قیمت ماه گذشته افت داشته است
            products['price_drop_ratio'] = (products['min_price_last_month'] - products[price_col]) / (products[price_col] + 1e-5)
            # مقادیر منفی (افزایش قیمت) را روی صفر تنظیم می‌کنیم تا فقط افت قیمت‌ها بماند
            products['price_drop_ratio'] = products['price_drop_ratio'].clip(lower=0)

        # --------------------------------------------------

        self.tables["product_master"] = products
        print(f"  product_master: {products.shape}")
        import gc
        gc.collect()

    # =========================
    # SAVE
    # =========================
    def save_tables(self, output_dir=None):
        out = output_dir or OUTPUT_DIR
        os.makedirs(out, exist_ok=True)

        tables_to_save = [
            "user_features",
            "product_behavior",
            "city_features",
            "user_behavior_enriched",
            "comment_features",
            "product_master",
        ]

        print(f"\n=== SAVING TABLES to {out} ===")
        for name in tables_to_save:
            if name not in self.tables:
                continue
            path = os.path.join(out, f"{name}.parquet")
            self.tables[name].to_parquet(path, index=False)
            print(f"  Saved: {path}  shape={self.tables[name].shape}")
        print("=== SAVE COMPLETE ===\n")

    # =========================
    # RUN
    # =========================
    def run(self, save=True):
        self.load_data()
        self.user_behavior_features()
        self.advanced_user_features()
        self.comment_features()
        self.build_product_master()
        if save:
            self.save_tables()
        return self.tables