

import pandas as pd
from feature_engine import FeatureEngine

FEATURE_MAP = {

    # ==========================
    # Comment
    # ==========================

    "comment": {

        "file_name": "comment.csv",

        "primary_key": None,

        "foreign_keys": [
            "product_id"
        ],

        "join_columns": [
            "product_id"
        ],

        "text_columns": [
            "comment"
        ],

        "preprocess_columns": [
            "comment"
        ],

        "embedding_columns": [
            "comment"
        ],

        "rag_columns": [
            "comment"
        ],

        "sentiment_columns": [
            "comment"
        ],

        "datetime_columns": [
            "confirmed_at"
        ],

        "numeric_columns": [],

        "categorical_columns": [],

        "raw_text_columns": [
            "comment"
        ]
    },

    # ==========================
    # Reviews (Quality)
    # ==========================

   

    # ==========================
    # Orders
    # ==========================

    "orders": {

        "file_name": "orders.csv",

        "primary_key": "ID_Order",

        "foreign_keys": [
            "ID_Customer",
            "ID_Item"
        ],

        "join_columns": [
            "ID_Customer"
            
        ],

        "text_columns": [
            "city_name_fa"
        ],

        "preprocess_columns": [],

        "embedding_columns": [],

        "rag_columns": [],

        "sentiment_columns": [],

        "datetime_columns": [
            "DateTime_CartFinalize"
        ],

        "numeric_columns": [
            "Amount_Gross_Order",
            "Quantity_item"
        ],

        "categorical_columns": [],

        "raw_text_columns": [
            "comment"
        ]
    },

    # ==========================
    # Purchase History
    # ==========================

    "purchase_history": {

        "file_name": "tarikhche kharid.csv",

        "primary_key": "id",

        "foreign_keys": [
            "product_variant_id",
            "product_id",
            "marketplace_seller_id"
        ],

        "join_columns": [
            "product_variant_id",
            "product_id",
            "marketplace_seller_id"
        ],

        "text_columns": [
            "tags"
        ],

        "preprocess_columns": [],

        "embedding_columns": [],

        "rag_columns": [],

        "sentiment_columns": [],

        "datetime_columns": [
            "created_at",
            "start_at",
            "end_at"
        ],

        "numeric_columns": [
            "selling_price",
            "rrp_price",
            "base_price",
            "buy_price",
            "order_limit"
        ],

        "categorical_columns": [
            "active",
            "show_in_price_history"
        ],

        "raw_text_columns": [
            "comment"
        ]
    },

    # ==========================
    # Products
    # ==========================

    "products": {

        "file_name": "product.xlsx",

        "primary_key": "id",

        "foreign_keys": [],

        "join_columns": [
            "id"
        ],

        "text_columns": [
            "product_title_fa",
            "product_title_en",
            "title_alt",
            "category_title_fa",
            "category_keywords",
            "brand_name_fa",
            "brand_name_en",
            "product_attributes"
        ],

        "preprocess_columns": [
            "product_title_fa",
            "product_title_en",
            "title_alt",
            "category_title_fa",
            "category_keywords",
            "brand_name_fa",
            "brand_name_en"
        ],

        "embedding_columns": [
            "product_title_fa",
            "title_alt",
            "category_keywords",
            "product_attributes"
        ],

        "rag_columns": [
            "product_title_fa",
            "title_alt",
            "product_attributes"
        ],

        "sentiment_columns": [],

        "datetime_columns": [],

        "numeric_columns": [],

        "categorical_columns": [],
         
        "raw_text_columns": [
            "product_description_fa",
            "product_description_en"
        ]
    }

}











engine = FeatureEngine(FEATURE_MAP)
print("okkkkkkk")
df = engine.run()

print("ok")
print(df.head())




# ================= KPI =================

df["order_count"] = df["Amount_Gross_Order_count"]
df["total_spent"] = df["Amount_Gross_Order_sum"]
df["avg_order_value"] = df["total_spent"] / df["order_count"]

df["recency_days"] = (
    pd.Timestamp.now() - pd.to_datetime(df["DateTime_CartFinalize_max"])
).dt.days

# ================= Segmentation =================

df["customer_type"] = "normal"
df.loc[df["total_spent"] > 10000000, "customer_type"] = "VIP"
df.loc[df["order_count"] > 20, "customer_type"] = "loyal"

# ================= SAVE =================

df.to_csv(r"C:\Users\PSG\Desktop\final_kpi.csv", index=False)

print("KPI FILE SAVED ✅")





