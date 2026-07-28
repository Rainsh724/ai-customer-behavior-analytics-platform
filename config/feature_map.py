"""
# ==========================
Feature Map
# ==========================
این فایل تنها مرجع معرفی ساختار دیتاست‌ها در کل پروژه است.

تمام بخش‌های پروژه از این فایل استفاده می‌کنند:

✔ Text Preprocessing
✔ Sentiment Analysis
✔ PostgreSQL
✔ pgvector
✔ Embedding
✔ RAG
✔ Text-to-SQL
✔ BI Dashboard
✔ Clustering
✔ LangGraph Agents

در صورت تغییر دیتاست فقط این فایل بروزرسانی می‌شود.
=============================================================
"""

FEATURE_MAP = {

    # ==========================
    # Comment
    # ==========================

    "comment": {

        "file_name": "comment.xlsx",

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

        "categorical_columns": []
    },

    # ==========================
    # Reviews (Quality)
    # ==========================

    "keifiat": {

        "file_name": "keifiat.xlsx",

        "primary_key": None,

        "foreign_keys": [
            "product_id",
            "user_id"
        ],

        "join_columns": [
            "product_id",
            "user_id"
        ],

        "text_columns": [
            "title",
            "comment",
            "advantages",
            "disadvantages"
        ],

        "preprocess_columns": [
            "title",
            "comment",
            "advantages",
            "disadvantages"
        ],

        "embedding_columns": [
            "title",
            "comment",
            "advantages",
            "disadvantages"
        ],

        "rag_columns": [
            "comment",
            "advantages",
            "disadvantages"
        ],

        "sentiment_columns": [
            "comment"
        ],

        "datetime_columns": [],

        "numeric_columns": [
            "likes",
            "dislikes"
        ],

        "categorical_columns": [
            "verification_status",
            "recommend"
        ]
    },

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
            "ID_Customer",
            "ID_Item"
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

        "categorical_columns": []
    },

    # ==========================
    # Purchase History
    # ==========================

    "tarikhche kharid": {

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
        ]
    },

    # ==========================
    # Products
    # ==========================

    "product": {

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

        "categorical_columns": []
    }

}


# ==========================
# Helper Functions
# ==========================

def get_dataset(name):
    """برگرداندن تنظیمات یک دیتاست"""
    return FEATURE_MAP.get(name, {})


def get_text_columns(name):
    return FEATURE_MAP[name]["text_columns"]


def get_preprocess_columns(name):
    return FEATURE_MAP[name]["preprocess_columns"]


def get_embedding_columns(name):
    return FEATURE_MAP[name]["embedding_columns"]


def get_rag_columns(name):
    return FEATURE_MAP[name]["rag_columns"]


def get_sentiment_columns(name):
    return FEATURE_MAP[name]["sentiment_columns"]


def get_datetime_columns(name):
    return FEATURE_MAP[name]["datetime_columns"]


def get_numeric_columns(name):
    return FEATURE_MAP[name]["numeric_columns"]


def get_join_columns(name):
    return FEATURE_MAP[name]["join_columns"]