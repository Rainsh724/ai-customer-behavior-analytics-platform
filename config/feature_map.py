"""
# ==========================
Feature Map (Updated & Optimized)
# ==========================
این فایل تنها مرجع معرفی ساختار دیتاست‌ها در کل پروژه است.

تمام بخش‌های پروژه از این فایل استفاده می‌کنند:
✔ Text Preprocessing
✔ Sentiment Analysis
✔ PostgreSQL
✔ pgvector
✔ Embedding
✔ RAG
✔ BI Dashboard
✔ LangGraph Agents

در صورت تغییر دیتاست فقط این فایل بروزرسانی می‌شود.
=============================================================
"""

FEATURE_MAP = {

    # ==========================
    # 1. Comments (Reviews)
    # ==========================
    "digikala-comments": {
        "file_name": "digikala-comments.csv",
        "primary_key": "id",
        "foreign_keys": [
            "product_id"
        ],
        "join_columns": [
            "product_id"
        ],
        "text_columns": [
            "title",
            "body",
            "advantages",
            "disadvantages"
        ],
        "preprocess_columns": [
            "title",
            "body",
            "advantages",
            "disadvantages"
        ],
        "embedding_columns": [
            "title",
            "body",
            "advantages",
            "disadvantages"
        ],
        "rag_columns": [
            "body",
            "advantages",
            "disadvantages"
        ],
        "sentiment_columns": [
            "body",
            "advantages",
            "disadvantages"
        ],
        "datetime_columns": [
            "created_at"
        ],
        "numeric_columns": [
            "rate",
            "likes",
            "dislikes"
        ],
        "categorical_columns": [
            "recommendation_status",
            "seller_title",
            "seller_code",
            "true_to_size_rate"
        ],
        "boolean_columns": [
            "is_buyer"
        ]
    },

    # ==========================
    # 2. User Behavior Logs
    # ==========================
    "user_behavior_logs": {
        "file_name": "user_behavior_logs.csv",
        "primary_key": None,
        "foreign_keys": [
            "user_id",
            "product_id"
        ],
        "join_columns": [
            "user_id",
            "product_id"
        ],
        "text_columns": [],
        "preprocess_columns": [],
        "embedding_columns": [],
        "rag_columns": [],
        "sentiment_columns": [],
        "datetime_columns": [
            "timestamp"
        ],
        "numeric_columns": [],
        "categorical_columns": [
            "event_type",
            "city",
            "session_id"
        ],
        "boolean_columns": []
    },

    # ==========================
    # 3. Products
    # ==========================
    "digikala-products": {
        "file_name": "digikala-products.csv",
        "primary_key": "id",
        "foreign_keys": [],
        "join_columns": [
            "id"
        ],
        "text_columns": [
            "title_fa"
        ],
        "preprocess_columns": [
            "title_fa"
        ],
        "embedding_columns": [
            "title_fa",
            "Category1",
            "Category2",
            "sub_category",
            "Brand"
        ],
        "rag_columns": [
            "title_fa",
            "Category1",
            "Category2",
            "sub_category"
        ],
        "sentiment_columns": [],
        "datetime_columns": [],
        "numeric_columns": [
            "Rate",
            "Rate_cnt",
            "Price",
            "min_price_last_month"
        ],
        "categorical_columns": [
            "Category1",
            "Category2",
            "Brand",
            "Seller",
            "sub_category"
        ],
        "boolean_columns": [
            "Is_Fake"
        ]
    }
}

# ==========================
# Helper Functions
# ==========================

def get_dataset(name):
    """برگرداندن تنظیمات یک دیتاست"""
    return FEATURE_MAP.get(name, {})

def get_text_columns(name):
    return FEATURE_MAP[name].get("text_columns", [])

def get_preprocess_columns(name):
    return FEATURE_MAP[name].get("preprocess_columns", [])

def get_embedding_columns(name):
    return FEATURE_MAP[name].get("embedding_columns", [])

def get_rag_columns(name):
    return FEATURE_MAP[name].get("rag_columns", [])

def get_sentiment_columns(name):
    return FEATURE_MAP[name].get("sentiment_columns", [])

def get_datetime_columns(name):
    return FEATURE_MAP[name].get("datetime_columns", [])

def get_numeric_columns(name):
    return FEATURE_MAP[name].get("numeric_columns", [])

def get_join_columns(name):
    return FEATURE_MAP[name].get("join_columns", [])

def get_categorical_columns(name):
    return FEATURE_MAP[name].get("categorical_columns", [])

def get_boolean_columns(name):
    """برگرداندن ستون‌های بولین (True/False) - ایده اضافه شده"""
    return FEATURE_MAP[name].get("boolean_columns", [])