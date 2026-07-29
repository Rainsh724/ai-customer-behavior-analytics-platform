


FEATURE_MAP = {

    # ==========================
    # Comments (Reviews)
    # ==========================

    "comments": {

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
            "disadvantages",
            
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
    # User Behavior Logs
    # ==========================

    "user_behavior": {

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
        ]
    },

    # ==========================
    # Products
    # ==========================

    "products": {

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





