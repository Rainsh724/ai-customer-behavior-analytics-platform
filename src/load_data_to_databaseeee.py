import logging
import os
import time
from pathlib import Path
import duckdb


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

class ProjectConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent


    DATASET_DIR = BASE_DIR/ "Dataset" / "Cleaned_output"

    PRODUCTS_DIR = DATASET_DIR / "digikala-products_parts"
    LOGS_DIR = DATASET_DIR / "user_behavior_logs_parts"
    COMMENTS_DIR = DATASET_DIR / "absa_results_comments"

    # PostgreSQL Connection Settings
    DB_NAME = os.getenv("DB_NAME", "ai_project")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "zynb1223")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # DuckDB High-Performance Memory Limits
    DUCKDB_MEMORY_LIMIT = "6GB"
    DUCKDB_THREADS = 4

    @classmethod
    def get_pg_conn_str(cls) -> str:
        return (
            f"dbname={cls.DB_NAME} "
            f"user={cls.DB_USER} "
            f"password={cls.DB_PASSWORD} "
            f"host={cls.DB_HOST} "
            f"port={cls.DB_PORT}"
        )


# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)


# ============================================================
# HIGH-PERFORMANCE DUCKDB ETL LOADER
# ============================================================

class DuckDBETLLoader:

    def __init__(self):
        self.con = duckdb.connect()
        self.metrics = {}
        self._setup_connection()

    def _setup_connection(self):
        """تنظیمات بهینه‌سازی سیستم و اتصال پرسرعت DuckDB به PostgreSQL"""
        logging.info("در حال تنظیم اتصالات DuckDB و PostgreSQL...")
        
        self.con.sql(f"SET max_memory = '{ProjectConfig.DUCKDB_MEMORY_LIMIT}';")
        self.con.sql(f"SET threads = {ProjectConfig.DUCKDB_THREADS};")

        self.con.sql("INSTALL postgres; LOAD postgres;")
        self.con.sql("INSTALL json; LOAD json;")

        self.con.sql(
            f"ATTACH '{ProjectConfig.get_pg_conn_str()}' AS pg (TYPE POSTGRES);"
        )
        logging.info("اتصال به دیتابیس PostgreSQL با موفقیت برقرار شد.")

    def _get_glob_path(self, folder: Path) -> str:
        """تولید مسیر استاندارد فایل‌های Parquet"""
        path_pattern = folder / "*.parquet"
        return str(path_pattern).replace("\\", "/")

    def _execute_etl_step(self, step_name: str, query: str):
        """اجرای عملیات ETL با اندازه‌گیری دقیق زمان اجرا"""
        logging.info(f"========== شروع پردازش: {step_name} ==========")
        start_time = time.time()
        try:
            self.con.sql(query)
            elapsed = round(time.time() - start_time, 2)
            self.metrics[step_name] = f"SUCCESS ({elapsed}s)"
            logging.info(f"مرحله {step_name} با موفقیت انجام شد. (زمان: {elapsed} ثانیه)")
        except Exception as e:
            self.metrics[step_name] = f"FAILED: {e}"
            logging.error(f"خطا در مرحله {step_name}: {e}")
            raise

    # ========================================================
    # DIMENSION TABLES
    # ========================================================

    def load_users(self):
        path = self._get_glob_path(ProjectConfig.LOGS_DIR)
        query = f"""
            INSERT INTO pg.users (user_id)
            SELECT DISTINCT CAST(user_id AS BIGINT)
            FROM read_parquet('{path}', union_by_name=true)
            WHERE user_id IS NOT NULL
            ON CONFLICT (user_id) DO NOTHING;
        """
        self._execute_etl_step("users", query)

    def load_cities(self):
        path = self._get_glob_path(ProjectConfig.LOGS_DIR)
        query = f"""
            INSERT INTO pg.cities (city_id, name)
            SELECT 
                ROW_NUMBER() OVER () + COALESCE((SELECT MAX(city_id) FROM pg.cities), 0) AS city_id,
                city_name AS name
            FROM (
                SELECT DISTINCT TRIM(CAST(city AS VARCHAR)) AS city_name
                FROM read_parquet('{path}', union_by_name=true)
                WHERE city IS NOT NULL AND TRIM(CAST(city AS VARCHAR)) <> ''
            )
            ON CONFLICT (name) DO NOTHING;
        """
        self._execute_etl_step("cities", query)

    def load_brands(self):
        path = self._get_glob_path(ProjectConfig.PRODUCTS_DIR)
        query = f"""
            INSERT INTO pg.brands (brand_id, name)
            SELECT 
                ROW_NUMBER() OVER () + COALESCE((SELECT MAX(brand_id) FROM pg.brands), 0) AS brand_id,
                brand_name AS name
            FROM (
                SELECT DISTINCT TRIM(CAST(Brand AS VARCHAR)) AS brand_name
                FROM read_parquet('{path}', union_by_name=true)
                WHERE Brand IS NOT NULL AND TRIM(CAST(Brand AS VARCHAR)) <> ''
            )
            ON CONFLICT (name) DO NOTHING;
        """
        self._execute_etl_step("brands", query)

    def load_categories(self):
        path = self._get_glob_path(ProjectConfig.PRODUCTS_DIR)
        query = f"""
            INSERT INTO pg.categories (category_id, category1, category2, sub_category)
            SELECT 
                ROW_NUMBER() OVER () + COALESCE((SELECT MAX(category_id) FROM pg.categories), 0) AS category_id,
                category1,
                category2,
                sub_category
            FROM (
                SELECT DISTINCT
                    TRIM(CAST(Category1 AS VARCHAR)) AS category1,
                    TRIM(CAST(Category2 AS VARCHAR)) AS category2,
                    NULLIF(TRIM(CAST(sub_category AS VARCHAR)), '') AS sub_category
                FROM read_parquet('{path}', union_by_name=true)
                WHERE Category1 IS NOT NULL 
            )
            ON CONFLICT (category1, category2, sub_category) DO NOTHING;
        """
        self._execute_etl_step("categories", query)

    def load_sellers(self):
        path = self._get_glob_path(ProjectConfig.PRODUCTS_DIR)
        query = f"""
            INSERT INTO pg.sellers (seller_id, seller_title)
            SELECT 
                ROW_NUMBER() OVER () + COALESCE((SELECT MAX(seller_id) FROM pg.sellers), 0) AS seller_id,
                seller_title
            FROM (
                SELECT DISTINCT TRIM(CAST(Seller AS VARCHAR)) AS seller_title
                FROM read_parquet('{path}', union_by_name=true)
                WHERE Seller IS NOT NULL AND TRIM(CAST(Seller AS VARCHAR)) <> ''
            )
            ON CONFLICT (seller_title) DO NOTHING;
        """
        self._execute_etl_step("sellers", query)

    def load_sessions(self):
        path = self._get_glob_path(ProjectConfig.LOGS_DIR)
        query = f"""
            INSERT INTO pg.sessions (session_id, user_id, city_id)
            SELECT DISTINCT
                TRIM(CAST(src.session_id AS VARCHAR)),
                CAST(src.user_id AS BIGINT),
                c.city_id
            FROM read_parquet('{path}', union_by_name=true) AS src
            INNER JOIN pg.users AS u ON u.user_id = CAST(src.user_id AS BIGINT)
            LEFT JOIN pg.cities AS c ON c.name = TRIM(CAST(src.city AS VARCHAR))
            WHERE src.session_id IS NOT NULL AND TRIM(CAST(src.session_id AS VARCHAR)) <> ''
            ON CONFLICT (session_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                city_id = EXCLUDED.city_id;
        """
        self._execute_etl_step("sessions", query)

    # ========================================================
    # FACT & MAIN TABLES
    # ========================================================

    def load_products(self):
        path = self._get_glob_path(ProjectConfig.PRODUCTS_DIR)
        query = f"""
            INSERT INTO pg.products (
                id, title_fa, brand_id, category_id, seller_id,
<<<<<<< HEAD:src/load_data_to_databaseeee.py
                price, min_price_last_month, is_fake, rate, rate_cnt
                
=======
                price, min_price_last_month, is_fake, rate, rate_cnt, raw_text_normalized
>>>>>>> 0a0f2d42b43366764ceb272bd4791217c8bf6aba:src/load_data_to_database.py
            )
            SELECT DISTINCT
                CAST(src.id AS BIGINT),
                CAST(src.title_fa AS VARCHAR),
                b.brand_id,
                c.category_id,
                s.seller_id,
                CAST(src.Price AS BIGINT),
                CAST(src.min_price_last_month AS BIGINT),
                COALESCE(CAST(src.Is_Fake AS BOOLEAN), FALSE),
                CAST(src.Rate AS DOUBLE PRECISION),
<<<<<<< HEAD:src/load_data_to_databaseeee.py
                CAST(src.Rate_cnt AS BIGINT)

=======
                CAST(src.Rate_cnt AS BIGINT),
                CAST(src.raw_text_normalized AS TEXT)
>>>>>>> 0a0f2d42b43366764ceb272bd4791217c8bf6aba:src/load_data_to_database.py
            FROM read_parquet('{path}', union_by_name=true) AS src
            LEFT JOIN pg.brands AS b ON b.name = TRIM(CAST(src.Brand AS VARCHAR))
            LEFT JOIN pg.categories AS c 
                ON c.category1 = TRIM(CAST(src.Category1 AS VARCHAR))
               AND c.category2 = TRIM(CAST(src.Category2 AS VARCHAR))
               AND c.sub_category IS NOT DISTINCT FROM NULLIF(TRIM(CAST(src.sub_category AS VARCHAR)), '')
            LEFT JOIN pg.sellers AS s ON s.seller_title = TRIM(CAST(src.Seller AS VARCHAR))
            WHERE src.id IS NOT NULL
            ON CONFLICT (id) DO UPDATE SET
                title_fa = EXCLUDED.title_fa,
                brand_id = EXCLUDED.brand_id,
                category_id = EXCLUDED.category_id,
                seller_id = EXCLUDED.seller_id,
                price = EXCLUDED.price,
                min_price_last_month = EXCLUDED.min_price_last_month,
                is_fake = EXCLUDED.is_fake,
                rate = EXCLUDED.rate,
<<<<<<< HEAD:src/load_data_to_databaseeee.py
                rate_cnt = EXCLUDED.rate_cnt;
=======
                rate_cnt = EXCLUDED.rate_cnt,
                raw_text_normalized = EXCLUDED.raw_text_normalized;
>>>>>>> 0a0f2d42b43366764ceb272bd4791217c8bf6aba:src/load_data_to_database.py
        """
        self._execute_etl_step("products", query)

    def load_user_behavior_logs(self):
        path = self._get_glob_path(ProjectConfig.LOGS_DIR)
        query = f"""
            INSERT INTO pg.user_behavior_logs (log_id, session_id, product_id, event_type, timestamp)
            SELECT 
                ROW_NUMBER() OVER () + COALESCE((SELECT MAX(log_id) FROM pg.user_behavior_logs), 0) AS log_id,
                session_id,
                product_id,
                event_type,
                timestamp
            FROM (
                SELECT DISTINCT
                    TRIM(CAST(src.session_id AS VARCHAR)) AS session_id,
                    CAST(src.product_id AS BIGINT) AS product_id,
                    CAST(src.event_type AS VARCHAR) AS event_type,
                    CAST(src.timestamp AS TIMESTAMPTZ) AS timestamp
                FROM read_parquet('{path}', union_by_name=true) AS src
                INNER JOIN pg.sessions AS sess ON sess.session_id = TRIM(CAST(src.session_id AS VARCHAR))
                INNER JOIN pg.products AS prod ON prod.id = CAST(src.product_id AS BIGINT)
                WHERE src.session_id IS NOT NULL 
                  AND src.product_id IS NOT NULL
                  AND src.event_type IS NOT NULL
                  AND src.timestamp IS NOT NULL
            )
            ON CONFLICT (
                session_id,
                product_id,
                event_type,
                timestamp
            ) DO NOTHING;
        """
        self._execute_etl_step("user_behavior_logs", query)

    def load_comments(self):
        path = self._get_glob_path(ProjectConfig.COMMENTS_DIR)

        query = f"""
            INSERT INTO pg.comments (
                id,
                product_id,
                is_buyer,
                rate,
                recommendation_status,
                likes,
                dislikes,
<<<<<<< HEAD:src/load_data_to_databaseeee.py
=======
                advantages,
                disadvantages,
                true_to_size_rate,
                predicted_sentiment,
>>>>>>> 0a0f2d42b43366764ceb272bd4791217c8bf6aba:src/load_data_to_database.py
                raw_text_normalized,
                created_at
            )

            SELECT DISTINCT
                CAST(src.id AS BIGINT),

                CAST(src.product_id AS BIGINT),

                CAST(src.is_buyer AS BOOLEAN),

                CAST(src.rate AS DOUBLE PRECISION),

                CAST(src.recommendation_status AS VARCHAR),

                COALESCE(
                    CAST(src.likes AS INTEGER),
                    0
                ),

                COALESCE(
                    CAST(src.dislikes AS INTEGER),
                    0
                ),

<<<<<<< HEAD:src/load_data_to_databaseeee.py
                CAST(src.body AS TEXT),
=======
                
                CAST(src.advantages AS JSON),
                CAST(src.disadvantages AS JSON),

                CAST(src.true_to_size_rate AS VARCHAR),

                CAST(src.predicted_sentiment AS VARCHAR),


                CAST(src.raw_text_normalized AS TEXT),
>>>>>>> 0a0f2d42b43366764ceb272bd4791217c8bf6aba:src/load_data_to_database.py

                CAST(src.created_at AS TIMESTAMPTZ)

            FROM read_parquet(
                '{path}',
                union_by_name=true
            ) AS src

            INNER JOIN pg.products AS p
                ON p.id = CAST(src.product_id AS BIGINT)

            WHERE
                src.id IS NOT NULL
                AND src.product_id IS NOT NULL

            ON CONFLICT (id) DO UPDATE SET
                product_id = EXCLUDED.product_id,
                is_buyer = EXCLUDED.is_buyer,
                rate = EXCLUDED.rate,
                recommendation_status = EXCLUDED.recommendation_status,
                likes = EXCLUDED.likes,
                dislikes = EXCLUDED.dislikes,
                raw_text_normalized = EXCLUDED.raw_text_normalized,
                created_at = EXCLUDED.created_at;
        """

        self._execute_etl_step("comments", query)

    def load_comment_aspects(self):
        path = self._get_glob_path(ProjectConfig.COMMENTS_DIR)
        query = f"""
            INSERT INTO pg.comment_aspects (
                aspect_id,
                comment_id,
                term,
                sentiment,
                negative_pct,
                neutral_pct,
                positive_pct
            )

            SELECT
                ROW_NUMBER() OVER () AS aspect_id,

                CAST(src.id AS BIGINT) AS comment_id,

                NULLIF(
                    TRIM(CAST(aspect.term AS VARCHAR)),
                    ''
                ) AS term,

                NULLIF(
                    TRIM(CAST(aspect.sentiment AS VARCHAR)),
                    ''
                ) AS sentiment,

                CAST(aspect.negative_pct AS DOUBLE) AS negative_pct,
                CAST(aspect.neutral_pct AS DOUBLE) AS neutral_pct,
                CAST(aspect.positive_pct AS DOUBLE) AS positive_pct

            FROM read_parquet(
                '{path}',
                union_by_name=true
            ) AS src

            INNER JOIN pg.comments AS c
                ON c.id = CAST(src.id AS BIGINT)

            CROSS JOIN json_each(
                CAST(src.aspects_json AS JSON)
            ) AS t(key, aspect)

            WHERE
                src.id IS NOT NULL
                AND src.aspects_json IS NOT NULL
                AND src.aspects_json <> '[]'

            ON CONFLICT DO NOTHING;
        """

        self._execute_etl_step("comment_aspects", query)
        

    def sync_sequences(self):
        """همگام‌سازی sequenceهای PostgreSQL پس از درج دستی آی‌دی‌ها"""
        logging.info("در حال همگام‌سازی Sequenceهای PostgreSQL...")
        sync_sql = """
        SELECT setval('cities_city_id_seq', COALESCE((SELECT MAX(city_id) FROM cities), 1));
        SELECT setval('brands_brand_id_seq', COALESCE((SELECT MAX(brand_id) FROM brands), 1));
        SELECT setval('categories_category_id_seq', COALESCE((SELECT MAX(category_id) FROM categories), 1));
        SELECT setval('sellers_seller_id_seq', COALESCE((SELECT MAX(seller_id) FROM sellers), 1));
        SELECT setval('user_behavior_logs_log_id_seq', COALESCE((SELECT MAX(log_id) FROM user_behavior_logs), 1));
        """
        for stmt in sync_sql.strip().split(";"):
            if stmt.strip():
                escaped = stmt.strip().replace("'", "''")
                try:
                    self.con.sql(f"CALL postgres_execute('pg', '{escaped}');")
                except Exception:
                    pass

    # ========================================================
    # PIPELINE EXECUTION
    # ========================================================

    def run(self):
        total_start_time = time.time()
        try:
            logging.info("=" * 60)
            logging.info("شروع اجرای پایپ‌لاین بارگذاری داده‌ها (تزریق مستقیم به PostgreSQL)")
            logging.info("=" * 60)

            # ۱. بارگذاری جداول پایه (Dimension Tables)
            self.load_users()
            self.load_cities()
            self.load_brands()
            self.load_categories()
            self.load_sellers()
            self.load_sessions()
                        
            # ۲. بارگذاری جداول اصلی (Fact & Core Entities)
            self.load_products()
            self.load_user_behavior_logs()
            self.load_comments()
            self.load_comment_aspects()

            # ۳. همگام‌سازی توالی‌ها در دیتابیس
            self.sync_sequences()

            # ۴. گزارش خلاصه عملکرد
            total_time = round(time.time() - total_start_time, 2)
            logging.info("=" * 60)
            logging.info(f"عملیات ETL با موفقیت کامل انجام شد. (زمان کل: {total_time} ثانیه)")
            logging.info("خلاصه وضعیت مراحل:")
            for step, status in self.metrics.items():
                logging.info(f" - {step:<20}: {status}")
            logging.info("=" * 60)

        except Exception as e:
            logging.exception(f"خطای غیرمنتظره در اجرای ETL: {e}")
            raise
        finally:
            self.con.close()
            logging.info("اتصال DuckDB بسته شد.")


# if __name__ == "__main__":
#     loader = DuckDBETLLoader()
#     loader.run()




# برای لود کامنت فقط
# if __name__ == "__main__":
#     loader = DuckDBETLLoader()

#     try:
#         loader.load_comments()
#     finally:
#         loader.con.close()
#         logging.info("اتصال DuckDB بسته شد.")


if __name__ == "__main__":
    loader = DuckDBETLLoader()

    try:
        loader.load_comment_aspects()
    finally:
        loader.con.close()
        logging.info("اتصال DuckDB بسته شد.")