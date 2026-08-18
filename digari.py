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

    DATASET_DIR = BASE_DIR / "Dataset" / "Cleaned_output"

    PRODUCTS_DIR = DATASET_DIR / "digikala-products_parts"
    LOGS_DIR = DATASET_DIR / "user_behavior_logs_parts"
    COMMENTS_DIR = DATASET_DIR / "comment_json"

    # اگر فایل aspect جدا داری، مسیرش را اینجا قرار بده
    COMMENT_ASPECTS_DIR = DATASET_DIR / "comment_aspects"

    # PostgreSQL
    DB_NAME = os.getenv("DB_NAME", "ai_project")
    DB_USER = os.getenv("DB_USER", "admin1")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "13851385")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # DuckDB
    DUCKDB_MEMORY_LIMIT = os.getenv(
        "DUCKDB_MEMORY_LIMIT",
        "6GB"
    )

    DUCKDB_THREADS = int(
        os.getenv(
            "DUCKDB_THREADS",
            "4"
        )
    )

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
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DUCKDB ETL LOADER
# ============================================================

class DuckDBETLLoader:

    def __init__(self):

        self.con = None
        self.metrics = {}

        self._connect()
        self._setup_duckdb()
        self._attach_postgresql()

    # ========================================================
    # CONNECTION
    # ========================================================

    def _connect(self):

        logger.info("ایجاد اتصال DuckDB...")

        self.con = duckdb.connect(
            database=":memory:"
        )

    # ========================================================
    # DUCKDB CONFIGURATION
    # ========================================================

    def _setup_duckdb(self):

        logger.info("تنظیم DuckDB...")

        self.con.execute(
            f"""
            SET memory_limit = '{ProjectConfig.DUCKDB_MEMORY_LIMIT}';
            """
        )

        self.con.execute(
            f"""
            SET threads = {ProjectConfig.DUCKDB_THREADS};
            """
        )

        # PostgreSQL extension
        self.con.execute(
            "INSTALL postgres;"
        )

        self.con.execute(
            "LOAD postgres;"
        )

        # JSON extension
        self.con.execute(
            "INSTALL json;"
        )

        self.con.execute(
            "LOAD json;"
        )

    # ========================================================
    # POSTGRES ATTACH
    # ========================================================

    def _attach_postgresql(self):

        logger.info(
            "اتصال DuckDB به PostgreSQL..."
        )

        conn_str = ProjectConfig.get_pg_conn_str()

        self.con.execute(
            f"""
            ATTACH '{conn_str}'
            AS pg
            (TYPE POSTGRES);
            """
        )

        logger.info(
            "اتصال PostgreSQL با موفقیت برقرار شد."
        )

    # ========================================================
    # PATH HELPERS
    # ========================================================

    @staticmethod
    def _get_glob_path(folder: Path) -> str:

        if not folder.exists():

            raise FileNotFoundError(
                f"پوشه پیدا نشد: {folder}"
            )

        files = list(folder.glob("*.parquet"))

        if not files:

            raise FileNotFoundError(
                f"هیچ فایل Parquet در مسیر زیر پیدا نشد:\n{folder}"
            )

        return str(
            folder / "*.parquet"
        ).replace("\\", "/")

    # ========================================================
    # STEP EXECUTION
    # ========================================================

    def _execute_etl_step(
        self,
        step_name: str,
        query: str
    ):

        logger.info("=" * 70)
        logger.info(
            f"شروع مرحله: {step_name}"
        )
        logger.info("=" * 70)

        start_time = time.perf_counter()

        try:

            self.con.execute(query)

            elapsed = round(
                time.perf_counter() - start_time,
                2
            )

            self.metrics[step_name] = (
                f"SUCCESS ({elapsed}s)"
            )

            logger.info(
                f"مرحله {step_name} با موفقیت انجام شد."
            )

            logger.info(
                f"زمان اجرا: {elapsed} ثانیه"
            )

        except Exception as exc:

            elapsed = round(
                time.perf_counter() - start_time,
                2
            )

            self.metrics[step_name] = (
                f"FAILED ({elapsed}s): {exc}"
            )

            logger.exception(
                f"خطا در مرحله {step_name}"
            )

            raise

    # ========================================================
    # USERS
    # ========================================================

    def load_users(self):

        path = self._get_glob_path(
            ProjectConfig.LOGS_DIR
        )

        query = f"""
        INSERT INTO pg.users (
            user_id
        )

        SELECT DISTINCT
            CAST(user_id AS BIGINT)

        FROM read_parquet(
            '{path}',
            union_by_name = true
        )

        WHERE
            user_id IS NOT NULL

        ON CONFLICT (user_id)
        DO NOTHING;
        """

        self._execute_etl_step(
            "users",
            query
        )

    # ========================================================
    # CITIES
    # ========================================================

    def load_cities(self):

        path = self._get_glob_path(
            ProjectConfig.LOGS_DIR
        )

        query = f"""
        INSERT INTO pg.cities (
            name
        )

        SELECT DISTINCT
            TRIM(
                CAST(city AS VARCHAR)
            ) AS name

        FROM read_parquet(
            '{path}',
            union_by_name = true
        )

        WHERE
            city IS NOT NULL
            AND TRIM(
                CAST(city AS VARCHAR)
            ) <> ''

        ON CONFLICT (name)
        DO NOTHING;
        """

        self._execute_etl_step(
            "cities",
            query
        )

    # ========================================================
    # BRANDS
    # ========================================================

    def load_brands(self):

        path = self._get_glob_path(
            ProjectConfig.PRODUCTS_DIR
        )

        query = f"""
        INSERT INTO pg.brands (
            name
        )

        SELECT DISTINCT
            TRIM(
                CAST(Brand AS VARCHAR)
            ) AS name

        FROM read_parquet(
            '{path}',
            union_by_name = true
        )

        WHERE
            Brand IS NOT NULL
            AND TRIM(
                CAST(Brand AS VARCHAR)
            ) <> ''

        ON CONFLICT (name)
        DO NOTHING;
        """

        self._execute_etl_step(
            "brands",
            query
        )

    # ========================================================
    # CATEGORIES
    # ========================================================

    def load_categories(self):

        path = self._get_glob_path(
            ProjectConfig.PRODUCTS_DIR
        )

        query = f"""
        INSERT INTO pg.categories (
            category1,
            category2,
            sub_category
        )

        SELECT DISTINCT

            TRIM(
                CAST(Category1 AS VARCHAR)
            ) AS category1,

            NULLIF(
                TRIM(
                    CAST(Category2 AS VARCHAR)
                ),
                ''
            ) AS category2,

            NULLIF(
                TRIM(
                    CAST(sub_category AS VARCHAR)
                ),
                ''
            ) AS sub_category

        FROM read_parquet(
            '{path}',
            union_by_name = true
        )

        WHERE

            Category1 IS NOT NULL

            AND TRIM(
                CAST(Category1 AS VARCHAR)
            ) <> ''

        ON CONFLICT (
            category1,
            category2,
            sub_category
        )

        DO NOTHING;
        """

        self._execute_etl_step(
            "categories",
            query
        )

    # ========================================================
    # SELLERS
    # ========================================================

    def load_sellers(self):

        path = self._get_glob_path(
            ProjectConfig.PRODUCTS_DIR
        )

        query = f"""
        INSERT INTO pg.sellers (
            seller_title
        )

        SELECT DISTINCT

            TRIM(
                CAST(Seller AS VARCHAR)
            ) AS seller_title

        FROM read_parquet(
            '{path}',
            union_by_name = true
        )

        WHERE

            Seller IS NOT NULL

            AND TRIM(
                CAST(Seller AS VARCHAR)
            ) <> ''

        ON CONFLICT (seller_title)
        DO NOTHING;
        """

        self._execute_etl_step(
            "sellers",
            query
        )

    # ========================================================
    # SESSIONS
    # ========================================================

    def load_sessions(self):

        path = self._get_glob_path(
            ProjectConfig.LOGS_DIR
        )

        query = f"""
        INSERT INTO pg.sessions (
            session_id,
            user_id,
            city_id
        )

        SELECT DISTINCT

            TRIM(
                CAST(src.session_id AS VARCHAR)
            ) AS session_id,

            CAST(
                src.user_id AS BIGINT
            ) AS user_id,

            c.city_id

        FROM read_parquet(
            '{path}',
            union_by_name = true
        ) AS src

        INNER JOIN pg.users AS u

            ON u.user_id =
                CAST(
                    src.user_id AS BIGINT
                )

        LEFT JOIN pg.cities AS c

            ON c.name =
                TRIM(
                    CAST(src.city AS VARCHAR)
                )

        WHERE

            src.session_id IS NOT NULL

            AND TRIM(
                CAST(src.session_id AS VARCHAR)
            ) <> ''

            AND src.user_id IS NOT NULL

        ON CONFLICT (session_id)

        DO UPDATE SET

            user_id =
                EXCLUDED.user_id,

            city_id =
                EXCLUDED.city_id;
        """

        self._execute_etl_step(
            "sessions",
            query
        )

    # ========================================================
    # PRODUCTS
    # ========================================================

    def load_products(self):

        path = self._get_glob_path(
            ProjectConfig.PRODUCTS_DIR
        )

        query = f"""
        INSERT INTO pg.products (

            id,
            title_fa,
            brand_id,
            category_id,
            seller_id,
            price,
            min_price_last_month,
            is_fake,
            rate,
            rate_cnt

        )

        SELECT DISTINCT

            CAST(
                src.id AS BIGINT
            ) AS id,

            CAST(
                src.title_fa AS VARCHAR
            ) AS title_fa,

            b.brand_id,

            c.category_id,

            s.seller_id,

            CAST(
                src.Price AS BIGINT
            ) AS price,

            CAST(
                src.min_price_last_month
                AS BIGINT
            ) AS min_price_last_month,

            COALESCE(
                CAST(
                    src.Is_Fake AS BOOLEAN
                ),
                FALSE
            ) AS is_fake,

            CAST(
                src.Rate AS DOUBLE PRECISION
            ) AS rate,

            CAST(
                src.Rate_cnt AS BIGINT
            ) AS rate_cnt

        FROM read_parquet(
            '{path}',
            union_by_name = true
        ) AS src

        LEFT JOIN pg.brands AS b

            ON b.name =
                TRIM(
                    CAST(src.Brand AS VARCHAR)
                )

        LEFT JOIN pg.categories AS c

            ON c.category1 =
                TRIM(
                    CAST(src.Category1 AS VARCHAR)
                )

            AND c.category2
                IS NOT DISTINCT FROM

                NULLIF(
                    TRIM(
                        CAST(
                            src.Category2
                            AS VARCHAR
                        )
                    ),
                    ''
                )

            AND c.sub_category
                IS NOT DISTINCT FROM

                NULLIF(
                    TRIM(
                        CAST(
                            src.sub_category
                            AS VARCHAR
                        )
                    ),
                    ''
                )

        LEFT JOIN pg.sellers AS s

            ON s.seller_title =
                TRIM(
                    CAST(src.Seller AS VARCHAR)
                )

        WHERE

            src.id IS NOT NULL

            AND src.title_fa IS NOT NULL

        ON CONFLICT (id)

        DO UPDATE SET

            title_fa =
                EXCLUDED.title_fa,

            brand_id =
                EXCLUDED.brand_id,

            category_id =
                EXCLUDED.category_id,

            seller_id =
                EXCLUDED.seller_id,

            price =
                EXCLUDED.price,

            min_price_last_month =
                EXCLUDED.min_price_last_month,

            is_fake =
                EXCLUDED.is_fake,

            rate =
                EXCLUDED.rate,

            rate_cnt =
                EXCLUDED.rate_cnt;
        """

        self._execute_etl_step(
            "products",
            query
        )

    # ========================================================
    # USER BEHAVIOR LOGS
    # ========================================================

    def load_user_behavior_logs(self):

        path = self._get_glob_path(
            ProjectConfig.LOGS_DIR
        )

        query = f"""
        INSERT INTO pg.user_behavior_logs (

            session_id,
            product_id,
            event_type,
            timestamp

        )

        SELECT DISTINCT

            TRIM(
                CAST(src.session_id AS VARCHAR)
            ) AS session_id,

            CAST(
                src.product_id AS BIGINT
            ) AS product_id,

            CAST(
                src.event_type AS VARCHAR
            ) AS event_type,

            CAST(
                src.timestamp AS TIMESTAMPTZ
            ) AS timestamp

        FROM read_parquet(
            '{path}',
            union_by_name = true
        ) AS src

        INNER JOIN pg.sessions AS sess

            ON sess.session_id =
                TRIM(
                    CAST(
                        src.session_id
                        AS VARCHAR
                    )
                )

        INNER JOIN pg.products AS prod

            ON prod.id =
                CAST(
                    src.product_id
                    AS BIGINT
                )

        WHERE

            src.session_id IS NOT NULL

            AND src.product_id IS NOT NULL

            AND src.event_type IS NOT NULL

            AND src.timestamp IS NOT NULL

        ON CONFLICT (
            session_id,
            product_id,
            event_type,
            timestamp
        )

        DO NOTHING;
        """

        self._execute_etl_step(
            "user_behavior_logs",
            query
        )

    # ========================================================
    # COMMENTS
    # ========================================================
    def load_comments(self):

        path = self._get_glob_path(
            ProjectConfig.COMMENTS_DIR
        )

        query = f"""
        INSERT INTO pg.comments (

            id,
            product_id,
            is_buyer,
            rate,
            recommendation_status,
            likes,
            dislikes,
            raw_text_normalized,
            created_at

        )

        SELECT DISTINCT

            CAST(
                src.id AS BIGINT
            ) AS id,

            CAST(
                src.product_id AS BIGINT
            ) AS product_id,

            CAST(
                src.is_buyer AS BOOLEAN
            ) AS is_buyer,

            CAST(
                src.rate AS DOUBLE PRECISION
            ) AS rate,

            CAST(
                src.recommendation_status
                AS VARCHAR
            ) AS recommendation_status,

            COALESCE(
                CAST(
                    src.likes AS INTEGER
                ),
                0
            ) AS likes,

            COALESCE(
                CAST(
                    src.dislikes AS INTEGER
                ),
                0
            ) AS dislikes,

            CAST(
                src.raw_text_normalized AS TEXT
            ) AS raw_text_normalized,

            CAST(
                src.created_at AS TIMESTAMPTZ
            ) AS created_at

        FROM read_parquet(
            '{path}',
            union_by_name = true
        ) AS src

        INNER JOIN pg.products AS p

            ON p.id =
                CAST(
                    src.product_id AS BIGINT
                )

        WHERE

            src.id IS NOT NULL

            AND src.product_id IS NOT NULL

        ON CONFLICT (id)

        DO UPDATE SET

            product_id =
                EXCLUDED.product_id,

            is_buyer =
                EXCLUDED.is_buyer,

            rate =
                EXCLUDED.rate,

            recommendation_status =
                EXCLUDED.recommendation_status,

            likes =
                EXCLUDED.likes,

            dislikes =
                EXCLUDED.dislikes,

            raw_text_normalized =
                EXCLUDED.raw_text_normalized,

            created_at =
                EXCLUDED.created_at;
        """

        self._execute_etl_step(
            "comments",
            query
        )

    # ========================================================
    # COMMENT ASPECTS
    # ========================================================

    def load_comment_aspects(self):

        folder = ProjectConfig.COMMENT_ASPECTS_DIR

        if not folder.exists():

            logger.info(
                "پوشه comment_aspects وجود ندارد."
            )

            logger.info(
                "مرحله comment_aspects رد شد."
            )

            return

        files = list(
            folder.glob("*.parquet")
        )

        if not files:

            logger.info(
                "هیچ فایل Parquet برای "
                "comment_aspects وجود ندارد."
            )

            return

        path = self._get_glob_path(
            folder
        )

        query = f"""
        INSERT INTO pg.comment_aspects (

            comment_id,
            term,
            sentiment,
            negative_pct,
            neutral_pct,
            positive_pct

        )

        SELECT DISTINCT

            CAST(
                src.comment_id AS BIGINT
            ) AS comment_id,

            CAST(
                src.term AS VARCHAR
            ) AS term,

            CAST(
                src.sentiment AS VARCHAR
            ) AS sentiment,

            CAST(
                src.negative_pct
                AS DOUBLE PRECISION
            ) AS negative_pct,

            CAST(
                src.neutral_pct
                AS DOUBLE PRECISION
            ) AS neutral_pct,

            CAST(
                src.positive_pct
                AS DOUBLE PRECISION
            ) AS positive_pct

        FROM read_parquet(
            '{path}',
            union_by_name = true
        ) AS src

        INNER JOIN pg.comments AS c

            ON c.id =
                CAST(
                    src.comment_id
                    AS BIGINT
                )

        WHERE

            src.comment_id IS NOT NULL;
        """

        self._execute_etl_step(
            "comment_aspects",
            query
        )

    # ========================================================
    # DATABASE VALIDATION
    # ========================================================

    def validate_counts(self):

        logger.info("=" * 70)
        logger.info(
            "بررسی تعداد رکوردهای PostgreSQL"
        )
        logger.info("=" * 70)

        tables = [
            "users",
            "cities",
            "brands",
            "categories",
            "sellers",
            "sessions",
            "products",
            "user_behavior_logs",
            "comments",
            "comment_aspects",
        ]

        for table in tables:

            try:

                result = self.con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM pg.{table};
                    """
                ).fetchone()

                count = result[0]

                logger.info(
                    f"{table:<25} {count:,}"
                )

            except Exception as exc:

                logger.warning(
                    f"خطا در شمارش {table}: {exc}"
                )

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    def run(self):

        total_start = time.perf_counter()

        try:

            logger.info("=" * 70)
            logger.info(
                "شروع ETL Parquet → DuckDB → PostgreSQL"
            )
            logger.info("=" * 70)

            # ------------------------------------------------
            # DIMENSION TABLES
            # ------------------------------------------------

            self.load_users()

            self.load_cities()

            self.load_brands()

            self.load_categories()

            self.load_sellers()

            # ------------------------------------------------
            # SESSIONS
            # ------------------------------------------------

            self.load_sessions()

            # ------------------------------------------------
            # PRODUCTS
            # ------------------------------------------------

            self.load_products()

            # ------------------------------------------------
            # BEHAVIOR
            # ------------------------------------------------

            self.load_user_behavior_logs()

            # ------------------------------------------------
            # COMMENTS
            # ------------------------------------------------

            self.load_comments()

            # ------------------------------------------------
            # COMMENT ASPECTS
            # ------------------------------------------------

            self.load_comment_aspects()

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            self.validate_counts()

            # ------------------------------------------------
            # FINAL REPORT
            # ------------------------------------------------

            total_time = round(
                time.perf_counter()
                - total_start,
                2
            )

            logger.info("=" * 70)
            logger.info(
                "ETL با موفقیت کامل انجام شد."
            )

            logger.info(
                f"زمان کل: {total_time} ثانیه"
            )

            logger.info("=" * 70)

            logger.info(
                "خلاصه مراحل:"
            )

            for step, status in self.metrics.items():

                logger.info(
                    f"{step:<25} : {status}"
                )

            logger.info("=" * 70)

            logger.info(
                "Embeddingها در این مرحله "
                "وارد نشده‌اند."
            )

            logger.info(
                "vector_title و embedded_comment "
                "فعلاً NULL باقی می‌مانند."
            )

        except Exception as exc:

            logger.exception(
                f"ETL با خطا متوقف شد: {exc}"
            )

            raise

        finally:

            self.close()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.con is not None:

            try:
                self.con.close()

                logger.info(
                    "اتصال DuckDB بسته شد."
                )

            except Exception:

                pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    loader = DuckDBETLLoader()

    loader.run()