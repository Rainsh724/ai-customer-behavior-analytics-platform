import logging
from pathlib import Path
import duckdb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ProjectConfig:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATASET_DIR = BASE_DIR / "Dataset"

    # الگوهای دقیق فایل‌ها
    PRODUCTS_PATTERN = (
        DATASET_DIR / "Cleaned_output" / "digikala-products_parts" / "*.parquet"
    )
    LOGS_PATTERN = (
        DATASET_DIR
        / "Cleaned_output"
        / "user_behavior_logs_parts"
        / "*.parquet"
    )

    # تفکیک فایل‌های کامنت بر اساس الگو
    COMMENTS_PATTERN = (
        DATASET_DIR / "Cleaned_output" / "Comments_ABSA" / "part_*.parquet"
    )
    ASPECT_KPI_PATTERN = (
        DATASET_DIR
        / "Cleaned_output"
        / "Comments_ABSA"
        / "aspect_sentiment_percentages.parquet"
    )
    FEATURE_KPI_PATTERN = DATASET_DIR / "Feature_KPI_output" / "*.parquet"

    # تنظیمات دیتابیس PostgreSQL
    DB_NAME = "postgres"
    DB_USER = "postgres"
    DB_PASSWORD = "HiddenPatern"  # <--- رمز عبور دیتابیس را قرار دهید
    DB_HOST = "127.0.0.1"
    DB_PORT = 5432

    @classmethod
    def get_pg_conn_str(cls) -> str:
        return f"dbname={cls.DB_NAME} user={cls.DB_USER} password={cls.DB_PASSWORD} host={cls.DB_HOST} port={cls.DB_PORT}"


class ETLLoader:

    def __init__(self):
        self.con = duckdb.connect()
        self._setup_connection()

    def _setup_connection(self):
        logging.info("در حال اتصال به PostgreSQL...")
        self.con.sql("INSTALL postgres; LOAD postgres;")
        self.con.sql(
            f"ATTACH '{ProjectConfig.get_pg_conn_str()}' AS pg (TYPE POSTGRES);"
        )
        logging.info("اتصال به PostgreSQL برقرار شد.")

    def load_table(self, file_pattern: Path, target_table: str):
        path_str = str(file_pattern).replace("\\", "/")
        logging.info(
            f"در حال لود داده‌ها از {file_pattern.name} به {target_table}..."
        )

        try:
            # ۱. پاکسازی پیشین
            self.con.sql(f"TRUNCATE pg.{target_table};")

            # ۲. تطبیق ستون‌ها
            db_cols = [
                r[0]
                for r in self.con.sql(
                    f"DESCRIBE SELECT * FROM pg.{target_table}"
                ).fetchall()
            ]
            pq_cols = [
                r[0]
                for r in self.con.sql(
                    f"DESCRIBE SELECT * FROM read_parquet('{path_str}', union_by_name=True)"
                ).fetchall()
            ]

            pq_cols_map = {c.lower(): c for c in pq_cols}
            matched_db_cols = []
            matched_pq_cols = []

            for db_col in db_cols:
                if db_col.lower() in pq_cols_map:
                    matched_db_cols.append(f'"{db_col}"')
                    # تبدیل صریح true_to_size_rate به متن جهت جلوگیری از خطای Cast
                    if db_col.lower() == "true_to_size_rate":
                        matched_pq_cols.append(
                            f'CAST("{pq_cols_map[db_col.lower()]}" AS VARCHAR)'
                        )
                    else:
                        matched_pq_cols.append(
                            f'"{pq_cols_map[db_col.lower()]}"'
                        )

            db_cols_str = ", ".join(matched_db_cols)
            pq_cols_str = ", ".join(matched_pq_cols)

            # ۳. شرط فیلتر سطرهای NULL برای product_behavior
            where_clause = ""
            if target_table == "ecommerce.product_behavior":
                where_clause = "WHERE product_id IS NOT NULL"

            query = f"""
                INSERT INTO pg.{target_table} ({db_cols_str})
                SELECT {pq_cols_str} 
                FROM read_parquet('{path_str}', union_by_name=True)
                {where_clause};
            """
# شرط عدم ارسال مقادیر NULL و حذف مقادیر تکراری
            where_clause = ""
            distinct_clause = ""

            if target_table == "ecommerce.product_behavior":
                where_clause = "WHERE product_id IS NOT NULL"
                distinct_clause = "DISTINCT ON (product_id)"

            query = f"""
                INSERT INTO pg.{target_table} ({db_cols_str})
                SELECT {distinct_clause} {pq_cols_str} 
                FROM read_parquet('{path_str}', union_by_name=True)
                {where_clause};
            """
            self.con.sql(query)
            logging.info(f"جدول {target_table} با موفقیت بارگذاری شد.")

        except Exception as e:
            logging.error(f"خطا در بارگذاری جدول {target_table}: {e}")
            raise e


    def run(self):
        try:
            # ۱. کاتالوگ و رفتار کاربران
            self.load_table(
                ProjectConfig.PRODUCTS_PATTERN, "ecommerce.product_master"
            )
            self.load_table(
                ProjectConfig.LOGS_PATTERN, "ecommerce.user_behavior_logs"
            )

            # ۲. کامنت‌ها و KPI جنبه‌ها
            self.load_table(
                ProjectConfig.COMMENTS_PATTERN, "ecommerce.comment_embedding"
            )
            self.load_table(
                ProjectConfig.ASPECT_KPI_PATTERN, "ecommerce.aspect_kpi"
            )

            # ۳. شاخص‌های رفتاری محصول (در صورت وجود)
            if ProjectConfig.FEATURE_KPI_PATTERN.parent.exists():
                self.load_table(
                    ProjectConfig.FEATURE_KPI_PATTERN,
                    "ecommerce.product_behavior",
                )

            logging.info("🎉 تمامی جداول با موفقیت وارد PostgreSQL شدند!")

        except Exception as e:
            logging.error(f"خطا در اجرای پروژه: {e}")
            raise e
        finally:
            self.con.close()


if __name__ == "__main__":
    loader = ETLLoader()
    loader.run()