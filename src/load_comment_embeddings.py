import logging
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_batch


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

class ProjectConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Folder containing embedded comment parquet files
    EMBEDDED_COMMENTS_DIR = BASE_DIR / "Dataset" / "embedded_comments+"

    # PostgreSQL connection settings
    DB_NAME = os.getenv("DB_NAME", "ai_project")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # Number of rows sent to PostgreSQL at once
    BATCH_SIZE = 1000

    @classmethod
    def get_pg_conn_params(cls):
        return {
            "dbname": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
        }


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)


# ============================================================
# COMMENT EMBEDDING LOADER
# ============================================================

class CommentEmbeddingLoader:

    def __init__(self):
        self.conn = None

        self.total_parquet_rows = 0
        self.valid_rows = 0
        self.updated_rows = 0
        self.not_found_rows = 0
        self.invalid_id_rows = 0
        self.invalid_embedding_rows = 0

    # --------------------------------------------------------
    # Connect to PostgreSQL
    # --------------------------------------------------------

    def connect(self):
        logging.info("در حال اتصال به PostgreSQL...")

        self.conn = psycopg2.connect(
            **ProjectConfig.get_pg_conn_params()
        )

        self.conn.autocommit = False

        logging.info(
            "اتصال به PostgreSQL با موفقیت برقرار شد."
        )

    # --------------------------------------------------------
    # Find parquet files
    # --------------------------------------------------------

    def get_parquet_files(self):
        folder = ProjectConfig.EMBEDDED_COMMENTS_DIR

        if not folder.exists():
            raise FileNotFoundError(
                f"پوشه embedded_comment پیدا نشد:\n{folder}"
            )

        files = sorted(folder.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"هیچ فایل Parquet در این پوشه پیدا نشد:\n{folder}"
            )

        logging.info(
            f"تعداد فایل‌های embedding پیدا شده: {len(files)}"
        )

        for file in files:
            logging.info(f"  - {file.name}")

        return files

    # --------------------------------------------------------
    # Check parquet schema
    # --------------------------------------------------------

    def validate_schema(self, parquet_file):
        schema = pq.read_schema(parquet_file)

        if "id" not in schema.names:
            raise ValueError(
                f"ستون id در فایل {parquet_file.name} وجود ندارد."
            )

        if "embedding" not in schema.names:
            raise ValueError(
                f"ستون embedding در فایل {parquet_file.name} وجود ندارد."
            )

        id_type = schema.field("id").type
        embedding_type = schema.field("embedding").type

        logging.info(
            f"فایل: {parquet_file.name}"
        )

        logging.info(
            f"نوع id: {id_type}"
        )

        logging.info(
            f"نوع embedding: {embedding_type}"
        )

    # --------------------------------------------------------
    # Convert embedding to pgvector format
    # --------------------------------------------------------

    @staticmethod
    def embedding_to_vector(embedding):
        if embedding is None:
            return None

        values = list(embedding)

        # PostgreSQL column is VECTOR(768)
        if len(values) != 768:
            return None

        try:
            return "[" + ",".join(
                str(float(value))
                for value in values
            ) + "]"

        except (TypeError, ValueError):
            return None

    # --------------------------------------------------------
    # Update one batch
    # --------------------------------------------------------

    def update_batch(self, rows):
        if not rows:
            return

        query = """
            UPDATE comments
            SET embedded_comment = %s::vector
            WHERE id = %s
        """

        with self.conn.cursor() as cursor:

            execute_batch(
                cursor,
                query,
                rows,
                page_size=ProjectConfig.BATCH_SIZE
            )

        self.conn.commit()

        # execute_batch rowcount is not reliable for knowing
        # exactly how many rows were affected, so we separately
        # check the IDs that exist in PostgreSQL.

        ids = [comment_id for _, comment_id in rows]

        placeholders = ",".join(
            ["%s"] * len(ids)
        )

        check_query = f"""
            SELECT id
            FROM comments
            WHERE id IN ({placeholders})
        """

        with self.conn.cursor() as cursor:
            cursor.execute(check_query, ids)

            existing_ids = {
                row[0]
                for row in cursor.fetchall()
            }

        batch_updated = sum(
            1
            for _, comment_id in rows
            if comment_id in existing_ids
        )

        batch_not_found = len(rows) - batch_updated

        self.updated_rows += batch_updated
        self.not_found_rows += batch_not_found

    # --------------------------------------------------------
    # Process one parquet file
    # --------------------------------------------------------

    def process_file(self, parquet_file):

        logging.info("=" * 70)
        logging.info(
            f"شروع پردازش فایل: {parquet_file.name}"
        )

        self.validate_schema(parquet_file)

        parquet = pq.ParquetFile(parquet_file)

        batch_number = 0

        for record_batch in parquet.iter_batches(
            batch_size=ProjectConfig.BATCH_SIZE,
            columns=["id", "embedding"]
        ):

            batch_number += 1

            rows = record_batch.to_pylist()

            update_rows = []

            for row in rows:

                self.total_parquet_rows += 1

                raw_id = row.get("id")
                raw_embedding = row.get("embedding")

                # ------------------------------------------------
                # Validate ID
                # ------------------------------------------------

                if raw_id is None:
                    self.invalid_id_rows += 1
                    continue

                try:
                    comment_id = int(raw_id)
                except (TypeError, ValueError):
                    self.invalid_id_rows += 1

                    logging.warning(
                        f"id نامعتبر پیدا شد: {raw_id}"
                    )

                    continue

                # ------------------------------------------------
                # Validate embedding
                # ------------------------------------------------

                vector = self.embedding_to_vector(
                    raw_embedding
                )

                if vector is None:
                    self.invalid_embedding_rows += 1

                    logging.warning(
                        f"Embedding نامعتبر برای comment_id="
                        f"{comment_id}"
                    )

                    continue

                update_rows.append(
                    (vector, comment_id)
                )

            self.valid_rows += len(update_rows)

            # ------------------------------------------------
            # Update PostgreSQL
            # ------------------------------------------------

            if update_rows:
                self.update_batch(update_rows)

            logging.info(
                f"Batch {batch_number} | "
                f"rows={len(rows)} | "
                f"valid={len(update_rows)}"
            )

        logging.info(
            f"پردازش فایل {parquet_file.name} تمام شد."
        )

    # --------------------------------------------------------
    # Run loader
    # --------------------------------------------------------

    def run(self):

        start_time = time.time()

        try:
            parquet_files = self.get_parquet_files()

            self.connect()

            for parquet_file in parquet_files:
                self.process_file(parquet_file)

            elapsed = round(
                time.time() - start_time,
                2
            )

            logging.info("=" * 70)
            logging.info(
                "پردازش embedding کامنت‌ها با موفقیت تمام شد."
            )

            logging.info(
                f"کل ردیف‌های خوانده‌شده از Parquet: "
                f"{self.total_parquet_rows}"
            )

            logging.info(
                f"ردیف‌های معتبر: "
                f"{self.valid_rows}"
            )

            logging.info(
                f"کامنت‌های پیدا شده و Update شده: "
                f"{self.updated_rows}"
            )

            logging.info(
                f"IDهایی که در comments پیدا نشدند: "
                f"{self.not_found_rows}"
            )

            logging.info(
                f"IDهای نامعتبر: "
                f"{self.invalid_id_rows}"
            )

            logging.info(
                f"Embeddingهای نامعتبر: "
                f"{self.invalid_embedding_rows}"
            )

            logging.info(
                f"زمان کل اجرا: {elapsed} ثانیه"
            )

            logging.info("=" * 70)

        except Exception as e:

            if self.conn:
                self.conn.rollback()

            logging.error(
                f"خطا در پردازش embeddingها: {e}"
            )

            raise

        finally:

            if self.conn:
                self.conn.close()

                logging.info(
                    "اتصال PostgreSQL بسته شد."
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    loader = CommentEmbeddingLoader()
    loader.run()