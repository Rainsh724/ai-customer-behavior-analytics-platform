import logging
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm


class ProjectConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    EMBEDDED_COMMENTS_DIR = BASE_DIR / "Dataset" / "embedded_comments"
    DB_NAME = os.getenv("DB_NAME", "ai_project")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "") 
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class CommentEmbeddingLoader:
    def __init__(self):
        self.conn = None
        self.total_parquet_rows = 0
        self.valid_rows = 0
        self.inserted_rows = 0
        self.invalid_id_rows = 0
        self.invalid_embedding_rows = 0

    def connect(self):
        self.conn = psycopg2.connect(**ProjectConfig.get_pg_conn_params())
        self.conn.autocommit = False

    def get_parquet_files(self):
        folder = ProjectConfig.EMBEDDED_COMMENTS_DIR
        files = sorted(folder.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"هیچ فایلی پیدا نشد: {folder}")
        return files

    @staticmethod
    def embedding_to_vector(embedding):
        if embedding is None or len(list(embedding)) != 768:
            return None
        try:
            return "[" + ",".join(str(float(v)) for v in embedding) + "]"
        except (TypeError, ValueError):
            return None

    def insert_batch(self, rows):
        if not rows: return
        query = """
            INSERT INTO comments_embedding (id, embedded_comment)
            SELECT v.id, v.vector::vector
            FROM (VALUES %s) AS v(id, vector)
            INNER JOIN comments c ON c.id = v.id
            ON CONFLICT (id) DO NOTHING
            RETURNING comments_embedding.id;
        """
        with self.conn.cursor() as cursor:
            inserted_ids = execute_values(cursor, query, rows, page_size=ProjectConfig.BATCH_SIZE, fetch=True)
        self.conn.commit()
        self.inserted_rows += len(inserted_ids) if inserted_ids else 0

    def process_file(self, parquet_file, pbar):
        parquet = pq.ParquetFile(parquet_file)
        
        for record_batch in parquet.iter_batches(batch_size=ProjectConfig.BATCH_SIZE, columns=["id", "embedding"]):
            rows = record_batch.to_pylist()
            insert_rows = []

            for row in rows:
                self.total_parquet_rows += 1
                raw_id = row.get("id")
                raw_embedding = row.get("embedding")

                if raw_id is None:
                    self.invalid_id_rows += 1
                    continue
                try:
                    comment_id = int(raw_id)
                except:
                    self.invalid_id_rows += 1
                    continue

                vector = self.embedding_to_vector(raw_embedding)
                if vector is None:
                    self.invalid_embedding_rows += 1
                    continue

                insert_rows.append((comment_id, vector))

            self.valid_rows += len(insert_rows)
            if insert_rows:
                self.insert_batch(insert_rows)
            
            # آپدیت کردن نوار پیشرفت
            pbar.update(len(rows)) 

    def run(self):
        start_time = time.time()
        try:
            parquet_files = self.get_parquet_files()
            self.connect()

            # محاسبه سریع کل ردیف‌ها برای تنظیم نوار پیشرفت
            logging.info("در حال محاسبه حجم کل فایل‌ها...")
            total_rows_across_files = sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files)

            logging.info("شروع انتقال داده‌ها به دیتابیس...")
            
            # ساخت نوار پیشرفت
            with tqdm(total=total_rows_across_files, desc="Processing Embeddings", unit="row") as pbar:
                for parquet_file in parquet_files:
                    self.process_file(parquet_file, pbar)

            elapsed = round(time.time() - start_time, 2)
            logging.info(f"پایان موفقیت‌آمیز! ردیف‌های Insert شده: {self.inserted_rows} در {elapsed} ثانیه.")

        except Exception as e:
            if self.conn: self.conn.rollback()
            logging.error(f"خطا: {e}")
        finally:
            if self.conn: self.conn.close()

if __name__ == "__main__":
    CommentEmbeddingLoader().run()