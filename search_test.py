'''
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# مشخص کردن یک پوشه کاملاً امن در درایو D
CACHE_DIR = "D:/HuggingFace_Cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"   
USE_E5_PREFIXES = True

print("Loading embedding model (CPU) on D drive...")
# انتقال مستقیم کش به درایو D با پارامتر cache_folder
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", cache_folder=CACHE_DIR)

# ================== تنظیمات شما ==================
# مشخصات دیتابیس خودت را اینجا وارد کن
POSTGRES_DSN = "dbname=ai_project user=postgres password=zynb1223 host=localhost port=5432"


def embed_query(text: str) -> np.ndarray:
    prefixed = f"query: {text}" if USE_E5_PREFIXES else text
    vec = model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32)

def get_pg_conn():
    conn = psycopg2.connect(POSTGRES_DSN)
    register_vector(conn)
    return conn

def search_comments(query_text: str, top_k: int = 5):
    
    # زمان embedding
    start = time.perf_counter()
    
    q_vec = embed_query(query_text)
    
    embedding_time = time.perf_counter() - start

    conn = get_pg_conn()

    # زمان PostgreSQL
    start = time.perf_counter()

    sql = """
        SELECT id, raw_text_normalized,
               embedded_comment <=> %s AS distance
        FROM comments
        WHERE embedded_comment IS NOT NULL
        ORDER BY embedded_comment <=> %s
        LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (q_vec, q_vec, top_k))
        results = cur.fetchall()

    search_time = time.perf_counter() - start

    conn.close()

    print(f"Embedding time: {embedding_time:.3f} sec")
    print(f"PostgreSQL search time: {search_time:.3f} sec")
    print(f"Total: {embedding_time + search_time:.3f} sec")

    return results

if __name__ == "__main__":
    print("\n=== موتور جستجوی معنایی هوشمند فعال شد ===")
    
    # جستجوی تعاملی (مثل یک چت‌بات)
    while True:
        q = input("\nیک چیزی درباره محصولات بپرس (یا بزن Enter برای خروج): ").strip()
        if not q:
            break
            
        print("در حال جستجو در میان صدها هزار کامنت...")
        results = search_comments(q, top_k=5)
        
        print("\n--- 📝 نتایج یافت شده ---")
        for rid, text, dist in results:
            # هرچه distance به صفر نزدیک‌تر باشد، یعنی شبیه‌تر است
            print(f"ID: {rid} | شباهت (Distance): {dist:.4f}")
            print(f"متن کامنت: {str(text)[:150]}...\n")
'''


import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import os
import time

# مشخص کردن یک پوشه کاملاً امن در درایو D
CACHE_DIR = "D:/HuggingFace_Cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"   
USE_E5_PREFIXES = True

print("Loading embedding model (CPU) on D drive...")
# انتقال مستقیم کش به درایو D با پارامتر cache_folder
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", cache_folder=CACHE_DIR)

# ================== تنظیمات دیتابیس ==================
POSTGRES_DSN = "dbname=ai_project user=postgres password=zynb1223 host=localhost port=5432"


def embed_query(text: str) -> np.ndarray:
    prefixed = f"query: {text}" if USE_E5_PREFIXES else text
    vec = model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32)

def get_pg_conn():
    conn = psycopg2.connect(POSTGRES_DSN)
    register_vector(conn)
    return conn

def search_comments(query_text: str, top_k: int = 5):
    
    # زمان embedding
    start = time.perf_counter()
    q_vec = embed_query(query_text)
    embedding_time = time.perf_counter() - start

    conn = get_pg_conn()

    # زمان PostgreSQL
    start = time.perf_counter()

    # کوئری اصلاح‌شده با ساختار جدید جداول و تنظیمات فوق‌سریع ایندکس HNSW
    sql = """
        SET hnsw.ef_search = 20;
        
        SELECT c.id, c.raw_text_normalized,
               e.embedded_comment <=> %s::vector AS distance
        FROM comment_embeddings e
        INNER JOIN comments c ON e.id = c.id
        ORDER BY e.embedded_comment <=> %s::vector
        LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (q_vec, q_vec, top_k))
        results = cur.fetchall()

    search_time = time.perf_counter() - start
    conn.close()

    print(f"Embedding time: {embedding_time:.3f} sec")
    print(f"PostgreSQL search time: {search_time:.3f} sec")
    print(f"Total: {embedding_time + search_time:.3f} sec")

    return results

if __name__ == "__main__":
    print("\n=== موتور جستجوی معنایی هوشمند فعال شد ===")
    
    # جستجوی تعاملی (مثل یک چت‌بات)
    while True:
        q = input("\nیک چیزی درباره محصولات بپرس (یا بزن Enter برای خروج): ").strip()
        if not q:
            break
            
        print("در حال جستجو در میان صدها هزار کامنت...")
        results = search_comments(q, top_k=5)
        
        print("\n--- 📝 نتایج یافت شده ---")
        for rid, text, dist in results:
            # هرچه distance به صفر نزدیک‌تر باشد، یعنی شبیه‌تر است
            print(f"ID: {rid} | شباهت (Distance): {dist:.4f}")
            print(f"متن کامنت: {str(text)[:150]}...\n")