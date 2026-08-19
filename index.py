CREATE INDEX IF NOT EXISTS idx_comments_embedding_hnsw
ON comments
USING hnsw (embedded_comment vector_cosine_ops);


# ایندکس های  امبدینگ های کامنت ها 