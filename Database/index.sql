CREATE INDEX IF NOT EXISTS idx_comments_embedding_hnsw
ON comments
USING hnsw (embedded_comment vector_cosine_ops);

--# ایندکس های  امبدینگ های کامنت ها 


CREATE INDEX idx_comment_aspects_sentiment_comment
    ON comment_aspects (sentiment, comment_id);

VACUUM ANALYZE comment_aspects;

--# ایندکس های sentimentها

CREATE TABLE product_negative_feedback_summary AS
SELECT c.product_id,
       AVG(ca.negative_pct) AS avg_negative_pct,
       COUNT(DISTINCT c.id) AS comment_cnt
FROM comments c
JOIN comment_aspects ca ON ca.comment_id = c.id
GROUP BY c.product_id;

CREATE UNIQUE INDEX ON product_negative_feedback_summary (product_id);

VACUUM ANALYZE;

CREATE INDEX idx_behavior_event_product
    ON user_behavior_logs (event_type, product_id);


CREATE INDEX ON user_behavior_logs(product_id, event_type); 
CREATE INDEX ON user_behavior_logs(timestamp); 
CREATE INDEX ON comment_aspects(comment_id, sentiment); 
CREATE INDEX ON comments(product_id); 
CREATE INDEX ON products(category_id)
CREATE INDEX idx_behavior_purchase_time
ON user_behavior_logs(event_type, timestamp, product_id);
CREATE INDEX idx_comment_aspects_negative_comment
ON comment_aspects(sentiment, comment_id);
CREATE INDEX idx_comments_created_at
ON comments(created_at);
CREATE INDEX idx_comments_product_date
ON comments(product_id, created_at);
CREATE INDEX idx_behavior_event_time
ON user_behavior_logs(event_type, timestamp);
CREATE INDEX idx_behavior_user_time
ON user_behavior_logs(session_id, timestamp);
CREATE INDEX idx_behavior_product_time
ON user_behavior_logs(product_id, timestamp);
CREATE INDEX idx_comments_product_time
ON comments(product_id, created_at);
CREATE INDEX idx_comments_time_id
ON comments(created_at, id);
CREATE INDEX idx_aspects_sentiment_term
ON comment_aspects(sentiment, term);
CREATE INDEX idx_aspects_comment_sentiment
ON comment_aspects(comment_id, sentiment, term);
CREATE INDEX idx_products_category_brand
ON products(category_id, brand_id);
CREATE INDEX idx_sessions_user_city
ON sessions(user_id, city_id);
CREATE INDEX idx_comments_recommendation
ON comments(recommendation_status);
CREATE INDEX idx_comments_recommendation
ON comments(recommendation_status);
CREATE INDEX idx_products_brand_category
ON products(brand_id, category_id);
CREATE INDEX idx_aspects_sentiment_comment_term
ON comment_aspects(sentiment, comment_id, term);

ANALYZE user_behavior_logs;
ANALYZE comments;
ANALYZE comment_aspects;
ANALYZE products;
ANALYZE categories;
ANALYZE sessions;


-- کد بالا برام ارور میداد چون ستون امبدد کامنت توی جدول کامنت بود کوئری پایین معادل بالاییه


CREATE INDEX IF NOT EXISTS idx_comments_embedding_hnsw
ON comments_embedding
USING hnsw (embedded_comment vector_cosine_ops);


CREATE INDEX IF NOT EXISTS idx_comments_product
ON comments(product_id);

CREATE INDEX IF NOT EXISTS idx_comments_created_at
ON comments(created_at);

CREATE INDEX IF NOT EXISTS idx_comments_product_date
ON comments(product_id, created_at);

CREATE INDEX IF NOT EXISTS idx_comments_time_id
ON comments(created_at, id);

CREATE INDEX IF NOT EXISTS idx_comments_recommendation
ON comments(recommendation_status);

CREATE INDEX IF NOT EXISTS idx_behavior_event_product
ON user_behavior_logs (event_type, product_id);

CREATE INDEX IF NOT EXISTS idx_behavior_product_event
ON user_behavior_logs (product_id, event_type);

CREATE INDEX IF NOT EXISTS idx_behavior_event_time
ON user_behavior_logs (event_type, timestamp);

CREATE INDEX IF NOT EXISTS idx_behavior_user_time
ON user_behavior_logs (session_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_behavior_product_time
ON user_behavior_logs (product_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_behavior_purchase_time
ON user_behavior_logs (event_type, timestamp, product_id);


CREATE INDEX IF NOT EXISTS idx_products_category
ON products(category_id);

CREATE INDEX IF NOT EXISTS idx_products_category_brand
ON products(category_id, brand_id);

CREATE INDEX IF NOT EXISTS idx_products_brand_category
ON products(brand_id, category_id);


CREATE INDEX IF NOT EXISTS idx_sessions_user_city
ON sessions(user_id, city_id);


SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'comment_aspects'
ORDER BY ordinal_position;

CREATE TABLE product_negative_feedback_summary AS
SELECT c.product_id,
       AVG(ca.negative_pct) AS avg_negative_pct,
       COUNT(DISTINCT c.id) AS comment_cnt
FROM comments c
JOIN comment_aspects ca
    ON ca.comment_id = c.id
GROUP BY c.product_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_product_negative_feedback_product
ON product_negative_feedback_summary(product_id);


ANALYZE product_negative_feedback_summary;


ANALYZE user_behavior_logs;
ANALYZE comments;
ANALYZE comments_embedding;
ANALYZE comment_aspects;
ANALYZE products;
ANALYZE categories;
ANALYZE sessions;







--- جدول جدید برای دقت و معیارهای ارزیابی چت ها و عملکرد Llm

CREATE TABLE public.eval_log (
    id                  BIGSERIAL PRIMARY KEY,
    chat_id             TEXT,
    question            TEXT,
    faithfulness_score  INTEGER,
    relevance_score     INTEGER,
    confidence_score    INTEGER,
    grounded            BOOLEAN,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);