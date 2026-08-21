

CREATE TABLE IF NOT EXISTS cities (
    city_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id)  ON DELETE CASCADE,
    city_id INT REFERENCES cities(city_id)ON DELETE SET NULL
);


CREATE TABLE IF NOT EXISTS brands (
    brand_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS categories (
    category_id BIGSERIAL PRIMARY KEY,
    category2 VARCHAR(255) ,
    category1 VARCHAR(255) NOT NULL,
    sub_category VARCHAR(255),

    UNIQUE NULLS NOT DISTINCT (category1, category2, sub_category)

);



CREATE TABLE IF NOT EXISTS sellers (
    seller_id SERIAL PRIMARY KEY,
    seller_title VARCHAR(255) UNIQUE NOT NULL
);





CREATE TABLE IF NOT EXISTS products (
    id BIGINT PRIMARY KEY,
    title_fa VARCHAR(500) NOT NULL,
    brand_id INT REFERENCES brands(brand_id),           
    category_id BIGINT REFERENCES categories(category_id), 
    seller_id INT REFERENCES sellers(seller_id),
    price BIGINT,                                       
    min_price_last_month BIGINT,
    is_fake BOOLEAN DEFAULT FALSE,
    rate DOUBLE PRECISION,
    rate_cnt BIGINT
    
);
 

CREATE TABLE IF NOT EXISTS user_behavior_logs (
    log_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES sessions(session_id), 
    product_id BIGINT REFERENCES products(id),             
    event_type VARCHAR(50),
    timestamp TIMESTAMPTZ ,                                  
  

    CONSTRAINT unique_behavior_event
    UNIQUE (session_id, product_id, event_type, timestamp)
);


CREATE TABLE IF NOT EXISTS comments (
    id BIGINT PRIMARY KEY,
    product_id BIGINT REFERENCES products(id),
    is_buyer BOOLEAN,

    rate DOUBLE PRECISION,
    recommendation_status VARCHAR(50),
    likes INT DEFAULT 0,
    dislikes INT DEFAULT 0,
    raw_text_normalized TEXT,
                               
    created_at TIMESTAMPTZ 
    -- embedded_comment  VECTOR(768)                             
);

CREATE TABLE IF NOT EXISTS comments_embedding (
     id BIGINT PRIMARY KEY REFERENCES comments(id) ON DELETE CASCADE,
    embedded_comment  VECTOR(768)  
); 




CREATE TABLE IF NOT EXISTS comment_aspects (
    aspect_id BIGSERIAL PRIMARY KEY,

    comment_id BIGINT NOT NULL
        REFERENCES comments(id)
        ON DELETE CASCADE,

    term VARCHAR(255),
    sentiment VARCHAR(50),

    negative_pct DOUBLE PRECISION,
    neutral_pct DOUBLE PRECISION,
    positive_pct DOUBLE PRECISION

);
-- ==========================================
-- INDEXES - CORE  ESSENTIAL
-- ==========================================


CREATE INDEX IF NOT EXISTS idx_sessions_user
ON sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_sessions_city
ON sessions(city_id);

CREATE INDEX IF NOT EXISTS idx_products_category
ON products(category_id);

CREATE INDEX IF NOT EXISTS idx_products_brand
ON products(brand_id);

CREATE INDEX IF NOT EXISTS idx_products_seller
ON products(seller_id);

CREATE INDEX IF NOT EXISTS idx_behavior_session_event
ON user_behavior_logs(session_id, event_type);

CREATE INDEX IF NOT EXISTS idx_behavior_product_event
ON user_behavior_logs(product_id, event_type);

CREATE INDEX IF NOT EXISTS idx_comments_product
ON comments(product_id);

CREATE INDEX IF NOT EXISTS idx_comment_aspects_comment
ON comment_aspects(comment_id);

CREATE INDEX IF NOT EXISTS idx_comment_aspects_term
ON comment_aspects(term);