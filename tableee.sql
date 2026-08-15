

CREATE TABLE IF NOT EXISTS cities (
    city_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id),
    city_id INT REFERENCES cities(city_id)
);


CREATE TABLE IF NOT EXISTS brands (
    brand_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS categories (
    category_id BIGSERIAL PRIMARY KEY,
    category2 VARCHAR(255) NOT NULL,
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
    rate_cnt BIGINT,
    raw_text TEXT,
    raw_text_normalized TEXT
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
    
    body TEXT,
    rate DOUBLE PRECISION,
    recommendation_status VARCHAR(50),
    likes INT DEFAULT 0,
    dislikes INT DEFAULT 0,
    advantages JSONB,                                   
    disadvantages JSONB,                                
    true_to_size_rate VARCHAR(20),                     
    predicted_sentiment VARCHAR(50),
    raw_text TEXT,
    raw_text_normalized TEXT,
    created_at TIMESTAMPTZ                                   
);


-- ==========================================
-- INDEXES - CORE  ESSENTIAL
-- ==========================================


CREATE INDEX IF NOT EXISTS idx_sessions_user
ON sessions(user_id);



CREATE INDEX IF NOT EXISTS idx_products_category
ON products(category_id);

CREATE INDEX IF NOT EXISTS idx_products_brand
ON products(brand_id);



CREATE INDEX IF NOT EXISTS idx_behavior_session
ON user_behavior_logs(session_id);



CREATE INDEX IF NOT EXISTS idx_behavior_product_event
ON user_behavior_logs(product_id, event_type);


CREATE INDEX IF NOT EXISTS idx_comments_product
ON comments(product_id);