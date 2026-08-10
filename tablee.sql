
-- ==========================================
-- لایه اول: جداول پایه و مرجع (Dimension Tables)
-- ==========================================

-- ۱. جدول شهرها
CREATE TABLE IF NOT EXISTS cities (
    city_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- ۲. جدول کاربران
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY
);

-- ۳. جدول نشست‌ها (Sessions) - جدول جدید و بهینه‌شده
-- هر نشست متعلق به یک کاربر است
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id),
    city_id INT REFERENCES cities(city_id)
);

-- ۴. جدول برندها
CREATE TABLE IF NOT EXISTS brands (
    brand_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- ۵. جدول دسته‌بندی‌ها
CREATE TABLE IF NOT EXISTS categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    parent_id INT REFERENCES categories(category_id)
);

"""-- ۶. جدول فروشندگان
CREATE TABLE IF NOT EXISTS sellers (
    seller_code VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL
);
"""

-- ==========================================
-- لایه دوم: جداول اصلی و لاگ‌ها (Fact Tables)
-- ==========================================

-- ۷. جدول محصولات
CREATE TABLE IF NOT EXISTS products (
    id BIGINT PRIMARY KEY,
    title_fa VARCHAR(500) NOT NULL,
    brand_id INT REFERENCES brands(brand_id),           
    category_id INT REFERENCES categories(category_id), 
    seller VARCHAR(255),
    price BIGINT,                                       
    min_price_last_month BIGINT,
    is_fake BOOLEAN DEFAULT FALSE,
    rate INT,
    rate_cnt INT,
    raw_text TEXT,
    raw_text_normalized TEXT
);

-- ۸. جدول لاگ رفتار کاربران
-- نکته مهم: user_id از اینجا حذف شد، زیرا از طریق session_id قابل دسترسی است
CREATE TABLE IF NOT EXISTS user_behavior_logs (
    log_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES sessions(session_id), 
    product_id BIGINT REFERENCES products(id),             
    event_type VARCHAR(50),
    timestamp TIMESTAMPTZ,                                   
  
);

-- ۹. جدول نظرات
CREATE TABLE IF NOT EXISTS comments (
    id BIGINT PRIMARY KEY,
    product_id BIGINT REFERENCES products(id),
    seller_code VARCHAR(50),
    title VARCHAR(255),

    is_buyer BOOLEAN,
    
    body TEXT,
    rate INT,
    recommendation_status VARCHAR(50),
    likes INT DEFAULT 0,
    dislikes INT DEFAULT 0,
    advantages JSONB,                                   
    disadvantages JSONB,                                
    true_to_size_rate NUMERIC(5,2),                     
    predicted_sentiment VARCHAR(50),
    raw_text TEXT,
    raw_text_normalized TEXT,
    created_at TIMESTAMPTZ                                   
);

-- ۱۰. جدول جنبه‌های استخراج شده از نظرات (Aspects)
CREATE TABLE IF NOT EXISTS comment_aspects (
    aspect_id BIGSERIAL PRIMARY KEY,
    comment_id BIGINT REFERENCES comments(id) ON DELETE CASCADE,
    aspect VARCHAR(100) NOT NULL,
    sentiment VARCHAR(50) NOT NULL
);


-- ==========================================
-- لایه سوم: ایندکس‌ها برای سرعت بخشیدن به کوئری‌ها
-- ==========================================

-- ایندکس‌های لاگ رفتار برای محاسبات قیف فروش (Funnel)
CREATE INDEX idx_behavior_session ON user_behavior_logs(session_id);
CREATE INDEX idx_behavior_product_event ON user_behavior_logs(product_id, event_type);

-- ایندکس نشست‌ها برای اتصال سریع به کاربر
CREATE INDEX idx_sessions_user ON sessions(user_id);

-- ایندکس‌های نظرات و جنبه‌ها (ابزار اصلی برای تحلیل احساسات)
CREATE INDEX idx_comments_product ON comments(product_id);
CREATE INDEX idx_aspects_comment ON comment_aspects(comment_id);
CREATE INDEX idx_aspects_sentiment ON comment_aspects(aspect, sentiment);