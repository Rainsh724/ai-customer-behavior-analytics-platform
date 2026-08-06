-- ۱. تعریف فضای ذخیره‌سازی جدید روی درایو F
CREATE TABLESPACE f_drive_space LOCATION 'F:/pg_data';

-- ۲. هدایت جدول سنگین کامنت‌ها به درایو F
ALTER TABLE ecommerce.comment_embedding SET TABLESPACE f_drive_space;

-- (اختیاری) می‌توانید جدول لوگ‌های رفتاری را هم منتقل کنید:
ALTER TABLE ecommerce.user_behavior_logs SET TABLESPACE f_drive_space;


-- ۱. پاکسازی کامل جداول برای آزاد ساختن فضای دیسک
TRUNCATE TABLE ecommerce.comment_embedding RESTART IDENTITY;
TRUNCATE TABLE ecommerce.user_behavior_logs RESTART IDENTITY;
TRUNCATE TABLE ecommerce.product_master RESTART IDENTITY;

-- ۲. بازگرداندن فضای دیسک به سیستم‌عامل
VACUUM FULL;

-- ==============================================================================
-- 1. ایحاد اسکیما (SCHEMA)
-- ==============================================================================
CREATE SCHEMA IF NOT EXISTS ecommerce;

-- ==============================================================================
-- 2. ساخت جداول اصلی (TABLES)
-- ==============================================================================

-- ۲.۱ جدول کاتالوگ و اطلاعات محصولات (۱۴ ستون مطابقت داده‌شده با Parquet)
DROP TABLE IF EXISTS ecommerce.product_master CASCADE;
CREATE TABLE ecommerce.product_master (
                                          id                   BIGINT PRIMARY KEY,
                                          title_fa             TEXT,
                                          rate                 DOUBLE PRECISION,
                                          rate_cnt             DOUBLE PRECISION,
                                          category1            VARCHAR(255),
                                          category2            VARCHAR(255),
                                          brand                VARCHAR(255),
                                          price                DOUBLE PRECISION,
                                          seller               VARCHAR(255),
                                          is_fake              BOOLEAN,
                                          min_price_last_month DOUBLE PRECISION,
                                          sub_category         VARCHAR(255),
                                          raw_text             TEXT,
                                          raw_text_normalized  TEXT
);

-- ۲.۲ جدول لاگ‌های رفتار کاربران (۸ ستون مطابقت داده‌شده با Parquet)
DROP TABLE IF EXISTS ecommerce.user_behavior_logs CASCADE;
CREATE TABLE ecommerce.user_behavior_logs (
                                              user_id              BIGINT,
                                              session_id           VARCHAR(255),
                                              product_id           BIGINT,
                                              event_type           VARCHAR(100),
                                              timestamp            TIMESTAMP,
                                              city                 VARCHAR(100),
                                              raw_text             TEXT,
                                              raw_text_normalized  TEXT
);

-- ۲.۳ جدول کامنت‌ها و تحلیل تحلیلی-متنی (Comments_ABSA)
DROP TABLE IF EXISTS ecommerce.comment_embedding CASCADE;
CREATE TABLE ecommerce.comment_embedding (
                                             id                   BIGINT PRIMARY KEY,
                                             title                TEXT,
                                             body                 TEXT,
                                             created_at           TIMESTAMP,
                                             rate                 DOUBLE PRECISION,
                                             recommendation_status VARCHAR(100),
                                             is_buyer             BOOLEAN,
                                             product_id           BIGINT,
                                             advantages           TEXT,
                                             disadvantages        TEXT,
                                             likes                INT,
                                             dislikes             INT,
                                             seller_title         VARCHAR(255),
                                             seller_code          VARCHAR(100),
                                             true_to_size_rate    VARCHAR(100),
                                             raw_text             TEXT,
                                             raw_text_normalized  TEXT,
                                             predicted_sentiment  VARCHAR(50),
                                             predicted_aspects    TEXT
);

-- ۲.۴ جدول شاخص‌های رفتاری محصولات (Feature_KPI_output)
DROP TABLE IF EXISTS ecommerce.product_behavior CASCADE;
CREATE TABLE ecommerce.product_behavior (
                                            product_id           BIGINT PRIMARY KEY,
                                            view_count           BIGINT DEFAULT 0,
                                            cart_count           BIGINT DEFAULT 0,
                                            purchase_count       BIGINT DEFAULT 0,
                                            remove_count         BIGINT DEFAULT 0,
                                            conversion_rate      DOUBLE PRECISION DEFAULT 0.0,
                                            updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ۲.۵ جدول شاخص‌های جنبه‌های احساسی محصولات (ABSA_KPI)
DROP TABLE IF EXISTS ecommerce.aspect_kpi CASCADE;
CREATE TABLE ecommerce.aspect_kpi (
                                      aspect               VARCHAR(100),
                                      count                BIGINT,
                                      positive_pct         DOUBLE PRECISION,
                                      negative_pct         DOUBLE PRECISION,
                                      neutral_pct          DOUBLE PRECISION
);

-- ==============================================================================
-- 3. ساخت ایندکس‌ها برای اتصال سریع (JOIN Performance Optimization)
-- ==============================================================================
CREATE INDEX IF NOT EXISTS idx_ubl_product_id ON ecommerce.user_behavior_logs(product_id);
CREATE INDEX IF NOT EXISTS idx_ubl_user_id ON ecommerce.user_behavior_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_ubl_event_type ON ecommerce.user_behavior_logs(event_type);

CREATE INDEX IF NOT EXISTS idx_comments_product_id ON ecommerce.comment_embedding(product_id);
-- ساخت ایندکس برای افزایش سرعت کوئری‌های کامنت
CREATE INDEX IF NOT EXISTS idx_comments_prod_id ON ecommerce.comment_embedding(product_id);
-- ==============================================================================
-- 4. ساخت MATERIALIZED VIEWها برای محاسبه KPIها و تحلیل داده‌ها
-- ==============================================================================

-- ۴.۱ تحلیل فانل تبدیل محصولات (مشاهده -> افزودن به سبد -> خرید)
DROP MATERIALIZED VIEW IF EXISTS ecommerce.mv_kpi_product_best_funnel CASCADE;
CREATE MATERIALIZED VIEW ecommerce.mv_kpi_product_best_funnel AS
SELECT
    p.id AS product_id,
    p.title_fa,
    p.category1,
    COUNT(CASE WHEN l.event_type = 'page_view' THEN 1 END) AS views_count,
    COUNT(CASE WHEN l.event_type = 'add_to_cart' THEN 1 END) AS cart_add_count,
    COUNT(CASE WHEN l.event_type = 'purchase' THEN 1 END) AS purchase_count,
    ROUND(
            (COUNT(CASE WHEN l.event_type = 'purchase' THEN 1 END)::DECIMAL /
             NULLIF(COUNT(CASE WHEN l.event_type = 'page_view' THEN 1 END), 0)) * 100, 2
    ) AS conversion_rate_pct
FROM ecommerce.product_master p
         LEFT JOIN ecommerce.user_behavior_logs l ON p.id = l.product_id
GROUP BY p.id, p.title_fa, p.category1;

-- ۴.۲ بیشترین درآمد زایی محصولات
DROP MATERIALIZED VIEW IF EXISTS ecommerce.mv_kpi_product_top_revenue CASCADE;
CREATE MATERIALIZED VIEW ecommerce.mv_kpi_product_top_revenue AS
SELECT
    p.id AS product_id,
    p.title_fa,
    p.price,
    COUNT(l.user_id) AS total_purchases,
    (p.price * COUNT(l.user_id)) AS estimated_total_revenue
FROM ecommerce.product_master p
         JOIN ecommerce.user_behavior_logs l ON p.id = l.product_id
WHERE l.event_type = 'purchase'
GROUP BY p.id, p.title_fa, p.price
ORDER BY estimated_total_revenue DESC;

-- ۴.۳ بیشترین محصولات حذف شده از سبد خرید
DROP MATERIALIZED VIEW IF EXISTS ecommerce.mv_kpi_product_high_remove CASCADE;
CREATE MATERIALIZED VIEW ecommerce.mv_kpi_product_high_remove AS
SELECT
    p.id AS product_id,
    p.title_fa,
    COUNT(l.user_id) AS remove_from_cart_count
FROM ecommerce.product_master p
         JOIN ecommerce.user_behavior_logs l ON p.id = l.product_id
WHERE l.event_type = 'remove_from_cart'
GROUP BY p.id, p.title_fa
ORDER BY remove_from_cart_count DESC;

-- ۴.۴ فعال‌ترین کاربران (Heavy Users)
DROP MATERIALIZED VIEW IF EXISTS ecommerce.mv_kpi_user_heavy_users CASCADE;
CREATE MATERIALIZED VIEW ecommerce.mv_kpi_user_heavy_users AS
SELECT
    l.user_id,
    COUNT(DISTINCT l.session_id) AS total_sessions,
    COUNT(l.product_id) AS total_interactions,
    COUNT(CASE WHEN l.event_type = 'purchase' THEN 1 END) AS total_purchases
FROM ecommerce.user_behavior_logs l
GROUP BY l.user_id
ORDER BY total_interactions DESC;

-- ۴.۵ کاربران با بالاترین نرخ تبدیل خرید
DROP MATERIALIZED VIEW IF EXISTS ecommerce.mv_kpi_user_high_conversion CASCADE;
CREATE MATERIALIZED VIEW ecommerce.mv_kpi_user_high_conversion AS
SELECT
    l.user_id,
    COUNT(CASE WHEN l.event_type = 'page_view' THEN 1 END) AS views,
    COUNT(CASE WHEN l.event_type = 'purchase' THEN 1 END) AS purchases,
    ROUND(
            (COUNT(CASE WHEN l.event_type = 'purchase' THEN 1 END)::DECIMAL /
             NULLIF(COUNT(CASE WHEN l.event_type = 'page_view' THEN 1 END), 0)) * 100, 2
    ) AS user_conversion_rate
FROM ecommerce.user_behavior_logs l
GROUP BY l.user_id
HAVING COUNT(CASE WHEN l.event_type = 'page_view' THEN 1 END) > 5
ORDER BY user_conversion_rate DESC;

-- ساخت ایندکس‌های منحصر‌به‌فرد روی Materialized Viewها جهت تسریع در عملیات REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_funnel ON ecommerce.mv_kpi_product_best_funnel(product_id);
CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_revenue ON ecommerce.mv_kpi_product_top_revenue(product_id);
CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_user_heavy ON ecommerce.mv_kpi_user_heavy_users(user_id);

-- *******************************************************************
SELECT 'product_master' AS table_name, COUNT(*) AS total_rows FROM ecommerce.product_master
UNION ALL
SELECT 'user_behavior_logs', COUNT(*) FROM ecommerce.user_behavior_logs
UNION ALL
SELECT 'comment_embedding', COUNT(*) FROM ecommerce.comment_embedding
UNION ALL
SELECT 'aspect_kpi', COUNT(*) FROM ecommerce.aspect_kpi
UNION ALL
SELECT 'product_behavior', COUNT(*) FROM ecommerce.product_behavior;



-- نباید هیچ سطری با product_id خالی برگردد (خروجی باید 0 باشد)
SELECT COUNT(*) AS null_product_count
FROM ecommerce.product_behavior
WHERE product_id IS NULL;

-- بررسی ۵ سطر اول کامنت‌ها و ابعاد امبدینگ‌ها
SELECT
    id,
    product_id,
    raw_text,
    predicted_sentiment,
    predicted_aspects
FROM ecommerce.comment_embedding
LIMIT 5;


-- نمایش ۱۰ محصولی که بیشترین کامنت ثبت‌شده را دارند
SELECT
    p.id AS product_id,
    p.title_fa,
    COUNT(c.id) AS total_comments
FROM ecommerce.product_master p
         JOIN ecommerce.comment_embedding c ON p.id = c.product_id
GROUP BY p.id, p.title_fa
ORDER BY total_comments DESC
LIMIT 10;