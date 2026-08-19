-- ============================================================
-- 0. SCHEMA SETUP
-- ساخت محیط ایزوله برای جداول داشبورد و KPI
-- ============================================================
CREATE SCHEMA IF NOT EXISTS kpi;

DROP TABLE IF EXISTS kpi.global_funnel CASCADE;
DROP TABLE IF EXISTS kpi.product_performance CASCADE;
DROP TABLE IF EXISTS kpi.user_segments CASCADE;
DROP TABLE IF EXISTS kpi.category_performance CASCADE;
DROP TABLE IF EXISTS kpi.time_trends CASCADE;
DROP TABLE IF EXISTS kpi.product_sentiment CASCADE;

-- ============================================================
-- 1. FUNNEL KPIs (جایگزین تابع funnel_kpis در پایتون)
-- ============================================================
CREATE TABLE kpi.global_funnel AS
SELECT
    SUM(total_views) AS total_views,
    SUM(total_cart_adds) AS total_carts,
    SUM(total_purchases) AS total_purchases,
    SUM(total_removes) AS total_removes,
    
    -- محاسبه نرخ‌های تبدیل قیف فروش
    SUM(total_cart_adds)::FLOAT / NULLIF(SUM(total_views), 0) AS view_to_cart_rate,
    SUM(total_purchases)::FLOAT / NULLIF(SUM(total_cart_adds), 0) AS cart_to_purchase_rate,
    SUM(total_purchases)::FLOAT / NULLIF(SUM(total_views), 0) AS overall_conversion,
    
    -- محاسبه نرخ ترک سبد و حذف
    (SUM(total_cart_adds) - SUM(total_purchases))::FLOAT / NULLIF(SUM(total_cart_adds), 0) AS cart_abandon_rate,
    SUM(total_removes)::FLOAT / NULLIF(SUM(total_cart_adds), 0) AS remove_rate,
    SUM(total_removes)::FLOAT / (NULLIF(SUM(total_purchases), 0) + 1) AS remove_to_purchase_ratio
FROM analytics.feature_time; -- استفاده از جدول تجمیع‌شده زمانی برای سرعت بیشتر

-- ============================================================
-- 2. PRODUCT KPIs (جایگزین تابع product_kpis در پایتون)
-- شامل بهترین محصولات، محصولات با بیشترین درآمد و بیشترین حذفی
-- ============================================================
CREATE TABLE kpi.product_performance AS
SELECT
    p.product_id,
    rp.title_fa,
    p.total_views,
    p.total_cart_adds,
    p.total_purchases,
    p.total_removes,
    
    -- Funnel Rates (برای پیدا کردن best_funnel_products و worst_cart_products)
    p.total_cart_adds::FLOAT / NULLIF(p.total_views, 0) AS view_to_cart_rate,
    p.total_purchases::FLOAT / NULLIF(p.total_cart_adds, 0) AS cart_to_purchase_rate,
    (p.total_cart_adds - p.total_purchases)::FLOAT / NULLIF(p.total_cart_adds, 0) AS cart_abandon_rate,
    p.total_removes::FLOAT / NULLIF(p.total_cart_adds, 0) AS remove_rate,
    
    -- محاسبه درآمد (برای top_revenue_products)
    (p.total_purchases * rp.price) AS total_revenue
FROM analytics.feature_product p[cite: 3]
JOIN public.products rp ON p.product_id = rp.id;

-- ============================================================
-- 3. USER SEGMENTATION & RETENTION (جایگزین user_segmentation و retention_kpis)
-- ============================================================
CREATE TABLE kpi.user_segments AS
SELECT
    user_id,
    total_views,
    total_purchases,
    total_removes,
    total_cart_adds,
    
    -- نرخ تبدیل کاربر
    total_purchases::FLOAT / NULLIF(total_views, 0) AS conversion_rate,
    
    -- برچسب‌گذاری (Tagging) کاربران بر اساس منطق پایتون قبلی
    CASE WHEN total_purchases > 10 THEN true ELSE false END AS is_heavy_user,
    CASE WHEN total_views < 5 THEN true ELSE false END AS is_low_engagement_user,
    CASE WHEN (total_removes::FLOAT / NULLIF(total_cart_adds, 0)) > 0.5 THEN true ELSE false END AS is_high_remove_user,
    
    -- وضعیت بازگشت کاربر (Retention)
    CASE WHEN total_sessions > 1 THEN 'Returning' ELSE 'One-Time' END AS retention_status
FROM analytics.feature_user;[cite: 3]

-- ============================================================
-- 4. CATEGORY KPIs (جایگزین تابع category_kpis در پایتون)
-- ============================================================
CREATE TABLE kpi.category_performance AS
SELECT
    c.category_id,
    rc.category1,
    rc.category2,
    c.total_views,
    c.total_cart_adds,
    c.total_purchases,
    -- محاسبه تقریبی درآمد هر دسته
    (c.total_purchases * c.avg_product_price) AS estimated_revenue
FROM analytics.feature_category c[cite: 3]
JOIN public.categories rc ON c.category_id = rc.category_id;

-- ============================================================
-- 5. TIME TRENDS (جایگزین تابع time_kpis در پایتون)
-- ============================================================
CREATE TABLE kpi.time_trends AS
SELECT
    hour,
    iso_weekday AS weekday,
    total_events AS hourly_activity,
    total_purchases AS purchase_by_hour
FROM analytics.feature_time;[cite: 3, 5]

-- ============================================================
-- 6. ABSA: PRODUCT SENTIMENT (فیچر جدید بر اساس Aspectها)
-- محاسبه درصد احساسات و ترکیب با امتیازدهی (Sentiment Score)
-- ============================================================
CREATE TABLE kpi.product_sentiment AS
SELECT 
    ps.product_id,
    ps.comment_count,
    ROUND(ps.avg_rate::NUMERIC, 2) AS avg_rate,
    ps.total_aspect_mentions,
    
    -- تبدیل نسبت‌ها به درصد (0 تا 100) برای نمایش در داشبورد
    ROUND((ps.positive_aspect_ratio * 100)::NUMERIC, 2) AS positive_pct,
    ROUND((ps.negative_aspect_ratio * 100)::NUMERIC, 2) AS negative_pct,
    ROUND((ps.neutral_aspect_ratio * 100)::NUMERIC, 2) AS neutral_pct,
    
    -- محاسبه یکپارچه نمره احساسات (مثبت منهای منفی)
    ROUND(((ps.positive_aspect_ratio - ps.negative_aspect_ratio) * 100)::NUMERIC, 2) AS sentiment_score
FROM analytics.feature_product_sentiment ps;[cite: 3]