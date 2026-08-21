-- ============================================================
-- 0. SCHEMA SETUP
-- ساخت محیط ایزوله و حرفه‌ای برای لایه هوش تجاری (BI) و چت‌بات
-- ============================================================
CREATE SCHEMA IF NOT EXISTS kpi;

-- ============================================================
-- 1. GLOBAL EXECUTIVE FUNNEL (دیدگاه کلان برای مدیر)
-- پاسخ به سوالات آماری: نرخ تبدیل کل سایت چقدر است؟
-- ============================================================
CREATE OR REPLACE VIEW kpi.global_funnel AS
SELECT
    SUM(total_views) AS total_views,
    SUM(total_cart_adds) AS total_carts,
    SUM(total_purchases) AS total_purchases,
    SUM(total_removes) AS total_removes,
    
    -- محاسبه دقیق نرخ‌های تبدیل با جلوگیری از خطای تقسیم بر صفر
    ROUND((SUM(total_cart_adds)::NUMERIC / NULLIF(SUM(total_views), 0)) * 100, 2) AS view_to_cart_pct,
    ROUND((SUM(total_purchases)::NUMERIC / NULLIF(SUM(total_cart_adds), 0)) * 100, 2) AS cart_to_purchase_pct,
    ROUND((SUM(total_purchases)::NUMERIC / NULLIF(SUM(total_views), 0)) * 100, 2) AS overall_conversion_pct,
    ROUND((SUM(total_removes)::NUMERIC / NULLIF(SUM(total_cart_adds), 0)) * 100, 2) AS cart_abandonment_pct
FROM analytics.feature_time;


-- ============================================================
-- 2. PRODUCT 360 & ACTIONABLE INSIGHTS (مغز متفکر ایجنت)
-- ترکیب رفتار کاربر + فروش + تحلیل احساسات (ABSA)
-- ============================================================
CREATE OR REPLACE VIEW kpi.product_360 AS
SELECT
    p.product_id,
    rp.title_fa,
    rp.price,
    
    -- متریک‌های فروش و رفتار
    p.total_views,
    p.total_purchases,
    (p.total_purchases * rp.price) AS total_revenue,
    ROUND((p.total_purchases::NUMERIC / NULLIF(p.total_views, 0)) * 100, 2) AS conversion_rate,
    
    -- متریک‌های احساسات (از مدل ABSA)
    COALESCE(ps.comment_count, 0) AS comment_count,
    ROUND(ps.avg_rate::NUMERIC, 2) AS star_rating,
    ROUND((ps.positive_aspect_ratio * 100)::NUMERIC, 2) AS positive_sentiment_pct,
    ROUND(((ps.positive_aspect_ratio - ps.negative_aspect_ratio) * 100)::NUMERIC, 2) AS sentiment_score,

    -- برچسب‌گذاری هوشمند مدیریتی (Actionable Tags) برای تصمیم‌گیری ایجنت
    CASE 
        WHEN (p.total_purchases * rp.price) > 50000000 AND ps.positive_aspect_ratio > 0.7 THEN 'Hero Product (قهرمان)'
        WHEN p.total_views > 1000 AND p.total_purchases < 5 THEN 'High Traffic, Low Conversion (نیازمند بررسی قیمت)'
        WHEN (ps.negative_aspect_ratio > 0.5) AND p.total_purchases > 10 THEN 'High Risk (فروش بالا اما به شدت ناراضی)'
        WHEN ps.comment_count = 0 THEN 'Needs Reviews (نیازمند کمپین ثبت نظر)'
        ELSE 'Normal'
    END AS managerial_action_tag

FROM analytics.feature_product p
JOIN public.products rp ON p.product_id = rp.id
LEFT JOIN analytics.feature_product_sentiment ps ON p.product_id = ps.product_id;


-- ============================================================
-- 3. USER RFM & SEGMENTATION (بخش‌بندی مشتریان)
-- ============================================================
CREATE OR REPLACE VIEW kpi.user_segments AS
SELECT
    user_id,
    active_days,
    total_views,
    total_purchases,
    total_spend,
    
    -- برچسب‌گذاری رفتار کاربر
    CASE 
        WHEN total_purchases >= 5 AND total_spend > 10000000 THEN 'VIP Customer'
        WHEN total_purchases > 1 AND total_purchases < 5 THEN 'Returning Customer'
        WHEN total_purchases = 1 THEN 'One-Time Buyer'
        WHEN total_purchases = 0 AND total_views > 20 THEN 'Window Shopper (فقط بازدیدکننده)'
        ELSE 'Low Engagement'
    END AS user_segment,
    
    -- نسبت بازدید به خرید
    ROUND((total_purchases::NUMERIC / NULLIF(total_views, 0)) * 100, 2) AS user_conversion_pct

FROM analytics.feature_user;


-- ============================================================
-- 4. BRAND & CATEGORY DIAGNOSTICS (عملکرد برندها)
-- ============================================================
CREATE OR REPLACE VIEW kpi.brand_diagnostics AS
SELECT
    b.brand_id,
    rb.name AS brand_name,
    b.total_views,
    b.total_purchases,
    
    -- وضعیت احساسات برند
    bs.total_comments,
    ROUND(bs.avg_comment_rating::NUMERIC, 2) AS avg_rating,
    ROUND(((bs.positive_aspect_mentions::NUMERIC - bs.negative_aspect_mentions::NUMERIC) / NULLIF(bs.total_aspect_mentions, 0)) * 100, 2) AS brand_sentiment_score

FROM analytics.feature_brand b
JOIN public.brands rb ON b.brand_id = rb.brand_id
LEFT JOIN analytics.feature_brand_sentiment bs ON b.brand_id = bs.brand_id;


-- ============================================================
-- 5. ROOT CAUSE ANALYSIS - ASPECTS (علت‌یابی ریشه‌ای)
-- ============================================================
CREATE OR REPLACE VIEW kpi.aspect_diagnostics AS
SELECT
    term AS aspect_name,
    total_mentions,
    positive_mentions,
    negative_mentions,
    
    -- محاسبه میزان بحرانی بودن یک ویژگی
    ROUND((negative_mentions::NUMERIC / NULLIF(total_mentions, 0)) * 100, 2) AS negative_impact_pct,
    
    CASE
        WHEN (negative_mentions::NUMERIC / NULLIF(total_mentions, 0)) > 0.6 THEN 'Critical Weakness (نقطه ضعف بحرانی)'
        WHEN (positive_mentions::NUMERIC / NULLIF(total_mentions, 0)) > 0.7 THEN 'Key Strength (نقطه قوت کلیدی)'
        ELSE 'Neutral'
    END AS aspect_status

FROM analytics.feature_aspect
WHERE total_mentions > 5 -- حذف نویزها
ORDER BY total_mentions DESC;