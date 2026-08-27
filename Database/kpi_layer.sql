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




-- جدید برای کلاسترینگ

CREATE OR REPLACE VIEW kpi.rfm_segments AS
WITH recency_data AS (
    SELECT
        s.user_id,
        -- محاسبه تعداد روزهای گذشته از آخرین فعالیت کاربر
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(l.timestamp)))/86400 AS recency_days
    FROM public.sessions s
    JOIN public.user_behavior_logs l ON s.session_id = l.session_id
    GROUP BY s.user_id
),
rfm_raw AS (
    SELECT
        fu.user_id,
        COALESCE(rd.recency_days, 9999) AS recency_days,
        fu.total_purchases AS frequency,
        fu.total_spend AS monetary
    FROM analytics.feature_user fu
    LEFT JOIN recency_data rd ON fu.user_id = rd.user_id
    WHERE fu.total_purchases > 0 -- RFM معمولاً برای خریداران محاسبه می‌شود
),
rfm_scoring AS (
    SELECT
        user_id,
        recency_days,
        frequency,
        monetary,
        -- تخصیص امتیاز ۱ تا ۵ (۵ بهترین است)
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score, 
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM rfm_raw
)
SELECT
    user_id,
    recency_days,
    frequency,
    monetary,
    (r_score::text || f_score::text || m_score::text) AS rfm_code,
    
	CASE
	        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'vip'
	        WHEN r_score >= 4 AND f_score <= 2 THEN 'promising'
	        WHEN r_score <= 2 AND f_score >= 4 THEN 'at_risk'
	        WHEN r_score <= 2 AND f_score <= 2 THEN 'lost'
	        ELSE 'regular'
	    END AS rfm_label
FROM rfm_scoring;




CREATE TABLE IF NOT EXISTS kpi.ml_user_clusters (
    user_id BIGINT PRIMARY KEY REFERENCES public.users(user_id) ON DELETE CASCADE,
    cluster_id INT NOT NULL,
    cluster_name VARCHAR(100) NOT NULL
);


# این ها kpi های جدیدمون هستن برای داشبورد



-- ============================================================
-- KPI VIEWS - DYNAMIC 30 DAYS BASED ON LATEST DATA
-- ============================================================

CREATE SCHEMA IF NOT EXISTS kpi;


-- ============================================================
-- 1. DAILY FUNNEL
-- ============================================================
-- این View برای نمودارهای زمانی داشبورد است.
-- هر ردیف = یک روز
--
-- بازه:
-- آخرین 30 روز موجود در user_behavior_logs
--
-- مثال فعلی:
-- 2023-01-30 → 2023-03-01
-- ============================================================

CREATE OR REPLACE VIEW kpi.daily_funnel AS

WITH latest_data AS (
    SELECT
        MAX(timestamp) AS max_timestamp
    FROM public.user_behavior_logs
),

daily_data AS (
    SELECT
        DATE(l.timestamp) AS event_date,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'view'
        ) AS total_views,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'add_to_cart'
        ) AS total_carts,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'purchase'
        ) AS total_purchases,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'remove_from_cart'
        ) AS total_removes

    FROM public.user_behavior_logs l
    CROSS JOIN latest_data ld

    WHERE l.timestamp >= (
        ld.max_timestamp - INTERVAL '30 days'
    )
    AND l.timestamp <= ld.max_timestamp

    GROUP BY DATE(l.timestamp)
)

SELECT
    event_date,

    total_views,
    total_carts,
    total_purchases,
    total_removes,

    -- View → Cart
    ROUND(
        (
            total_carts::NUMERIC
            / NULLIF(total_views, 0)
        ) * 100,
        2
    ) AS view_to_cart_pct,

    -- Cart → Purchase
    ROUND(
        (
            total_purchases::NUMERIC
            / NULLIF(total_carts, 0)
        ) * 100,
        2
    ) AS cart_to_purchase_pct,

    -- View → Purchase
    ROUND(
        (
            total_purchases::NUMERIC
            / NULLIF(total_views, 0)
        ) * 100,
        2
    ) AS overall_conversion_pct,

    -- Cart Abandonment
    ROUND(
        (
            total_removes::NUMERIC
            / NULLIF(total_carts, 0)
        ) * 100,
        2
    ) AS cart_abandonment_pct

FROM daily_data

ORDER BY event_date;


-- ============================================================
-- 2. TOP PRODUCTS - LAST 30 DAYS
-- ============================================================
-- هر ردیف = یک محصول
--
-- فقط رفتارهای 30 روز اخیر موجود در دیتابیس
--
-- مرتب شده بر اساس درآمد
-- ============================================================

CREATE OR REPLACE VIEW kpi.top_products_30d AS

WITH latest_data AS (
    SELECT
        MAX(timestamp) AS max_timestamp
    FROM public.user_behavior_logs
),

product_stats AS (
    SELECT

        p.id AS product_id,
        p.title_fa AS product_name,
        p.price,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'view'
        ) AS total_views_30d,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'add_to_cart'
        ) AS total_carts_30d,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'purchase'
        ) AS total_purchases_30d,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'remove_from_cart'
        ) AS total_removes_30d

    FROM public.user_behavior_logs l

    INNER JOIN public.products p
        ON l.product_id = p.id

    CROSS JOIN latest_data ld

    WHERE l.timestamp >= (
        ld.max_timestamp - INTERVAL '30 days'
    )
    AND l.timestamp <= ld.max_timestamp

    GROUP BY
        p.id,
        p.title_fa,
        p.price
)

SELECT

    product_id,
    product_name,
    price,

    total_views_30d,
    total_carts_30d,
    total_purchases_30d,
    total_removes_30d,

    -- Revenue
    (
        total_purchases_30d * COALESCE(price, 0)
    ) AS total_revenue_30d,

    -- Conversion Rate
    ROUND(
        (
            total_purchases_30d::NUMERIC
            / NULLIF(total_views_30d, 0)
        ) * 100,
        2
    ) AS conversion_rate_30d

FROM product_stats

WHERE total_purchases_30d > 0

ORDER BY
    total_revenue_30d DESC;


-- ============================================================
-- 3. TOP BRANDS - LAST 30 DAYS
-- ============================================================
-- هر ردیف = یک Brand
--
-- داده‌ها مستقیماً از user_behavior_logs
-- + products
-- + brands
-- گرفته می‌شوند.
-- ============================================================

CREATE OR REPLACE VIEW kpi.top_brands_30d AS

WITH latest_data AS (
    SELECT
        MAX(timestamp) AS max_timestamp
    FROM public.user_behavior_logs
),

brand_stats AS (
    SELECT

        b.brand_id,
        b.name AS brand_name,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'view'
        ) AS total_views_30d,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'add_to_cart'
        ) AS total_carts_30d,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'purchase'
        ) AS total_purchases_30d,

        COUNT(*) FILTER (
            WHERE LOWER(l.event_type) = 'remove_from_cart'
        ) AS total_removes_30d,

        SUM(
            CASE
                WHEN LOWER(l.event_type) = 'purchase'
                THEN COALESCE(p.price, 0)
                ELSE 0
            END
        ) AS total_revenue_30d

    FROM public.user_behavior_logs l

    INNER JOIN public.products p
        ON l.product_id = p.id

    INNER JOIN public.brands b
        ON p.brand_id = b.brand_id

    CROSS JOIN latest_data ld

    WHERE l.timestamp >= (
        ld.max_timestamp - INTERVAL '30 days'
    )
    AND l.timestamp <= ld.max_timestamp

    GROUP BY
        b.brand_id,
        b.name
)

SELECT

    brand_id,
    brand_name,

    total_views_30d,
    total_carts_30d,
    total_purchases_30d,
    total_removes_30d,
    total_revenue_30d,

    -- Conversion Rate
    ROUND(
        (
            total_purchases_30d::NUMERIC
            / NULLIF(total_views_30d, 0)
        ) * 100,
        2
    ) AS conversion_rate_30d

FROM brand_stats

WHERE total_purchases_30d > 0

ORDER BY
    total_revenue_30d DESC;