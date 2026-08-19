-- ============================================================
-- 01_feature_engineering.sql
--
-- NORMALIZED ANALYTICS / FEATURE ENGINEERING LAYER
--
-- SOURCE TABLES:
--     public.cities
--     public.users
--     public.sessions
--     public.brands
--     public.categories
--     public.sellers
--     public.products
--     public.user_behavior_logs
--     public.comments
--     public.comment_aspects
--
-- DESIGN PRINCIPLES
--
-- 1. Raw attributes are NOT copied into analytics tables.
--
-- 2. Every feature table contains:
--      - only the key(s) required by its grain
--      - calculated / aggregated features
--
-- 3. Foreign keys already available through the source schema
--    are NOT duplicated unnecessarily.
--
-- 4. Entity attributes such as:
--      product title
--      brand name
--      category name
--      seller title
--      city name
--      comment text
--      embeddings
--    remain in their original source tables.
--
-- 5. Feature tables are connected to source tables through
--    their grain keys.
--
-- ============================================================


-- ============================================================
-- 0. ANALYTICS SCHEMA
-- ============================================================

CREATE SCHEMA IF NOT EXISTS analytics;


-- ============================================================
-- 1. CLEAN OLD FEATURE TABLES
-- ============================================================

DROP TABLE IF EXISTS analytics.feature_product_aspect CASCADE;
DROP TABLE IF EXISTS analytics.feature_product_sentiment CASCADE;

DROP TABLE IF EXISTS analytics.feature_brand_sentiment CASCADE;
DROP TABLE IF EXISTS analytics.feature_category_sentiment CASCADE;

DROP TABLE IF EXISTS analytics.feature_aspect CASCADE;

DROP TABLE IF EXISTS analytics.feature_user_category CASCADE;
DROP TABLE IF EXISTS analytics.feature_user_product CASCADE;

DROP TABLE IF EXISTS analytics.feature_behavior CASCADE;
DROP TABLE IF EXISTS analytics.feature_time CASCADE;

DROP TABLE IF EXISTS analytics.feature_brand CASCADE;
DROP TABLE IF EXISTS analytics.feature_category CASCADE;
DROP TABLE IF EXISTS analytics.feature_city CASCADE;
DROP TABLE IF EXISTS analytics.feature_product CASCADE;
DROP TABLE IF EXISTS analytics.feature_user CASCADE;


-- ============================================================
-- 2. EVENT / BEHAVIOR FEATURES
--
-- Grain:
--     ONE ROW PER BEHAVIOR LOG
--
-- Key:
--     log_id
--
-- IMPORTANT:
--     No user_id
--     No session_id
--     No product_id
--     No city_id
--
-- These can be obtained through:
--
-- feature_behavior.log_id
--       -> user_behavior_logs.log_id
--       -> sessions
--       -> users / products / cities
--
-- ============================================================

CREATE TABLE analytics.feature_behavior AS

SELECT

    b.log_id,

    -- --------------------------------------------------------
    -- Temporal Features
    -- --------------------------------------------------------

    EXTRACT(
        HOUR FROM b.timestamp
    )::SMALLINT AS hour,

    EXTRACT(
        DAY FROM b.timestamp
    )::SMALLINT AS day,

    EXTRACT(
        MONTH FROM b.timestamp
    )::SMALLINT AS month,

    EXTRACT(
        ISODOW FROM b.timestamp
    )::SMALLINT AS weekday,

    -- Iran weekend:
    -- Thursday = 4
    -- Friday   = 5

    CASE
        WHEN EXTRACT(ISODOW FROM b.timestamp) IN (4,5)
        THEN 1
        ELSE 0
    END AS is_weekend,


    -- --------------------------------------------------------
    -- Event Flags
    -- --------------------------------------------------------

    CASE
        WHEN b.event_type = 'view'
        THEN 1
        ELSE 0
    END AS is_view,

    CASE
        WHEN b.event_type = 'add_to_cart'
        THEN 1
        ELSE 0
    END AS is_cart,

    CASE
        WHEN b.event_type = 'remove_from_cart'
        THEN 1
        ELSE 0
    END AS is_remove,

    CASE
        WHEN b.event_type = 'purchase'
        THEN 1
        ELSE 0
    END AS is_purchase

FROM public.user_behavior_logs b;


CREATE UNIQUE INDEX idx_feature_behavior_log
ON analytics.feature_behavior(log_id);


-- ============================================================
-- 3. USER FEATURES
--
-- Grain:
--     ONE ROW PER USER
--
-- Key:
--     user_id
--
-- No session_id
-- No product_id
-- No city_id
--
-- ============================================================

CREATE TABLE analytics.feature_user AS

WITH base AS (

    SELECT

        s.user_id,

        b.log_id,
        b.session_id,
        b.product_id,
        s.city_id,

        l.timestamp,

        fb.hour,
        fb.weekday,
        fb.is_weekend,

        fb.is_view,
        fb.is_cart,
        fb.is_remove,
        fb.is_purchase

    FROM analytics.feature_behavior fb

    INNER JOIN public.user_behavior_logs l
        ON fb.log_id = l.log_id

    INNER JOIN public.sessions s
        ON l.session_id = s.session_id
),


-- ============================================================
-- USER EVENT STATISTICS
-- ============================================================

event_stats AS (

    SELECT

        user_id,

        COUNT(*) AS total_events,

        COUNT(DISTINCT session_id)
            AS total_sessions,

        COUNT(DISTINCT DATE(timestamp))
            AS active_days,

        SUM(is_view)
            AS total_views,

        SUM(is_cart)
            AS total_cart_adds,

        SUM(is_remove)
            AS total_removes,

        SUM(is_purchase)
            AS total_purchases,

        COUNT(DISTINCT product_id)
            FILTER (
                WHERE is_view = 1
            ) AS unique_products_viewed,

        COUNT(DISTINCT product_id)
            FILTER (
                WHERE is_purchase = 1
            ) AS unique_products_purchased,

        COUNT(DISTINCT city_id)
            AS cities_visited,

        AVG(is_weekend)::DOUBLE PRECISION
            AS weekend_activity_ratio,

        AVG(
            CASE
                WHEN hour >= 6
                 AND hour < 12
                THEN 1.0
                ELSE 0.0
            END
        ) AS morning_activity_ratio,

        AVG(
            CASE
                WHEN hour >= 12
                 AND hour < 18
                THEN 1.0
                ELSE 0.0
            END
        ) AS afternoon_activity_ratio,

        AVG(
            CASE
                WHEN hour >= 18
                 AND hour < 24
                THEN 1.0
                ELSE 0.0
            END
        ) AS evening_activity_ratio,

        AVG(
            CASE
                WHEN hour < 6
                THEN 1.0
                ELSE 0.0
            END
        ) AS night_activity_ratio

    FROM base

    GROUP BY user_id
),


-- ============================================================
-- USER SESSION STATISTICS
-- ============================================================

session_stats AS (

    SELECT

        user_id,

        AVG(session_events)::DOUBLE PRECISION
            AS avg_session_events,

        MAX(session_events)
            AS max_session_events,

        AVG(session_duration_minutes)::DOUBLE PRECISION
            AS avg_session_duration_minutes,

        MAX(session_duration_minutes)::DOUBLE PRECISION
            AS max_session_duration_minutes

    FROM (

        SELECT

            user_id,
            session_id,

            COUNT(*) AS session_events,

            EXTRACT(
                EPOCH FROM (
                    MAX(timestamp)
                    -
                    MIN(timestamp)
                )
            ) / 60.0
                AS session_duration_minutes

        FROM base

        GROUP BY
            user_id,
            session_id
    ) x

    GROUP BY user_id
),


-- ============================================================
-- USER PURCHASE STATISTICS
-- ============================================================

purchase_stats AS (

    SELECT

        b.user_id,

        COUNT(*) AS purchase_frequency,

        COUNT(DISTINCT DATE(b.timestamp))
            AS purchase_days,

        SUM(p.price)::DOUBLE PRECISION
            AS total_spend,

        AVG(p.price)::DOUBLE PRECISION
            AS avg_purchase_value,

        MIN(p.price)
            AS min_purchase_price,

        MAX(p.price)
            AS max_purchase_price,

        COUNT(DISTINCT p.brand_id)
            AS brand_diversity,

        COUNT(DISTINCT p.category_id)
            AS category_diversity

    FROM base b

    INNER JOIN public.products p
        ON b.product_id = p.id

    WHERE b.is_purchase = 1

    GROUP BY b.user_id
),


-- ============================================================
-- USER TIME PREFERENCE
-- ============================================================

time_pref AS (

    SELECT

        user_id,

        MODE() WITHIN GROUP (
            ORDER BY hour
        ) AS preferred_hour,

        MODE() WITHIN GROUP (
            ORDER BY weekday
        ) AS preferred_weekday

    FROM base

    GROUP BY user_id
)


-- ============================================================
-- FINAL USER FEATURE TABLE
-- ============================================================

SELECT

    e.user_id,

    e.total_events,
    e.total_sessions,
    e.active_days,

    e.total_views,
    e.total_cart_adds,
    e.total_removes,
    e.total_purchases,

    s.avg_session_events,
    s.max_session_events,

    s.avg_session_duration_minutes,
    s.max_session_duration_minutes,

    e.weekend_activity_ratio,
    e.morning_activity_ratio,
    e.afternoon_activity_ratio,
    e.evening_activity_ratio,
    e.night_activity_ratio,

    t.preferred_hour,
    t.preferred_weekday,

    e.unique_products_viewed,
    e.unique_products_purchased,

    e.cities_visited,

    COALESCE(p.total_spend, 0)
        AS total_spend,

    COALESCE(p.avg_purchase_value, 0)
        AS avg_purchase_value,

    COALESCE(p.min_purchase_price, 0)
        AS min_purchase_price,

    COALESCE(p.max_purchase_price, 0)
        AS max_purchase_price,

    COALESCE(p.purchase_frequency, 0)
        AS purchase_frequency,

    COALESCE(p.purchase_days, 0)
        AS purchase_days,

    COALESCE(p.brand_diversity, 0)
        AS brand_diversity,

    COALESCE(p.category_diversity, 0)
        AS category_diversity

FROM event_stats e

LEFT JOIN session_stats s
    ON e.user_id = s.user_id

LEFT JOIN purchase_stats p
    ON e.user_id = p.user_id

LEFT JOIN time_pref t
    ON e.user_id = t.user_id;


CREATE UNIQUE INDEX idx_feature_user
ON analytics.feature_user(user_id);


-- ============================================================
-- 4. PRODUCT FEATURES
--
-- Grain:
--     ONE ROW PER PRODUCT
--
-- Key:
--     product_id
--
-- IMPORTANT:
--     brand_id
--     category_id
--     seller_id
--
-- are NOT copied into this table.
--
-- They already exist in public.products.
--
-- ============================================================

CREATE TABLE analytics.feature_product AS

WITH behavior AS (

    SELECT

        l.product_id,

        COUNT(*) AS total_events,

        SUM(fb.is_view)
            AS total_views,

        SUM(fb.is_cart)
            AS total_cart_adds,

        SUM(fb.is_remove)
            AS total_removes,

        SUM(fb.is_purchase)
            AS total_purchases,

        COUNT(DISTINCT s.user_id)
            FILTER (
                WHERE fb.is_view = 1
            ) AS unique_viewers,

        COUNT(DISTINCT s.user_id)
            FILTER (
                WHERE fb.is_cart = 1
            ) AS unique_carters,

        COUNT(DISTINCT s.user_id)
            FILTER (
                WHERE fb.is_purchase = 1
            ) AS unique_buyers,

        COUNT(DISTINCT l.session_id)
            AS total_sessions

    FROM analytics.feature_behavior fb

    INNER JOIN public.user_behavior_logs l
        ON fb.log_id = l.log_id

    INNER JOIN public.sessions s
        ON l.session_id = s.session_id

    GROUP BY l.product_id
)

SELECT

    p.id AS product_id,

    COALESCE(b.total_events, 0)
        AS total_events,

    COALESCE(b.total_views, 0)
        AS total_views,

    COALESCE(b.total_cart_adds, 0)
        AS total_cart_adds,

    COALESCE(b.total_removes, 0)
        AS total_removes,

    COALESCE(b.total_purchases, 0)
        AS total_purchases,

    COALESCE(b.unique_viewers, 0)
        AS unique_viewers,

    COALESCE(b.unique_carters, 0)
        AS unique_carters,

    COALESCE(b.unique_buyers, 0)
        AS unique_buyers,

    COALESCE(b.total_sessions, 0)
        AS total_sessions,

    GREATEST(

        (
            p.min_price_last_month - p.price
        )::DOUBLE PRECISION

        /

        NULLIF(
            p.min_price_last_month,
            0
        ),

        0

    ) AS price_drop_ratio

FROM public.products p

LEFT JOIN behavior b
    ON p.id = b.product_id;


CREATE UNIQUE INDEX idx_feature_product
ON analytics.feature_product(product_id);


-- ============================================================
-- 5. CITY FEATURES
--
-- Grain:
--     ONE ROW PER CITY
--
-- Key:
--     city_id
--
-- ============================================================

CREATE TABLE analytics.feature_city AS

SELECT

    s.city_id,

    COUNT(DISTINCT s.user_id)
        AS total_users,

    COUNT(DISTINCT l.session_id)
        AS total_sessions,

    COUNT(*)
        AS total_events,

    SUM(fb.is_view)
        AS total_views,

    SUM(fb.is_cart)
        AS total_cart_adds,

    SUM(fb.is_purchase)
        AS total_purchases,

    SUM(fb.is_remove)
        AS total_removes,

    COUNT(DISTINCT l.product_id)
        FILTER (
            WHERE fb.is_view = 1
        ) AS unique_products_viewed,

    COUNT(DISTINCT l.product_id)
        FILTER (
            WHERE fb.is_purchase = 1
        ) AS unique_products_purchased

FROM analytics.feature_behavior fb

INNER JOIN public.user_behavior_logs l
    ON fb.log_id = l.log_id

INNER JOIN public.sessions s
    ON l.session_id = s.session_id

GROUP BY s.city_id;


CREATE UNIQUE INDEX idx_feature_city
ON analytics.feature_city(city_id);


-- ============================================================
-- 6. CATEGORY FEATURES
--
-- Grain:
--     ONE ROW PER CATEGORY
--
-- Key:
--     category_id
--
-- ============================================================

CREATE TABLE analytics.feature_category AS

SELECT

    p.category_id,

    COUNT(*) AS total_events,

    SUM(fb.is_view)
        AS total_views,

    SUM(fb.is_cart)
        AS total_cart_adds,

    SUM(fb.is_purchase)
        AS total_purchases,

    SUM(fb.is_remove)
        AS total_removes,

    COUNT(DISTINCT s.user_id)
        FILTER (
            WHERE fb.is_view = 1
        ) AS unique_viewers,

    COUNT(DISTINCT s.user_id)
        FILTER (
            WHERE fb.is_purchase = 1
        ) AS unique_buyers,

    AVG(p.price)::DOUBLE PRECISION
        AS avg_product_price

FROM analytics.feature_behavior fb

INNER JOIN public.user_behavior_logs l
    ON fb.log_id = l.log_id

INNER JOIN public.sessions s
    ON l.session_id = s.session_id

INNER JOIN public.products p
    ON l.product_id = p.id

GROUP BY p.category_id;


CREATE UNIQUE INDEX idx_feature_category
ON analytics.feature_category(category_id);


-- ============================================================
-- 7. BRAND FEATURES
--
-- Grain:
--     ONE ROW PER BRAND
--
-- Key:
--     brand_id
--
-- ============================================================

CREATE TABLE analytics.feature_brand AS

SELECT

    p.brand_id,

    COUNT(*) AS total_events,

    SUM(fb.is_view)
        AS total_views,

    SUM(fb.is_cart)
        AS total_cart_adds,

    SUM(fb.is_purchase)
        AS total_purchases,

    SUM(fb.is_remove)
        AS total_removes,

    COUNT(DISTINCT s.user_id)
        FILTER (
            WHERE fb.is_view = 1
        ) AS unique_viewers,

    COUNT(DISTINCT s.user_id)
        FILTER (
            WHERE fb.is_purchase = 1
        ) AS unique_buyers

FROM analytics.feature_behavior fb

INNER JOIN public.user_behavior_logs l
    ON fb.log_id = l.log_id

INNER JOIN public.sessions s
    ON l.session_id = s.session_id

INNER JOIN public.products p
    ON l.product_id = p.id

GROUP BY p.brand_id;


CREATE UNIQUE INDEX idx_feature_brand
ON analytics.feature_brand(brand_id);


-- ============================================================
-- 8. USER × PRODUCT FEATURES
--
-- Grain:
--     ONE ROW PER USER / PRODUCT
--
-- Keys:
--     user_id
--     product_id
--
-- Both keys are necessary because they define the grain.
--
-- ============================================================

CREATE TABLE analytics.feature_user_product AS

SELECT

    s.user_id,

    l.product_id,

    COUNT(*) AS total_events,

    SUM(fb.is_view)
        AS view_count,

    SUM(fb.is_cart)
        AS cart_count,

    SUM(fb.is_remove)
        AS remove_count,

    SUM(fb.is_purchase)
        AS purchase_count,

    COUNT(DISTINCT DATE(l.timestamp))
        AS active_days,

    COUNT(DISTINCT l.session_id)
        AS session_count

FROM analytics.feature_behavior fb

INNER JOIN public.user_behavior_logs l
    ON fb.log_id = l.log_id

INNER JOIN public.sessions s
    ON l.session_id = s.session_id

GROUP BY

    s.user_id,
    l.product_id;


CREATE UNIQUE INDEX idx_feature_user_product
ON analytics.feature_user_product(
    user_id,
    product_id
);


-- ============================================================
-- 9. USER × CATEGORY FEATURES
--
-- Grain:
--     ONE ROW PER USER / CATEGORY
--
-- Keys:
--     user_id
--     category_id
--
-- ============================================================

CREATE TABLE analytics.feature_user_category AS

WITH base AS (

    SELECT

        s.user_id,

        p.category_id,

        COUNT(*) AS total_events,

        SUM(fb.is_view)
            AS view_count,

        SUM(fb.is_cart)
            AS cart_count,

        SUM(fb.is_remove)
            AS remove_count,

        SUM(fb.is_purchase)
            AS purchase_count,

        SUM(
            CASE
                WHEN fb.is_purchase = 1
                THEN p.price
                ELSE 0
            END
        )::DOUBLE PRECISION
            AS category_spend

    FROM analytics.feature_behavior fb

    INNER JOIN public.user_behavior_logs l
        ON fb.log_id = l.log_id

    INNER JOIN public.sessions s
        ON l.session_id = s.session_id

    INNER JOIN public.products p
        ON l.product_id = p.id

    GROUP BY

        s.user_id,
        p.category_id
),


totals AS (

    SELECT

        user_id,

        SUM(view_count)
            AS total_user_views,

        SUM(purchase_count)
            AS total_user_purchases,

        SUM(category_spend)
            AS total_user_spend

    FROM base

    GROUP BY user_id
)


SELECT

    b.user_id,

    b.category_id,

    b.total_events,

    b.view_count,

    b.cart_count,

    b.remove_count,

    b.purchase_count,

    b.category_spend,

    b.view_count::DOUBLE PRECISION
    /
    NULLIF(
        t.total_user_views,
        0
    ) AS view_share,

    b.purchase_count::DOUBLE PRECISION
    /
    NULLIF(
        t.total_user_purchases,
        0
    ) AS purchase_share,

    b.category_spend::DOUBLE PRECISION
    /
    NULLIF(
        t.total_user_spend,
        0
    ) AS spend_share

FROM base b

INNER JOIN totals t
    ON b.user_id = t.user_id;


CREATE UNIQUE INDEX idx_feature_user_category
ON analytics.feature_user_category(
    user_id,
    category_id
);


-- ============================================================
-- 10. PRODUCT SENTIMENT FEATURES
--
-- Grain:
--     ONE ROW PER PRODUCT
--
-- Key:
--     product_id
--
-- IMPORTANT:
-- New schema does NOT have predicted_sentiment
-- inside comments.
--
-- Sentiment now exists in comment_aspects.
--
-- Therefore:
--
-- comments
--     -> comment-level metrics
--
-- comment_aspects
--     -> aspect sentiment metrics
--
-- ============================================================

CREATE TABLE analytics.feature_product_sentiment AS

WITH comment_stats AS (

    SELECT

        product_id,

        COUNT(*) AS comment_count,

        AVG(rate)::DOUBLE PRECISION
            AS avg_rate,

        AVG(
            COALESCE(likes,0)::DOUBLE PRECISION
            /
            NULLIF(
                COALESCE(dislikes,0) + 1,
                0
            )
        ) AS avg_like_ratio,

        SUM(COALESCE(likes,0))
            AS total_likes,

        SUM(COALESCE(dislikes,0))
            AS total_dislikes

    FROM public.comments

    GROUP BY product_id
),


aspect_stats AS (

    SELECT

        c.product_id,

        COUNT(ca.aspect_id)
            AS total_aspect_mentions,

        COUNT(ca.aspect_id)
            FILTER (
                WHERE LOWER(ca.sentiment) = 'positive'
            ) AS positive_aspect_mentions,

        COUNT(ca.aspect_id)
            FILTER (
                WHERE LOWER(ca.sentiment) = 'negative'
            ) AS negative_aspect_mentions,

        COUNT(ca.aspect_id)
            FILTER (
                WHERE LOWER(ca.sentiment) = 'neutral'
            ) AS neutral_aspect_mentions,

        AVG(ca.positive_pct)
            AS avg_positive_pct,

        AVG(ca.negative_pct)
            AS avg_negative_pct,

        AVG(ca.neutral_pct)
            AS avg_neutral_pct

    FROM public.comments c

    INNER JOIN public.comment_aspects ca
        ON c.id = ca.comment_id

    GROUP BY c.product_id
)


SELECT

    p.id AS product_id,

    COALESCE(c.comment_count, 0)
        AS comment_count,

    COALESCE(c.avg_rate, 0)
        AS avg_rate,

    COALESCE(c.avg_like_ratio, 0)
        AS avg_like_ratio,

    COALESCE(c.total_likes, 0)
        AS total_likes,

    COALESCE(c.total_dislikes, 0)
        AS total_dislikes,

    COALESCE(a.total_aspect_mentions, 0)
        AS total_aspect_mentions,

    COALESCE(a.positive_aspect_mentions, 0)
        AS positive_aspect_mentions,

    COALESCE(a.negative_aspect_mentions, 0)
        AS negative_aspect_mentions,

    COALESCE(a.neutral_aspect_mentions, 0)
        AS neutral_aspect_mentions,

    COALESCE(a.avg_positive_pct, 0)
        AS avg_positive_pct,

    COALESCE(a.avg_negative_pct, 0)
        AS avg_negative_pct,

    COALESCE(a.avg_neutral_pct, 0)
        AS avg_neutral_pct,

    (
        COALESCE(a.positive_aspect_mentions,0)::DOUBLE PRECISION
        /
        NULLIF(
            a.total_aspect_mentions,
            0
        )
    ) AS positive_aspect_ratio,

    (
        COALESCE(a.negative_aspect_mentions,0)::DOUBLE PRECISION
        /
        NULLIF(
            a.total_aspect_mentions,
            0
        )
    ) AS negative_aspect_ratio,

    (
        COALESCE(a.neutral_aspect_mentions,0)::DOUBLE PRECISION
        /
        NULLIF(
            a.total_aspect_mentions,
            0
        )
    ) AS neutral_aspect_ratio

FROM public.products p

LEFT JOIN comment_stats c
    ON p.id = c.product_id

LEFT JOIN aspect_stats a
    ON p.id = a.product_id;


CREATE UNIQUE INDEX idx_feature_product_sentiment
ON analytics.feature_product_sentiment(product_id);


-- ============================================================
-- 11. PRODUCT × ASPECT FEATURES
--
-- Grain:
--     ONE ROW PER PRODUCT / ASPECT
--
-- Keys:
--     product_id
--     term
--
-- ============================================================

CREATE TABLE analytics.feature_product_aspect AS

SELECT

    c.product_id,

    ca.term,

    COUNT(*) AS total_mentions,

    COUNT(*) FILTER (
        WHERE LOWER(ca.sentiment) = 'positive'
    ) AS positive_mentions,

    COUNT(*) FILTER (
        WHERE LOWER(ca.sentiment) = 'negative'
    ) AS negative_mentions,

    COUNT(*) FILTER (
        WHERE LOWER(ca.sentiment) = 'neutral'
    ) AS neutral_mentions,

    AVG(ca.negative_pct)
        AS avg_negative_pct,

    AVG(ca.neutral_pct)
        AS avg_neutral_pct,

    AVG(ca.positive_pct)
        AS avg_positive_pct

FROM public.comments c

INNER JOIN public.comment_aspects ca
    ON c.id = ca.comment_id

WHERE ca.term IS NOT NULL

GROUP BY

    c.product_id,
    ca.term;


CREATE UNIQUE INDEX idx_feature_product_aspect
ON analytics.feature_product_aspect(
    product_id,
    term
);


-- ============================================================
-- 12. BRAND SENTIMENT FEATURES
--
-- Grain:
--     ONE ROW PER BRAND
--
-- Key:
--     brand_id
--
-- ============================================================

CREATE TABLE analytics.feature_brand_sentiment AS

SELECT

    p.brand_id,

    COUNT(DISTINCT c.id)
        AS total_comments,

    COUNT(ca.aspect_id)
        AS total_aspect_mentions,

    COUNT(ca.aspect_id)
        FILTER (
            WHERE LOWER(ca.sentiment) = 'positive'
        ) AS positive_aspect_mentions,

    COUNT(ca.aspect_id)
        FILTER (
            WHERE LOWER(ca.sentiment) = 'negative'
        ) AS negative_aspect_mentions,

    COUNT(ca.aspect_id)
        FILTER (
            WHERE LOWER(ca.sentiment) = 'neutral'
        ) AS neutral_aspect_mentions,

    AVG(c.rate)::DOUBLE PRECISION
        AS avg_comment_rating,

    SUM(COALESCE(c.likes,0))
        AS total_likes,

    SUM(COALESCE(c.dislikes,0))
        AS total_dislikes

FROM public.products p

LEFT JOIN public.comments c
    ON p.id = c.product_id

LEFT JOIN public.comment_aspects ca
    ON c.id = ca.comment_id

GROUP BY p.brand_id;


CREATE UNIQUE INDEX idx_feature_brand_sentiment
ON analytics.feature_brand_sentiment(brand_id);


-- ============================================================
-- 13. CATEGORY SENTIMENT FEATURES
--
-- Grain:
--     ONE ROW PER CATEGORY
--
-- Key:
--     category_id
--
-- ============================================================

CREATE TABLE analytics.feature_category_sentiment AS

SELECT

    p.category_id,

    COUNT(DISTINCT c.id)
        AS total_comments,

    COUNT(ca.aspect_id)
        AS total_aspect_mentions,

    COUNT(ca.aspect_id)
        FILTER (
            WHERE LOWER(ca.sentiment) = 'positive'
        ) AS positive_aspect_mentions,

    COUNT(ca.aspect_id)
        FILTER (
            WHERE LOWER(ca.sentiment) = 'negative'
        ) AS negative_aspect_mentions,

    COUNT(ca.aspect_id)
        FILTER (
            WHERE LOWER(ca.sentiment) = 'neutral'
        ) AS neutral_aspect_mentions,

    AVG(c.rate)::DOUBLE PRECISION
        AS avg_comment_rating,

    SUM(COALESCE(c.likes,0))
        AS total_likes,

    SUM(COALESCE(c.dislikes,0))
        AS total_dislikes

FROM public.products p

LEFT JOIN public.comments c
    ON p.id = c.product_id

LEFT JOIN public.comment_aspects ca
    ON c.id = ca.comment_id

GROUP BY p.category_id;


CREATE UNIQUE INDEX idx_feature_category_sentiment
ON analytics.feature_category_sentiment(category_id);


-- ============================================================
-- 14. GLOBAL ASPECT FEATURES
--
-- Grain:
--     ONE ROW PER ASPECT TERM
--
-- Key:
--     term
--
-- ============================================================

CREATE TABLE analytics.feature_aspect AS

SELECT

    ca.term,

    COUNT(*) AS total_mentions,

    COUNT(*) FILTER (
        WHERE LOWER(ca.sentiment) = 'positive'
    ) AS positive_mentions,

    COUNT(*) FILTER (
        WHERE LOWER(ca.sentiment) = 'negative'
    ) AS negative_mentions,

    COUNT(*) FILTER (
        WHERE LOWER(ca.sentiment) = 'neutral'
    ) AS neutral_mentions,

    AVG(ca.negative_pct)
        AS avg_negative_pct,

    AVG(ca.neutral_pct)
        AS avg_neutral_pct,

    AVG(ca.positive_pct)
        AS avg_positive_pct

FROM public.comment_aspects ca

WHERE ca.term IS NOT NULL

GROUP BY ca.term;


CREATE UNIQUE INDEX idx_feature_aspect
ON analytics.feature_aspect(term);


-- ============================================================
-- 15. TIME FEATURES
--
-- Grain:
--     ONE ROW PER HOUR / WEEKDAY
--
-- Keys:
--     hour
--     iso_weekday
--
-- ============================================================

CREATE TABLE analytics.feature_time AS

SELECT

    fb.hour,

    fb.weekday AS iso_weekday,

    COUNT(*) AS total_events,

    SUM(fb.is_view)
        AS total_views,

    SUM(fb.is_cart)
        AS total_cart_adds,

    SUM(fb.is_purchase)
        AS total_purchases,

    SUM(fb.is_remove)
        AS total_removes

FROM analytics.feature_behavior fb

GROUP BY

    fb.hour,
    fb.weekday;


CREATE UNIQUE INDEX idx_feature_time
ON analytics.feature_time(
    hour,
    iso_weekday
);


-- ============================================================
-- 16. ANALYZE
-- ============================================================

ANALYZE analytics.feature_behavior;
ANALYZE analytics.feature_user;
ANALYZE analytics.feature_product;
ANALYZE analytics.feature_city;
ANALYZE analytics.feature_category;
ANALYZE analytics.feature_brand;
ANALYZE analytics.feature_user_product;
ANALYZE analytics.feature_user_category;
ANALYZE analytics.feature_product_sentiment;
ANALYZE analytics.feature_product_aspect;
ANALYZE analytics.feature_brand_sentiment;
ANALYZE analytics.feature_category_sentiment;
ANALYZE analytics.feature_aspect;
ANALYZE analytics.feature_time;


-- ============================================================
-- 17. SUMMARY
-- ============================================================

SELECT
    'feature_behavior' AS table_name,
    COUNT(*) AS row_count
FROM analytics.feature_behavior

UNION ALL

SELECT
    'feature_user',
    COUNT(*)
FROM analytics.feature_user

UNION ALL

SELECT
    'feature_product',
    COUNT(*)
FROM analytics.feature_product

UNION ALL

SELECT
    'feature_city',
    COUNT(*)
FROM analytics.feature_city

UNION ALL

SELECT
    'feature_category',
    COUNT(*)
FROM analytics.feature_category

UNION ALL

SELECT
    'feature_brand',
    COUNT(*)
FROM analytics.feature_brand

UNION ALL

SELECT
    'feature_user_product',
    COUNT(*)
FROM analytics.feature_user_product

UNION ALL

SELECT
    'feature_user_category',
    COUNT(*)
FROM analytics.feature_user_category

UNION ALL

SELECT
    'feature_product_sentiment',
    COUNT(*)
FROM analytics.feature_product_sentiment

UNION ALL

SELECT
    'feature_product_aspect',
    COUNT(*)
FROM analytics.feature_product_aspect

UNION ALL

SELECT
    'feature_brand_sentiment',
    COUNT(*)
FROM analytics.feature_brand_sentiment

UNION ALL

SELECT
    'feature_category_sentiment',
    COUNT(*)
FROM analytics.feature_category_sentiment

UNION ALL

SELECT
    'feature_aspect',
    COUNT(*)
FROM analytics.feature_aspect

UNION ALL

SELECT
    'feature_time',
    COUNT(*)
FROM analytics.feature_time;