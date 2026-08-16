-- ============================================================
-- 01_feature_engineering.sql
--
-- FEATURE ENGINEERING LAYER
--
-- Source:
--   public.users
--   public.sessions
--   public.cities
--   public.products
--   public.brands
--   public.categories
--   public.sellers
--   public.user_behavior_logs
--   public.comments
--
-- Output:
--   analytics.feature_user
--   analytics.feature_product
--   analytics.feature_city
--   analytics.feature_category
--   analytics.feature_brand
--   analytics.feature_user_product
--   analytics.feature_user_category
--   analytics.feature_product_sentiment
--   analytics.feature_product_aspect
--   analytics.feature_brand_sentiment
--   analytics.feature_category_sentiment
--   analytics.feature_aspect
--   analytics.feature_time
--
-- IMPORTANT:
-- Dimension attributes such as product title, brand name,
-- category names, seller names, etc. are NOT duplicated.
-- Only foreign keys / join keys are retained.
-- ============================================================


-- ============================================================
-- 0. SCHEMA
-- ============================================================

CREATE SCHEMA IF NOT EXISTS analytics;


-- ============================================================
-- 1. CLEAN OLD ANALYTICS TABLES
-- ============================================================

DROP TABLE IF EXISTS analytics.feature_product_aspect CASCADE;
DROP TABLE IF EXISTS analytics.feature_product_sentiment CASCADE;
DROP TABLE IF EXISTS analytics.feature_brand_sentiment CASCADE;
DROP TABLE IF EXISTS analytics.feature_category_sentiment CASCADE;
DROP TABLE IF EXISTS analytics.feature_aspect CASCADE;

DROP TABLE IF EXISTS analytics.feature_user_category CASCADE;
DROP TABLE IF EXISTS analytics.feature_user_product CASCADE;

DROP TABLE IF EXISTS analytics.feature_time CASCADE;

DROP TABLE IF EXISTS analytics.feature_brand CASCADE;
DROP TABLE IF EXISTS analytics.feature_category CASCADE;
DROP TABLE IF EXISTS analytics.feature_city CASCADE;
DROP TABLE IF EXISTS analytics.feature_product CASCADE;
DROP TABLE IF EXISTS analytics.feature_user CASCADE;


-- ============================================================
-- 2. USER FEATURES
-- Grain:
-- ONE ROW PER USER
-- ============================================================

CREATE TABLE analytics.feature_user AS

WITH event_stats AS (

    SELECT
        s.user_id,

        COUNT(*) AS total_events,

        COUNT(DISTINCT b.session_id)
            AS total_sessions,

        COUNT(DISTINCT DATE(b.timestamp))
            AS active_days,

        MIN(b.timestamp)
            AS first_activity_at,

        MAX(b.timestamp)
            AS last_activity_at,

        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ) AS total_views,

        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ) AS total_cart_adds,

        COUNT(*) FILTER (
            WHERE b.event_type = 'remove_from_cart'
        ) AS total_removes,

        COUNT(*) FILTER (
            WHERE b.event_type = 'purchase'
        ) AS total_purchases,

        COUNT(DISTINCT b.product_id) FILTER (
            WHERE b.event_type = 'view'
        ) AS unique_products_viewed,

        COUNT(DISTINCT b.product_id) FILTER (
            WHERE b.event_type = 'purchase'
        ) AS unique_products_purchased,

        COUNT(DISTINCT s.city_id)
            AS cities_visited,

        SUM(
            CASE
                WHEN EXTRACT(ISODOW FROM b.timestamp) IN (4,5)
                THEN 1 ELSE 0
            END
        )::DOUBLE PRECISION / NULLIF(COUNT(*),0)
            AS weekend_activity_ratio,

        SUM(
            CASE
                WHEN EXTRACT(HOUR FROM b.timestamp) >= 6
                 AND EXTRACT(HOUR FROM b.timestamp) < 12
                THEN 1 ELSE 0
            END
        )::DOUBLE PRECISION / NULLIF(COUNT(*),0)
            AS morning_activity_ratio,

        SUM(
            CASE
                WHEN EXTRACT(HOUR FROM b.timestamp) >= 12
                 AND EXTRACT(HOUR FROM b.timestamp) < 18
                THEN 1 ELSE 0
            END
        )::DOUBLE PRECISION / NULLIF(COUNT(*),0)
            AS afternoon_activity_ratio,

        SUM(
            CASE
                WHEN EXTRACT(HOUR FROM b.timestamp) >= 18
                 AND EXTRACT(HOUR FROM b.timestamp) < 24
                THEN 1 ELSE 0
            END
        )::DOUBLE PRECISION / NULLIF(COUNT(*),0)
            AS evening_activity_ratio,

        SUM(
            CASE
                WHEN EXTRACT(HOUR FROM b.timestamp) < 6
                THEN 1 ELSE 0
            END
        )::DOUBLE PRECISION / NULLIF(COUNT(*),0)
            AS night_activity_ratio

    FROM public.user_behavior_logs b

    INNER JOIN public.sessions s
        ON b.session_id = s.session_id

    GROUP BY s.user_id
),

session_stats AS (

    SELECT
        x.user_id,

        AVG(session_events)::DOUBLE PRECISION
            AS avg_session_events,

        MAX(session_events)
            AS max_session_events,

        AVG(session_duration_minutes)::DOUBLE PRECISION
            AS avg_session_duration_minutes,

        MAX(session_duration_minutes)
            AS max_session_duration_minutes

    FROM (

        SELECT
            b.session_id,

            s.user_id,

            COUNT(*) AS session_events,

            EXTRACT(
                EPOCH FROM (
                    MAX(b.timestamp) - MIN(b.timestamp)
                )
            ) / 60.0 AS session_duration_minutes

        FROM public.user_behavior_logs b

        INNER JOIN public.sessions s
            ON b.session_id = s.session_id

        GROUP BY
            b.session_id,
            s.user_id

    ) x

    GROUP BY x.user_id
),

purchase_stats AS (

    SELECT

        s.user_id,

        MIN(b.timestamp)
            AS first_purchase_at,

        MAX(b.timestamp)
            AS last_purchase_at,

        COUNT(*) AS purchase_frequency,

        COUNT(DISTINCT DATE(b.timestamp))
            AS purchase_days,

        SUM(p.price)::DOUBLE PRECISION
            AS total_spend,

        AVG(p.price)::DOUBLE PRECISION
            AS avg_order_value,

        MIN(p.price)
            AS min_purchase_price,

        MAX(p.price)
            AS max_purchase_price,

        COUNT(DISTINCT p.brand_id)
            AS brand_diversity,

        COUNT(DISTINCT p.category_id)
            AS category_diversity

    FROM public.user_behavior_logs b

    INNER JOIN public.sessions s
        ON b.session_id = s.session_id

    INNER JOIN public.products p
        ON b.product_id = p.id

    WHERE b.event_type = 'purchase'

    GROUP BY s.user_id
),

time_pref AS (

    SELECT
        user_id,

        MODE() WITHIN GROUP (
            ORDER BY event_hour
        ) AS preferred_hour,

        MODE() WITHIN GROUP (
            ORDER BY iso_weekday
        ) AS preferred_weekday

    FROM (

        SELECT
            s.user_id,

            EXTRACT(
                HOUR FROM b.timestamp
            )::SMALLINT AS event_hour,

            EXTRACT(
                ISODOW FROM b.timestamp
            )::SMALLINT AS iso_weekday

        FROM public.user_behavior_logs b

        INNER JOIN public.sessions s
            ON b.session_id = s.session_id

    ) x

    GROUP BY user_id
)

SELECT

    e.user_id,

    e.total_events,
    e.total_sessions,
    e.active_days,

    e.first_activity_at,
    e.last_activity_at,

    GREATEST(
        DATE(e.last_activity_at)
        - DATE(e.first_activity_at),
        0
    ) AS lifetime_days,

    e.total_views,
    e.total_cart_adds,
    e.total_removes,
    e.total_purchases,

    e.total_cart_adds::DOUBLE PRECISION
        / NULLIF(e.total_views,0)
        AS view_to_cart_rate,

    e.total_purchases::DOUBLE PRECISION
        / NULLIF(e.total_cart_adds,0)
        AS cart_to_purchase_rate,

    e.total_purchases::DOUBLE PRECISION
        / NULLIF(e.total_views,0)
        AS conversion_rate,

    GREATEST(
        e.total_cart_adds - e.total_purchases,
        0
    )::DOUBLE PRECISION
        / NULLIF(e.total_cart_adds,0)
        AS cart_abandonment_rate,

    e.total_removes::DOUBLE PRECISION
        / NULLIF(e.total_cart_adds,0)
        AS remove_rate,

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

    p.first_purchase_at,
    p.last_purchase_at,

    COALESCE(p.total_spend,0)
        AS total_spend,

    COALESCE(p.avg_order_value,0)
        AS avg_order_value,

    COALESCE(p.min_purchase_price,0)
        AS min_purchase_price,

    COALESCE(p.max_purchase_price,0)
        AS max_purchase_price,

    COALESCE(p.purchase_frequency,0)
        AS purchase_frequency,

    COALESCE(p.purchase_days,0)
        AS purchase_days,

    COALESCE(p.brand_diversity,0)
        AS brand_diversity,

    COALESCE(p.category_diversity,0)
        AS category_diversity,

    CASE
        WHEN p.last_purchase_at IS NOT NULL
        THEN CURRENT_DATE - DATE(p.last_purchase_at)
        ELSE NULL
    END AS recency_days,

    COALESCE(p.purchase_frequency,0)
        AS frequency,

    COALESCE(p.total_spend,0)
        AS monetary

FROM event_stats e

LEFT JOIN session_stats s
    ON e.user_id = s.user_id

LEFT JOIN purchase_stats p
    ON e.user_id = p.user_id

LEFT JOIN time_pref t
    ON e.user_id = t.user_id;


CREATE INDEX idx_feature_user_user
ON analytics.feature_user(user_id);


-- ============================================================
-- 3. PRODUCT FEATURES
-- Grain:
-- ONE ROW PER PRODUCT
-- ============================================================

CREATE TABLE analytics.feature_product AS

WITH behavior AS (

    SELECT

        b.product_id,

        COUNT(*) AS total_events,

        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ) AS total_views,

        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ) AS total_cart_adds,

        COUNT(*) FILTER (
            WHERE b.event_type = 'remove_from_cart'
        ) AS total_removes,

        COUNT(*) FILTER (
            WHERE b.event_type = 'purchase'
        ) AS total_purchases,

        COUNT(DISTINCT s.user_id) FILTER (
            WHERE b.event_type = 'view'
        ) AS unique_viewers,

        COUNT(DISTINCT s.user_id) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ) AS unique_carters,

        COUNT(DISTINCT s.user_id) FILTER (
            WHERE b.event_type = 'purchase'
        ) AS unique_buyers,

        COUNT(DISTINCT b.session_id)
            AS total_sessions

    FROM public.user_behavior_logs b

    INNER JOIN public.sessions s
        ON b.session_id = s.session_id

    GROUP BY b.product_id
)

SELECT

    p.id AS product_id,

    p.brand_id,
    p.category_id,
    p.seller_id,

    p.price,
    p.min_price_last_month,

    p.is_fake,

    p.rate,
    p.rate_cnt,

    COALESCE(b.total_events,0)
        AS total_events,

    COALESCE(b.total_views,0)
        AS total_views,

    COALESCE(b.total_cart_adds,0)
        AS total_cart_adds,

    COALESCE(b.total_removes,0)
        AS total_removes,

    COALESCE(b.total_purchases,0)
        AS total_purchases,

    COALESCE(b.unique_viewers,0)
        AS unique_viewers,

    COALESCE(b.unique_carters,0)
        AS unique_carters,

    COALESCE(b.unique_buyers,0)
        AS unique_buyers,

    COALESCE(b.total_sessions,0)
        AS total_sessions,

    COALESCE(
        b.total_cart_adds::DOUBLE PRECISION
        / NULLIF(b.total_views,0),
        0
    ) AS view_to_cart_rate,

    COALESCE(
        b.total_purchases::DOUBLE PRECISION
        / NULLIF(b.total_cart_adds,0),
        0
    ) AS cart_to_purchase_rate,

    COALESCE(
        b.total_purchases::DOUBLE PRECISION
        / NULLIF(b.total_views,0),
        0
    ) AS conversion_rate,

    COALESCE(
        GREATEST(
            b.total_cart_adds - b.total_purchases,
            0
        )::DOUBLE PRECISION
        / NULLIF(b.total_cart_adds,0),
        0
    ) AS cart_abandonment_rate,

    COALESCE(
        b.total_removes::DOUBLE PRECISION
        / NULLIF(b.total_cart_adds,0),
        0
    ) AS remove_rate,

    COALESCE(
        b.total_purchases * p.price,
        0
    )::DOUBLE PRECISION AS estimated_revenue,

    GREATEST(
        (
            p.min_price_last_month - p.price
        )::DOUBLE PRECISION
        / NULLIF(p.min_price_last_month,0),
        0
    ) AS price_drop_ratio

FROM public.products p

LEFT JOIN behavior b
    ON p.id = b.product_id;


CREATE INDEX idx_feature_product_product
ON analytics.feature_product(product_id);

CREATE INDEX idx_feature_product_brand
ON analytics.feature_product(brand_id);

CREATE INDEX idx_feature_product_category
ON analytics.feature_product(category_id);


-- ============================================================
-- 4. CITY FEATURES
-- Grain:
-- ONE ROW PER CITY
-- ============================================================

CREATE TABLE analytics.feature_city AS

SELECT

    s.city_id,

    COUNT(DISTINCT s.user_id)
        AS total_users,

    COUNT(DISTINCT b.session_id)
        AS total_sessions,

    COUNT(*) AS total_events,

    COUNT(*) FILTER (
        WHERE b.event_type = 'view'
    ) AS total_views,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    ) AS total_cart_adds,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS total_purchases,

    COUNT(*) FILTER (
        WHERE b.event_type = 'remove_from_cart'
    ) AS total_removes,

    COUNT(DISTINCT b.product_id) FILTER (
        WHERE b.event_type = 'view'
    ) AS unique_products_viewed,

    COUNT(DISTINCT b.product_id) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS unique_products_purchased,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS view_to_cart_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ),
        0
    ) AS cart_to_purchase_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS conversion_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'remove_from_cart'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ),
        0
    ) AS remove_rate

FROM public.user_behavior_logs b

INNER JOIN public.sessions s
    ON b.session_id = s.session_id

GROUP BY s.city_id;


CREATE INDEX idx_feature_city_city
ON analytics.feature_city(city_id);


-- ============================================================
-- 5. CATEGORY FEATURES
-- Grain:
-- ONE ROW PER CATEGORY
-- ============================================================

CREATE TABLE analytics.feature_category AS

SELECT

    p.category_id,

    COUNT(*) AS total_events,

    COUNT(*) FILTER (
        WHERE b.event_type = 'view'
    ) AS total_views,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    ) AS total_cart_adds,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS total_purchases,

    COUNT(*) FILTER (
        WHERE b.event_type = 'remove_from_cart'
    ) AS total_removes,

    COUNT(DISTINCT s.user_id) FILTER (
        WHERE b.event_type = 'view'
    ) AS unique_viewers,

    COUNT(DISTINCT s.user_id) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS unique_buyers,

    AVG(p.price)::DOUBLE PRECISION
        AS avg_product_price,

    SUM(
        CASE
            WHEN b.event_type = 'purchase'
            THEN p.price
            ELSE 0
        END
    )::DOUBLE PRECISION
        AS estimated_revenue,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS view_to_cart_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ),
        0
    ) AS cart_to_purchase_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS conversion_rate

FROM public.user_behavior_logs b

INNER JOIN public.sessions s
    ON b.session_id = s.session_id

INNER JOIN public.products p
    ON b.product_id = p.id

GROUP BY p.category_id;


CREATE INDEX idx_feature_category_category
ON analytics.feature_category(category_id);


-- ============================================================
-- 6. BRAND FEATURES
-- Grain:
-- ONE ROW PER BRAND
-- ============================================================

CREATE TABLE analytics.feature_brand AS

SELECT

    p.brand_id,

    COUNT(*) AS total_events,

    COUNT(*) FILTER (
        WHERE b.event_type = 'view'
    ) AS total_views,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    ) AS total_cart_adds,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS total_purchases,

    COUNT(*) FILTER (
        WHERE b.event_type = 'remove_from_cart'
    ) AS total_removes,

    COUNT(DISTINCT s.user_id) FILTER (
        WHERE b.event_type = 'view'
    ) AS unique_viewers,

    COUNT(DISTINCT s.user_id) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS unique_buyers,

    SUM(
        CASE
            WHEN b.event_type = 'purchase'
            THEN p.price
            ELSE 0
        END
    )::DOUBLE PRECISION
        AS estimated_revenue,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS view_to_cart_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ),
        0
    ) AS cart_to_purchase_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS conversion_rate

FROM public.user_behavior_logs b

INNER JOIN public.sessions s
    ON b.session_id = s.session_id

INNER JOIN public.products p
    ON b.product_id = p.id

GROUP BY p.brand_id;


CREATE INDEX idx_feature_brand_brand
ON analytics.feature_brand(brand_id);


-- ============================================================
-- 7. USER × PRODUCT
-- Grain:
-- ONE ROW PER USER / PRODUCT
-- ============================================================

CREATE TABLE analytics.feature_user_product AS

SELECT

    s.user_id,

    b.product_id,

    COUNT(*) AS total_events,

    COUNT(*) FILTER (
        WHERE b.event_type = 'view'
    ) AS view_count,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    ) AS cart_count,

    COUNT(*) FILTER (
        WHERE b.event_type = 'remove_from_cart'
    ) AS remove_count,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS purchase_count,

    MIN(b.timestamp)
        AS first_interaction_at,

    MAX(b.timestamp)
        AS last_interaction_at,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS view_to_cart_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ),
        0
    ) AS cart_to_purchase_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    )::DOUBLE PRECISION
    /
    NULLIF(
        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ),
        0
    ) AS conversion_rate,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    ) > 0 AS was_purchased

FROM public.user_behavior_logs b

INNER JOIN public.sessions s
    ON b.session_id = s.session_id

GROUP BY
    s.user_id,
    b.product_id;


CREATE INDEX idx_feature_user_product_user
ON analytics.feature_user_product(user_id);

CREATE INDEX idx_feature_user_product_product
ON analytics.feature_user_product(product_id);


-- ============================================================
-- 8. USER × CATEGORY
-- Grain:
-- ONE ROW PER USER / CATEGORY
-- ============================================================

CREATE TABLE analytics.feature_user_category AS

WITH base AS (

    SELECT

        s.user_id,
        p.category_id,

        COUNT(*) AS total_events,

        COUNT(*) FILTER (
            WHERE b.event_type = 'view'
        ) AS view_count,

        COUNT(*) FILTER (
            WHERE b.event_type = 'add_to_cart'
        ) AS cart_count,

        COUNT(*) FILTER (
            WHERE b.event_type = 'purchase'
        ) AS purchase_count,

        SUM(
            CASE
                WHEN b.event_type = 'purchase'
                THEN p.price
                ELSE 0
            END
        )::DOUBLE PRECISION AS category_spend,

        MIN(b.timestamp)
            AS first_interaction_at,

        MAX(b.timestamp)
            AS last_interaction_at

    FROM public.user_behavior_logs b

    INNER JOIN public.sessions s
        ON b.session_id = s.session_id

    INNER JOIN public.products p
        ON b.product_id = p.id

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
    b.purchase_count,

    b.category_spend,

    b.first_interaction_at,
    b.last_interaction_at,

    b.view_count::DOUBLE PRECISION
    /
    NULLIF(t.total_user_views,0)
        AS view_share,

    b.purchase_count::DOUBLE PRECISION
    /
    NULLIF(t.total_user_purchases,0)
        AS purchase_share,

    b.category_spend::DOUBLE PRECISION
    /
    NULLIF(t.total_user_spend,0)
        AS spend_share

FROM base b

INNER JOIN totals t
    ON b.user_id = t.user_id;


CREATE INDEX idx_feature_user_category_user
ON analytics.feature_user_category(user_id);

CREATE INDEX idx_feature_user_category_category
ON analytics.feature_user_category(category_id);


-- ============================================================
-- 9. COMMENT / SENTIMENT — PRODUCT
--
-- Grain:
-- ONE ROW PER PRODUCT
-- ============================================================

CREATE TABLE analytics.feature_product_sentiment AS

SELECT

    c.product_id,

    COUNT(*) AS total_comments,

    COUNT(*) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'positive'
    ) AS positive_comments,

    COUNT(*) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'negative'
    ) AS negative_comments,

    COUNT(*) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'neutral'
    ) AS neutral_comments,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'positive'
        )
        / NULLIF(COUNT(*),0),
        2
    ) AS positive_sentiment_pct,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'negative'
        )
        / NULLIF(COUNT(*),0),
        2
    ) AS negative_sentiment_pct,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'neutral'
        )
        / NULLIF(COUNT(*),0),
        2
    ) AS neutral_sentiment_pct,

    ROUND(
        (
            100.0 *
            COUNT(*) FILTER (
                WHERE LOWER(c.predicted_sentiment) = 'positive'
            )
            / NULLIF(COUNT(*),0)
        )
        -
        (
            100.0 *
            COUNT(*) FILTER (
                WHERE LOWER(c.predicted_sentiment) = 'negative'
            )
            / NULLIF(COUNT(*),0)
        ),
        2
    ) AS sentiment_score,

    AVG(c.rate)::DOUBLE PRECISION
        AS avg_comment_rating,

    SUM(COALESCE(c.likes,0))
        AS total_likes,

    SUM(COALESCE(c.dislikes,0))
        AS total_dislikes,

    AVG(
        COALESCE(c.likes,0)::DOUBLE PRECISION
        /
        (COALESCE(c.dislikes,0) + 1)
    ) AS avg_like_ratio

FROM public.comments c

GROUP BY c.product_id;


CREATE INDEX idx_product_sentiment_product
ON analytics.feature_product_sentiment(product_id);


-- ============================================================
-- 10. PRODUCT × ASPECT
-- Grain:
-- ONE ROW PER PRODUCT / ASPECT
-- ============================================================

CREATE TABLE analytics.feature_product_aspect AS

WITH exploded AS (

    SELECT

        c.product_id,

        UNNEST(c.predicted_aspects)
            AS aspect,

        LOWER(c.predicted_sentiment)
            AS sentiment

    FROM public.comments c

    WHERE c.predicted_aspects IS NOT NULL
),

stats AS (

    SELECT

        product_id,
        aspect,

        COUNT(*) AS total_mentions,

        COUNT(*) FILTER (
            WHERE sentiment = 'positive'
        ) AS positive_mentions,

        COUNT(*) FILTER (
            WHERE sentiment = 'negative'
        ) AS negative_mentions,

        COUNT(*) FILTER (
            WHERE sentiment = 'neutral'
        ) AS neutral_mentions

    FROM exploded

    GROUP BY
        product_id,
        aspect
)

SELECT

    product_id,

    aspect,

    total_mentions,

    positive_mentions,
    negative_mentions,
    neutral_mentions,

    ROUND(
        100.0 * positive_mentions
        / NULLIF(total_mentions,0),
        2
    ) AS positive_pct,

    ROUND(
        100.0 * negative_mentions
        / NULLIF(total_mentions,0),
        2
    ) AS negative_pct,

    ROUND(
        100.0 * neutral_mentions
        / NULLIF(total_mentions,0),
        2
    ) AS neutral_pct

FROM stats;


CREATE INDEX idx_product_aspect_product
ON analytics.feature_product_aspect(product_id);

CREATE INDEX idx_product_aspect_aspect
ON analytics.feature_product_aspect(aspect);


-- ============================================================
-- 11. PRODUCT TOP / WORST ASPECT
-- Grain:
-- ONE ROW PER PRODUCT
-- ============================================================

ALTER TABLE analytics.feature_product
ADD COLUMN top_aspect TEXT,
ADD COLUMN worst_aspect TEXT;


WITH aspect_counts AS (

    SELECT

        product_id,
        aspect,

        SUM(total_mentions) AS mentions

    FROM analytics.feature_product_aspect

    GROUP BY
        product_id,
        aspect
),

top_aspects AS (

    SELECT DISTINCT ON (product_id)

        product_id,
        aspect

    FROM aspect_counts

    ORDER BY
        product_id,
        mentions DESC

),

worst_aspects AS (

    SELECT DISTINCT ON (product_id)

        product_id,
        aspect

    FROM analytics.feature_product_aspect

    WHERE negative_mentions > 0

    ORDER BY
        product_id,
        negative_mentions DESC

)

UPDATE analytics.feature_product p

SET top_aspect = t.aspect

FROM top_aspects t

WHERE p.product_id = t.product_id;


UPDATE analytics.feature_product p

SET worst_aspect = w.aspect

FROM worst_aspects w

WHERE p.product_id = w.product_id;


-- ============================================================
-- 12. BRAND SENTIMENT
-- Grain:
-- ONE ROW PER BRAND
-- ============================================================

CREATE TABLE analytics.feature_brand_sentiment AS

SELECT

    p.brand_id,

    COUNT(c.id) AS total_comments,

    COUNT(c.id) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'positive'
    ) AS positive_comments,

    COUNT(c.id) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'negative'
    ) AS negative_comments,

    COUNT(c.id) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'neutral'
    ) AS neutral_comments,

    ROUND(
        100.0 *
        COUNT(c.id) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'positive'
        )
        /
        NULLIF(COUNT(c.id),0),
        2
    ) AS positive_sentiment_pct,

    ROUND(
        100.0 *
        COUNT(c.id) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'negative'
        )
        /
        NULLIF(COUNT(c.id),0),
        2
    ) AS negative_sentiment_pct,

    ROUND(
        100.0 *
        COUNT(c.id) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'neutral'
        )
        /
        NULLIF(COUNT(c.id),0),
        2
    ) AS neutral_sentiment_pct,

    AVG(c.rate)::DOUBLE PRECISION
        AS avg_comment_rating

FROM public.products p

LEFT JOIN public.comments c
    ON p.id = c.product_id

GROUP BY p.brand_id;


-- ============================================================
-- 13. CATEGORY SENTIMENT
-- Grain:
-- ONE ROW PER CATEGORY
-- ============================================================

CREATE TABLE analytics.feature_category_sentiment AS

SELECT

    p.category_id,

    COUNT(c.id) AS total_comments,

    COUNT(c.id) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'positive'
    ) AS positive_comments,

    COUNT(c.id) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'negative'
    ) AS negative_comments,

    COUNT(c.id) FILTER (
        WHERE LOWER(c.predicted_sentiment) = 'neutral'
    ) AS neutral_comments,

    ROUND(
        100.0 *
        COUNT(c.id) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'positive'
        )
        /
        NULLIF(COUNT(c.id),0),
        2
    ) AS positive_sentiment_pct,

    ROUND(
        100.0 *
        COUNT(c.id) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'negative'
        )
        /
        NULLIF(COUNT(c.id),0),
        2
    ) AS negative_sentiment_pct,

    ROUND(
        100.0 *
        COUNT(c.id) FILTER (
            WHERE LOWER(c.predicted_sentiment) = 'neutral'
        )
        /
        NULLIF(COUNT(c.id),0),
        2
    ) AS neutral_sentiment_pct

FROM public.products p

LEFT JOIN public.comments c
    ON p.id = c.product_id

GROUP BY p.category_id;


-- ============================================================
-- 14. GLOBAL ASPECT FEATURES
-- Grain:
-- ONE ROW PER ASPECT
-- ============================================================

CREATE TABLE analytics.feature_aspect AS

SELECT

    aspect,

    COUNT(*) AS total_mentions,

    COUNT(*) FILTER (
        WHERE sentiment = 'positive'
    ) AS positive_mentions,

    COUNT(*) FILTER (
        WHERE sentiment = 'negative'
    ) AS negative_mentions,

    COUNT(*) FILTER (
        WHERE sentiment = 'neutral'
    ) AS neutral_mentions,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE sentiment = 'positive'
        )
        / NULLIF(COUNT(*),0),
        2
    ) AS positive_pct,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE sentiment = 'negative'
        )
        / NULLIF(COUNT(*),0),
        2
    ) AS negative_pct,

    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE sentiment = 'neutral'
        )
        / NULLIF(COUNT(*),0),
        2
    ) AS neutral_pct

FROM (

    SELECT

        UNNEST(c.predicted_aspects) AS aspect,

        LOWER(c.predicted_sentiment) AS sentiment

    FROM public.comments c

    WHERE c.predicted_aspects IS NOT NULL

) x

GROUP BY aspect;


-- ============================================================
-- 15. TIME FEATURES
-- Grain:
-- ONE ROW PER HOUR
-- ============================================================

CREATE TABLE analytics.feature_time AS

SELECT

    EXTRACT(
        HOUR FROM b.timestamp
    )::SMALLINT AS hour,

    EXTRACT(
        ISODOW FROM b.timestamp
    )::SMALLINT AS iso_weekday,

    COUNT(*) AS total_events,

    COUNT(*) FILTER (
        WHERE b.event_type = 'view'
    ) AS total_views,

    COUNT(*) FILTER (
        WHERE b.event_type = 'add_to_cart'
    ) AS total_cart_adds,

    COUNT(*) FILTER (
        WHERE b.event_type = 'purchase'
    ) AS total_purchases,

    COUNT(*) FILTER (
        WHERE b.event_type = 'remove_from_cart'
    ) AS total_removes

FROM public.user_behavior_logs b

GROUP BY
    EXTRACT(HOUR FROM b.timestamp),
    EXTRACT(ISODOW FROM b.timestamp);


-- ============================================================
-- 16. ANALYZE
-- ============================================================

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
-- 17. FEATURE TABLE SUMMARY
-- ============================================================

SELECT
    'feature_user' AS table_name,
    COUNT(*) AS row_count
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