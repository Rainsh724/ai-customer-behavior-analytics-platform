import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, text

# ۱. اتصال به دیتابیس (هر کس باید یوزر و پسورد خودش را بگذارد)
engine = create_engine('postgresql://postgres:رمز@localhost:5432/postgres')

# ۲. کوئری استخراج داده‌ها (نکته حیاتی: ORDER BY اضافه شد تا ترتیب برای همه یکسان باشد)
query = """
SELECT 
    u.user_id,
    r.recency_days,
    u.total_spend,
    u.purchase_frequency,
    u.category_diversity,
    u.avg_session_duration_minutes,
    u.night_activity_ratio,
    u.weekend_activity_ratio,
    COALESCE((u.total_purchases::numeric / NULLIF(u.total_views, 0)), 0) AS view_to_purchase_ratio
FROM analytics.feature_user u
JOIN kpi.rfm_segments r ON u.user_id = r.user_id
WHERE u.total_purchases > 0
ORDER BY u.user_id ASC; 
"""
df = pd.read_sql(query, engine)
df.fillna(0, inplace=True)

# ۳. جدا کردن آیدی و اسکیل کردن داده‌ها
features = df.drop(columns=['user_id'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ۴. اجرای الگوریتم (با random_state ثابت برای همگام‌سازی تیم)
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster_id'] = kmeans.fit_predict(scaled_features)

# ۵. نام‌گذاری دقیق خوشه‌ها بر اساس تحلیلی که با هم کردیم
cluster_names = {
    0: 'vip_champions',
    1: 'night_weekend_buyers',
    2: 'active_loyals',
    3: 'low_intent_shoppers',
    4: 'churned_customers'
}
df['cluster_name'] = df['cluster_id'].map(cluster_names)

# ۶. ذخیره در دیتابیس (Truncate & Load)
final_clusters = df[['user_id', 'cluster_id', 'cluster_name']]

with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE kpi.ml_user_clusters;"))

final_clusters.to_sql(
    name='ml_user_clusters', 
    schema='kpi', 
    con=engine, 
    if_exists='append', 
    index=False
)
print("خوشه‌بندی با موفقیت انجام و دیتابیس به‌روز شد!")