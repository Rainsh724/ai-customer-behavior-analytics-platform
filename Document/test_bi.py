# import json
# from Graph.bi_agent import run_bi_tool
# import os
# import plotly.graph_objects as go
# # os.environ["DATABASE_URL"] = "postgresql://postgres:zynb1223@localhost:5432/ai_project"
# # --- تست مستقیم ابزار BI ---
# def test_my_chart():
#     print("در حال اجرای تست ابزار BI...")
    
#     # فرض می‌کنیم ال‌ال‌ام این کوئری را برای مقایسه خوشه‌های مشتریان تولید کرده
#     test_sql = """
#     SELECT cluster_name, COUNT(user_id) as user_count 
#     FROM kpi.ml_user_clusters 
#     GROUP BY cluster_name;
#     """
    
#     # فراخوانی تابعی که دوستتان نوشته
#     # ما صریحاً بهش می‌گیم نمودار میله‌ای (bar) بساز و محورها رو چی بذار
#     result = run_bi_tool(
#         sql=test_sql,
#         chart_type="bar",
#         title="تعداد کاربران در هر خوشه رفتاری",
#         x_field="cluster_name",
#         y_field="user_count"
#     )
    
#     # بررسی می‌کنیم که آیا اروری رخ داده؟
#     if "error" in result:
#         print(f"❌ خطا در اجرای ابزار: {result['error']}")
#         return

#     # چاپ خروجی به شکل زیبا و فرمت شده
#     print("✅ نمودار با موفقیت ساخته شد!")
#     print("\n--- دیتای خام برای React (Recharts) ---")
#     print(json.dumps(result["raw_data"], indent=2, ensure_ascii=False))
    
#     print("\n--- کانفیگ ساخته شده برای ECharts ---")
#     print(json.dumps(result["echarts_option"], indent=2, ensure_ascii=False))

# if __name__ == "__main__":
#     test_my_chart()



import json
import os
import plotly.graph_objects as go
from Graph.bi_agent import run_bi_tool

def test_my_chart():
    print("در حال اجرای تست ابزار BI...")
    
    # کوئری واقعی روی جدول خوشه‌هایی که ساختیم
    test_sql = """
    SELECT cluster_name, COUNT(user_id) as user_count 
    FROM kpi.ml_user_clusters 
    GROUP BY cluster_name;
    """
    
    # فراخوانی ابزار
    result = run_bi_tool(
        sql=test_sql,
        chart_type="bar",
        title="تعداد کاربران در هر خوشه رفتاری",
        x_field="cluster_name",
        y_field="user_count"
    )
    
    if "error" in result:
        print(f"❌ خطا در اجرای ابزار: {result['error']}")
        return

    print("✅ داده‌ها با موفقیت از دیتابیس دریافت شد!")
    
    # --- بخش جادویی برای نمایش تصویری نمودار ---
    print("در حال باز کردن نمودار در مرورگر شما...")
    fig = go.Figure(result["plotly_figure"])
    
    # این دستور مرورگر شما را باز می‌کند و نمودار را نشان می‌دهد
    fig.show()

if __name__ == "__main__":
    test_my_chart()