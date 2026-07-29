import sys
from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.structured_cleaner import EcomDataCleaner
from config.feature_map import get_dataset

def test_with_real_data():
    cleaner = EcomDataCleaner()
    
 
    input_dir = Path("test_sandbox")
    output_dir = Path("test_sandbox/cleaned_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 شروع تست پایپ‌لاین ساختاریافته روی دیتای واقعی فروشگاه...\n" + "="*60)
    
    if not input_dir.exists() or not list(input_dir.glob("*.*")):
        print("⚠️ خطای مسیر: هیچ فایلی در پوشه test_sandbox پیدا نشد! لطفاً فایل‌های دیتا را آنجا کپی کن.")
        return

    for file_path in input_dir.glob("*.*"):
        dataset_name = file_path.stem
        
        # بررسی اینکه آیا در فیچرمپ تعریف شده یا نه
        if not get_dataset(dataset_name):
            print(f"\n⏭️ رد شدن از فایل [{file_path.name}]: در feature_map تعریف نشده است.")
            continue
            
        print(f"\n📂 در حال تحلیل و تمیزکاری فایل: [{file_path.name}] ...")
        
        # خواندن فایل خام
        if file_path.suffix == ".csv":
            raw_df = pd.read_csv(file_path, low_memory=False)
        elif file_path.suffix in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(file_path)
        else:
            continue
            
        initial_rows = len(raw_df)
        initial_nulls = raw_df.isnull().sum().sum()
        
        # ==========================================
        # اجرای موتور پاک‌سازی
        # ==========================================
        cleaned_df = cleaner.clean_structured_data(raw_df, dataset_name)
        
        final_rows = len(cleaned_df)
        final_nulls = cleaned_df.isnull().sum().sum()
        
        # چاپ گزارش تست چشمی برای ارزیابی عملکرد
        print("   📊 --- گزارش عملکرد پایپ‌لاین ---")
        print(f"   🔸 تعداد کل رکوردهای خام: {initial_rows:,} | رکوردهای تمیز و نهایی: {final_rows:,}")
        print(f"   🔸 رکوردهای تکراری/ناقص حذف‌شده: {initial_rows - final_rows:,}")
        print(f"   🔸 کل مقادیر گمشده (Null) قبل: {initial_nulls:,} | بعد از مدیریت: {final_nulls:,}")
        
        # ذخیره خروجی تمیزشده به صورت CSV استاندارد دیتابیس (بدون ایندکس)
        save_path = output_dir / f"cleaned_{dataset_name}.csv"
        cleaned_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"   💾 فایل تمیزشده و آماده بارگذاری در SQL ذخیره شد در: {save_path}")

    print("\n" + "="*60)
    print("🎉 تمام تست‌ها با موفقیت به اتمام رسید! فایل‌های تمیز در پوشه test_sandbox/cleaned_output آماده بررسی هستند.")

if __name__ == "__main__":
    test_with_real_data()