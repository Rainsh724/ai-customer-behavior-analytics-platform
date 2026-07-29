import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import jdatetime

# =========================================================================
# حل مشکل ModuleNotFoundError: معرفی روت اصلی پروژه به پایتون
# =========================================================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# اتصال به فایل فیچر مپ آپدیت‌شده
from config.feature_map import (
    get_dataset, 
    get_numeric_columns, 
    get_datetime_columns,
    get_boolean_columns
)

class EcomDataCleaner:
    """
    موتور داینامیک پاک‌سازی داده‌های ساختاریافته
    متصل به Feature Map برای پردازش خودکار دیتاست‌های هر سازمان
    """
    def __init__(self):
        pass

    # =========================================================================
    # توابع کمکی عمومی
    # =========================================================================
    def _clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """حذف فاصله‌های اضافی (strip) و یکدست‌سازی متون خالی به NaN"""
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        return df

    def _clean_currency(self, series: pd.Series) -> pd.Series:
        """حذف کاراکترهای غیرعددی، کاما و متون مالی و تبدیل به Float"""
        return (series.astype(str)
                      .str.replace(r'[^\d.]', '', regex=True)
                      .replace('', np.nan)
                      .astype(float))

    def _clean_booleans(self, series: pd.Series) -> pd.Series:
        """تبدیل مقادیر مختلف به True/False استاندارد دیتابیس"""
        mapping = {
            'yes': True, 'true': True, '1': True, 1: True, 'دارد': True,
            'no': False, 'false': False, '0': False, 0: False, 'ندارد': False
        }
        return series.astype(str).str.lower().str.strip().map(mapping).astype('boolean')

    def _standardize_dates(self, series: pd.Series) -> pd.Series:
        # تبدیل هوشمند تاریخ‌های متنی دیجی‌کالا به فرمت میلادی
        month_map = {
            'فروردین': 1, 'اردیبهشت': 2, 'خرداد': 3,
            'تیر': 4, 'مرداد': 5, 'شهریور': 6,
            'مهر': 7, 'آبان': 8, 'آذر': 9,
            'دی': 10, 'بهمن': 11, 'اسفند': 12
        }

        def convert_to_gregorian(val):
            # ۱. مدیریت مقادیر خالی
            if pd.isna(val) or str(val).strip() in ['', 'nan', 'None', 'NaT']:
                return np.nan
            
            val_str = str(val).strip()
            
            # ۲. اگر تاریخ از قبل میلادی است (مثل لاگ رفتار کاربران)
            if val_str.startswith('20'):
                return val_str
                
            try:
                y, m, d = None, None, None
                
                # ۳. استخراج نام ماه شمسی
                for m_name, m_num in month_map.items():
                    if m_name in val_str:
                        m = m_num
                        break
                
                # ۴. استخراج سال (یک عدد ۴ رقمی که با 13 یا 14 شروع شود)
                year_match = re.search(r'\b(13\d{2}|14\d{2})\b', val_str)
                if year_match:
                    y = int(year_match.group(1))
                    
                # ۵. استخراج روز (یک عدد ۱ یا ۲ رقمی که سال نباشد)
                val_str_no_year = re.sub(r'\b(13\d{2}|14\d{2})\b', '', val_str)
                day_match = re.search(r'\b(\d{1,2})\b', val_str_no_year)
                if day_match:
                    d = int(day_match.group(1))
                    
                # ۶. اگر هر ۳ جزء پیدا شد، تبدیل به میلادی کن
                if y and m and d:
                    greg_date = jdatetime.date(y, m, d).togregorian()
                    return f"{greg_date.year}-{greg_date.month:02d}-{greg_date.day:02d} 00:00:00"
                
                return val_str
                
            except Exception:
                return np.nan

        # اجرای تابع روی تک‌تک سلول‌های ستون تاریخ
        converted_series = series.apply(convert_to_gregorian)
        return pd.to_datetime(converted_series, errors='coerce')

    # =========================================================================
    # تابع اصلی و داینامیک پاک‌سازی
    # =========================================================================
    def clean_structured_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        print(f"\n⏳ در حال پاک‌سازی داینامیک جدول: [{dataset_name}] ...")
        df = df.copy()
        
        # ۱. دریافت شناسنامه جدول از فیچر مپ
        config = get_dataset(dataset_name)
        if not config:
            print(f"⚠️ هشدار: تنظیمات جدول '{dataset_name}' در feature_map پیدا نشد! پاک‌سازی عمومی اعمال می‌شود.")
        
        # ۲. استانداردسازی اولیه تمام رشته‌های متنی جدول
        df = self._clean_string_columns(df)
        
        if config:
            # ۳. اصلاح باگ تکراری‌ها: فقط بر اساس Primary Key یا کپی ۱۰۰ درصدی
            pk = config.get("primary_key")
            
            if pk and pk in df.columns:
                df = df.drop_duplicates(subset=[pk], keep='first')
                df = df.dropna(subset=[pk])
            else:
                df = df.drop_duplicates(keep='first')
            
            # ۴. پاک‌سازی خودکار ستون‌های مالی و عددی
            numeric_cols = get_numeric_columns(dataset_name)
            for col in numeric_cols:
                if col in df.columns:
                    print(f"   --> تمیزکاری ستون عددی/مالی: {col}")
                    df[col] = self._clean_currency(df[col])
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
            
            # ۵. استانداردسازی خودکار ستون‌های تاریخ
            date_cols = get_datetime_columns(dataset_name)
            for col in date_cols:
                if col in df.columns:
                    print(f"   --> استانداردسازی ستون تاریخ: {col}")
                    df[col] = self._standardize_dates(df[col])
                    
            # ۶. تبدیل شناسه‌ها به Int64 برای هماهنگی با SQL
            id_cols = []
            if pk and pk in df.columns:
                id_cols.append(pk)
            if config.get("foreign_keys"):
                id_cols.extend([fk for fk in config["foreign_keys"] if fk in df.columns])
                
            for id_col in set(id_cols):
                df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')

            # ۷. استانداردسازی ستون‌های بولین (ایده جدید هم‌تیمی)
            bool_cols = get_boolean_columns(dataset_name)
            for col in bool_cols:
                if col in df.columns:
                    print(f"   --> استانداردسازی ستون بولین: {col}")
                    df[col] = self._clean_booleans(df[col])
                
        return df.reset_index(drop=True)

# =========================================================================
# تست اجرای پایپ‌لاین روی دیتاست‌های جدید
# =========================================================================
if __name__ == "__main__":
    cleaner = EcomDataCleaner()
    
    # تغییر مسیر به پوشه test_sandbox برای دیتای جدید
    DATA_DIR = project_root / "test_sandbox"
    output_dir = DATA_DIR / "cleaned_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    
    for file_path in files:
        dataset_name = file_path.stem
        config = get_dataset(dataset_name)
        
        if not config:
            print(f"⏭️ رد شدن از فایل [{file_path.name}]: در feature_map تعریف نشده است.")
            continue
            
        print(f"\n📂 در حال خواندن فایل: [{file_path.name}] ...")
        # استفاده از low_memory=False برای جلوگیری از اخطار پانداس در فایل‌های سنگین
        raw_df = pd.read_csv(file_path, low_memory=False)
            
        cleaned_df = cleaner.clean_structured_data(raw_df, dataset_name)
        
        # ذخیره خروجی تمیزشده به فرمت استاندارد برای دیتابیس
        save_path = output_dir / f"cleaned_{dataset_name}.csv"
        cleaned_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ جدول {dataset_name} تمیز شد! تعداد سطرها: {len(cleaned_df):,}")
        print(f"💾 ذخیره شد در: {save_path}") 