
import sys
from pathlib import Path

# =========================================================================
# حل مشکل ModuleNotFoundError: معرفی روت اصلی پروژه به پایتون
# =========================================================================
# این دستور می‌گوید: ۲ پوشه از فایل فعلی عقب برو تا به روت پروژه برسی و آن را به پایتون بشناسان
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# حالا پایتون پوشه config را می‌شناسد و ارور نمی‌دهد!
import pandas as pd
import numpy as np
import re

# اتصال به فایل فیچر مپ ریحانه
from config.feature_map import (
    get_dataset, 
    get_numeric_columns, 
    get_datetime_columns
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

    def _standardize_dates(self, series: pd.Series) -> pd.Series:
        """تبدیل تاریخ‌ها به فرمت استاندارد datetime برای دیتابیس SQL"""
        return pd.to_datetime(series, errors='coerce')

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
            # ۳. حذف هوشمند تکراری‌ها براساس کلید اصلی یا کلیدهای اتصال
            pk = config.get("primary_key")
            join_cols = config.get("join_columns", [])
            
            if pk and pk in df.columns:
                df = df.drop_duplicates(subset=[pk], keep='first')
                df = df.dropna(subset=[pk])
            elif join_cols:
                valid_joins = [col for col in join_cols if col in df.columns]
                if valid_joins:
                    df = df.drop_duplicates(subset=valid_joins, keep='last')
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
                
        return df.reset_index(drop=True)

# =========================================================================
# تست اجرای پایپ‌لاین روی تمام دیتاست‌ها
# =========================================================================
if __name__ == "__main__":
    cleaner = EcomDataCleaner()
    
    DATA_DIR = Path("Dataset")
    files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    
    for file_path in files:
        dataset_name = file_path.stem
        config = get_dataset(dataset_name)
        
        if not config:
            continue
            
        if file_path.suffix == ".csv":
            raw_df = pd.read_csv(file_path)
        elif file_path.suffix in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(file_path)
        else:
            continue
            
        cleaned_df = cleaner.clean_structured_data(raw_df, dataset_name)
        print(f"✅ جدول {dataset_name} تمیز شد! تعداد سطرها: {len(cleaned_df)}")