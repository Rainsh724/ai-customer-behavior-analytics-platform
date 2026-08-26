"""
تست دستی text_summary.py

نحوه‌ی استفاده:
    ۱. لیست COMMENTS پایین رو با نظرات دلخواه خودت (هرچقدر می‌خوای) پر کن.
    ۲. اگه خواستی TOP_N_KEYWORDS / TOP_N_REPRESENTATIVE رو هم عوض کن.
    ۳. اجرا کن:  python3 test_summary_manual.py

این فایل هیچ وابستگی‌ای به بقیه‌ی پروژه (دیتابیس/LLM) نداره -- فقط
text_summary.py رو مستقیم صدا می‌زنه، پس نیازی به هیچ متغیر محیطی
(OPENAI_API_KEY و ...) نیست.

نکته: text_summary.py رو باید کنار همین فایل (یا در PYTHONPATH) داشته
باشی. اگه از پوشه‌ی app/graph/ کپیش کردی همین‌جا، مستقیم اجرا می‌شه.
"""
from text_summary import summarize_comments

# ============================================================
# ۱) نظرات خودتو اینجا بریز -- هرچقدر خواستی، هر جمله رو یک آیتم بذار.
# ============================================================
COMMENTS = [
    "باتری این گوشی خیلی زود خالی میشه و اصلا راضی نیستم",
    "باتری ضعیفه و شارژش خیلی سریع تموم میشه",
    "کیفیت دوربین عالیه ولی باتری واقعا بده",
    "بسته‌بندی خوب بود اما باتری بعد یک ماه افت کرد",
    "من از این محصول راضی‌ام، کارکرد خوبی داره",
    "قیمتش نسبت به کیفیت زیاده",
    "پشتیبانی فروشنده خیلی ضعیف بود و دیر جواب دادن",
]

# ============================================================
# ۲) پارامترهای خلاصه‌سازی -- می‌تونی عوضشون کنی
# ============================================================
TOP_N_KEYWORDS = 10
TOP_N_REPRESENTATIVE = 5


def main() -> None:
    print(f"تعداد نظرات ورودی: {len(COMMENTS)}\n")
    for i, c in enumerate(COMMENTS, 1):
        print(f"  [{i}] {c}")

    result = summarize_comments(
        COMMENTS,
        top_n_keywords=TOP_N_KEYWORDS,
        top_n_representative=TOP_N_REPRESENTATIVE,
    )

    print("\n" + "=" * 50)
    print("مضامین تکرارشونده (top_keywords):")
    print("=" * 50)
    if result["top_keywords"]:
        for kw in result["top_keywords"]:
            print(f"  - {kw}")
    else:
        print("  (چیزی پیدا نشد)")

    print("\n" + "=" * 50)
    print("نظرات نماینده (representative_comments):")
    print("=" * 50)
    if result["representative_comments"]:
        for i, c in enumerate(result["representative_comments"], 1):
            print(f"  {i}. {c}")
    else:
        print("  (چیزی پیدا نشد)")


if __name__ == "__main__":
    main()
