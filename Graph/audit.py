## PATH: app/graph/audit.py
"""
بررسی و اصلاحِ نرم (soft) جواب نهایی -- جبران بخشی از قابلیت ممیزی‌ای که
با حذف نودهای صریح evidence_fusion/answer_validator (نسخه‌ی پایپ‌لاین
قبلی) از دست رفت.

دو تابع اینجاست:
    validate_answer  -- جواب نهایی رو در برابر سوال کاربر و خلاصه‌ی شواهد
                         خام (tool_trace) می‌سنجه و سه محور عددی (۰ تا ۱۰۰)
                         برمی‌گردونه:
                             faithfulness_score -- قبلاً match_score بود؛
                                 یعنی چقدر جواب دقیقاً از شواهد ابزارها
                                 پشتیبانی می‌شه (آیا چیزی حدس/اختراع شده).
                             relevance_score -- آیا جواب واقعاً همون
                                 چیزیه که کاربر پرسیده (نه یک موضوع نزدیک
                                 یا جواب کلی/حاشیه‌ای).
                             confidence_score -- خودِ ممیز چقدر به کافی و
                                 بدون‌ابهام بودنِ شواهد برای این نتیجه‌گیری
                                 مطمئنه (مستقل از faithfulness: faithfulness
                                 یعنی "آیا جواب طبق شواهده"، confidence یعنی
                                 "آیا خودِ شواهد برای این نتیجه کافی/قطعی
                                 بودن").
                         تصمیم retry/correct در graph.py بر اساس همون
                         faithfulness_score گرفته می‌شه (دقیقاً مثل قبل،
                         فقط تغییر اسم). relevance_score و confidence_score
                         صرفاً برای لاگ/ارزیابی کیفیت (نگاه کن به
                         memory_store.py::log_evaluation/compute_calibration)
                         ذخیره می‌شن، در مسیر retry/correct تصمیم‌گیری
                         نمی‌کنن -- چون آستانه‌ی جداگانه برای هرکدوم نیاز به
                         تنظیم/تجربه‌ی جدا داره و فعلاً فقط یک معیار
                         (faithfulness) تصمیم‌گیرِ اصلاح خودکاره.

                         نکته‌ی مهم درباره‌ی calibration: calibration یک
                         معیار per-response نیست -- یعنی از روی یک جواب
                         تنها نمی‌شه گفت مدل "calibrated" هست یا نه. این
                         معیار فقط با جمع‌آوری (confidence_score,
                         faithfulness_score) در طول زمان و مقایسه‌ی
                         آماری‌شون معنی پیدا می‌کنه؛ به همین خاطر اینجا
                         محاسبه نمی‌شه -- confidence_score هر پاسخ لاگ
                         می‌شه (memory_store.py::log_evaluation) و
                         calibration به‌صورت تجمعی/آفلاین از روی همون لاگ
                         حساب می‌شه (memory_store.py::compute_calibration).
    correct_answer    -- وقتی faithfulness_score پایینه، جواب رو یک‌بار
                         (نه در حلقه!) بازنویسی می‌کنه تا هشدارها رفع بشن.

مهم -- چرا این هیچ‌وقت لوپ نمی‌شه:
------------------------------------
correct_answer فقط یک‌بار در graph.py صدا زده می‌شه (validate -> اگه
match_score زیر آستانه بود -> correct_answer -> END) و خروجیش هرگز
دوباره به validate برنمی‌گرده. یعنی حتی اگه جواب اصلاح‌شده هم کامل
grounded نباشه، دیگه یک تلاش دومی برای اصلاح یا ممیزی مجدد وجود نداره --
عمداً همین‌طور طراحی شده تا هزینه/تاخیر قابل‌پیش‌بینی بمونه (حداکثر یک
تماس اضافه‌ی LLM به‌ازای هر پاسخ، نه یک عدد نامشخص).

چون هزینه‌ی این کل مکانیزم (تماس validate + تماس احتمالی correct) به
ازای هر سوال داره، با متغیر محیطی ENABLE_ANSWER_VALIDATION قابل
خاموش‌کردنه (پیش‌فرض: روشن). وقتی خاموشه، correct_answer هم اصلاً صدا
زده نمی‌شه (چون بدون validation، آستانه‌ای برای تصمیم‌گیری نداریم).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .llm_client import call_llm_json

logger = logging.getLogger(__name__)

VALIDATION_ENABLED = os.getenv("ENABLE_ANSWER_VALIDATION", "true").strip().lower() in ("1", "true", "yes")

# اگه faithfulness_score زیر این عدد باشه، correct_answer صدا زده می‌شه.
CORRECTION_THRESHOLD = int(os.getenv("VALIDATION_CORRECTION_THRESHOLD", "70"))

VALIDATION_SYSTEM_PROMPT = """
تو یک ممیز مستقل هستی. یک "سوال کاربر"، یک "جواب نهایی" و خلاصه‌ای از
"شواهد خام" (نتایج واقعی ابزارهایی که صدا زده شدن) رو می‌گیری. فقط یک
JSON با این فرمت برگردون -- هیچ متن اضافه‌ای ننویس:

{
  "grounded": true|false,
  "faithfulness_score": <عدد صحیح ۰ تا ۱۰۰ -- چقدر جواب دقیقاً از شواهد پشتیبانی می‌شه>,
  "relevance_score": <عدد صحیح ۰ تا ۱۰۰ -- چقدر جواب واقعاً همون سوال کاربر رو جواب می‌ده، نه یک موضوع نزدیک/کلی>,
  "confidence_score": <عدد صحیح ۰ تا ۱۰۰ -- خودت چقدر مطمئنی که شواهد موجود برای این نتیجه‌گیری کافی و بدون‌ابهامه>,
  "warnings": ["<هر ادعای عددی یا علّی در جواب که مستقیم از شواهد پشتیبانی نمی‌شه>"]
}

قوانین امتیازدهی faithfulness_score:
- ۱۰۰ یعنی هر ادعای جواب مستقیم از شواهد قابل‌استخراجه.
- هر ادعای عددی/آماری که در شواهد نیست، امتیاز رو به‌طور محسوس کم کن.
- هر رابطه‌ی علّی ("چون X، پس Y") که شواهد فقط هم‌بستگی نشون می‌ده نه
  علیت، امتیاز رو کم کن.
- اگه هیچ ابزاری صدا زده نشده ولی جواب مدعی داده‌ی خاصیه، امتیاز خیلی
  پایین (زیر ۳۰) بده.
- اگه جواب کاملاً بر اساس شواهد موجوده -> warnings خالی، grounded=true،
  faithfulness_score نزدیک ۱۰۰.
- اگه faithfulness_score زیر ۷۰ باشه، warnings هرگز نباید خالی بمونه --
  حتماً حداقل یک ادعای مشخص (یا نبودِ کلی شواهدِ کافی) رو در warnings
  بنویس، وگرنه correct_answer نمی‌فهمه دقیقاً چیو باید اصلاح کنه.

قوانین امتیازدهی relevance_score (مستقل از faithfulness):
- اگه جواب دقیقاً به همون چیزی که کاربر پرسیده جواب بده، نزدیک ۱۰۰.
- اگه بخشی از سوال بی‌جواب مونده، یا جواب یک موضوع نزدیک/جانبی رو پوشش
  داده نه دقیقاً همون سوال، یا خیلی کلی‌گویی کرده به‌جای پاسخ دقیق،
  امتیاز رو محسوس کم کن.
- توجه: یک جواب می‌تونه کاملاً faithful (درست و مستند) باشه ولی relevance
  پایینی داشته باشه (مثلاً به سوال دیگه‌ای جواب داده)، یا برعکس.

قوانین امتیازدهی confidence_score (مستقل از faithfulness):
- این محور یعنی «آیا خودِ شواهد موجود، صرف‌نظر از اینکه جواب دقیقاً روشون
  سوار شده یا نه، برای این نتیجه‌گیری کافی/بدون‌ابهامه؟».
- اگه شواهد کامل، بدون تناقض داخلی، و حجم نمونه‌شون کافیه -> امتیاز بالا.
- اگه شواهد ناقصه (مثلاً فقط بخشی از بازه‌ی زمانی پوشش داده شده)، حجم
  نمونه کمه، یا بین ابزارهای مختلف (مثلاً SQL و RAG) تناقض هست -> امتیاز
  رو کم کن.
"""

CORRECTION_SYSTEM_PROMPT = """
تو داری یک جواب نهایی رو که ممیزی نشون داده بخشی از ادعاهاش بی‌پایه‌ست،
اصلاح می‌کنی. فقط یک JSON با این فرمت برگردون -- هیچ متن اضافه‌ای ننویس:

{"corrected_answer": "<جواب اصلاح‌شده>"}

قوانین:
- فقط ادعاهایی که در warnings مشخص شدن رو اصلاح کن؛ بقیه‌ی جواب رو تا
  حد امکان دست‌نخورده نگه دار.
- اگه شواهد کافی برای یک ادعا نبود، صریح بگو داده کافی نیست -- بی‌سروصدا
  حذفش نکن و چیز جدیدی هم اختراع نکن.
- خروجی باید همچنان فارسی، روان، و در قالب یک پاسخ مدیریتی باشه -- نه
  یک لیست تغییرات یا توضیح اینکه چی عوض شده.
"""


def _summarize_trace(tool_trace: list[dict[str, Any]]) -> str:
    if not tool_trace:
        return "(هیچ ابزاری صدا زده نشد)"
    lines = []
    for t in tool_trace:
        status = "موفق" if t.get("ok") else "خطا"
        lines.append(f"- ابزار {t.get('tool')} ({status}): {t.get('summary')}")
    return "\n".join(lines)


def validate_answer(
    final_answer: str,
    tool_trace: list[dict[str, Any]],
    question: str = "",
) -> dict[str, Any]:
    if not VALIDATION_ENABLED:
        return {"skipped": True, "reason": "ENABLE_ANSWER_VALIDATION=false"}

    if not final_answer or not final_answer.strip():
        return {
            "grounded": False,
            "faithfulness_score": 0,
            "relevance_score": 0,
            "confidence_score": 0,
            "warnings": ["جواب نهایی خالی بود."],
        }

    user_prompt = (
        f"سوال کاربر:\n{question}\n\n"
        f"جواب نهایی:\n{final_answer}\n\n"
        f"خلاصه‌ی شواهد خام:\n{_summarize_trace(tool_trace)}"
    )

    try:
        result = call_llm_json(VALIDATION_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001 - ممیزی نباید کل جواب رو خراب کنه
        logger.warning("validate_answer: LLM call failed: %s", exc)
        # همه‌ی امتیازها رو عمداً None می‌ذاریم (نه ۰ و نه ۱۰۰) تا
        # route_after_validate بفهمه این "امتیاز پایین" نیست، بلکه
        # "امتیازی نداریم" -- و در نتیجه سراغ اصلاح نره (فیل-سیف).
        return {
            "grounded": None,
            "faithfulness_score": None,
            "relevance_score": None,
            "confidence_score": None,
            "warnings": [],
            "error": f"validate_answer: {exc}",
        }

    return {
        "grounded": result.get("grounded"),
        "faithfulness_score": result.get("faithfulness_score"),
        "relevance_score": result.get("relevance_score"),
        "confidence_score": result.get("confidence_score"),
        "warnings": result.get("warnings", []),
    }


def correct_answer(
    question: str,
    final_answer: str,
    warnings: list[str],
    tool_trace: list[dict[str, Any]],
) -> str:
    """
    یک‌بار (و فقط یک‌بار -- نگاه کن به graph.py) جواب رو بازنویسی می‌کنه.
    اگه خودِ تماس اصلاح هم شکست بخوره، جواب اصلی/اولیه رو برمی‌گردونه --
    شکست در اصلاح نباید باعث بشه کاربر هیچ جوابی نگیره.
    """
    user_prompt = (
        f"سوال کاربر: {question}\n\n"
        f"جواب فعلی:\n{final_answer}\n\n"
        f"هشدارهای ممیزی:\n" + "\n".join(f"- {w}" for w in warnings) +
        f"\n\nخلاصه‌ی شواهد خام:\n{_summarize_trace(tool_trace)}"
    )

    try:
        result = call_llm_json(CORRECTION_SYSTEM_PROMPT, user_prompt)
        corrected = result.get("corrected_answer")
        return corrected if corrected and corrected.strip() else final_answer
    except Exception as exc:  # noqa: BLE001
        logger.warning("correct_answer: LLM call failed, جواب اصلی حفظ می‌شه: %s", exc)
        return final_answer
