## PATH: app/graph/audit.py
"""
بررسی و اصلاحِ نرم (soft) جواب نهایی -- جبران بخشی از قابلیت ممیزی‌ای که
با حذف نودهای صریح evidence_fusion/answer_validator (نسخه‌ی پایپ‌لاین
قبلی) از دست رفت.

دو تابع اینجاست:
    validate_answer  -- جواب نهایی رو در برابر خلاصه‌ی شواهد خام
                         (tool_trace) می‌سنجه و یک "match_score" عددی
                         (۰ تا ۱۰۰) + لیست هشدارها برمی‌گردونه.
    correct_answer    -- وقتی match_score پایینه، جواب رو یک‌بار (نه در
                         حلقه!) بازنویسی می‌کنه تا هشدارها رفع بشن.

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

# اگه match_score زیر این عدد باشه، correct_answer صدا زده می‌شه.
CORRECTION_THRESHOLD = int(os.getenv("VALIDATION_CORRECTION_THRESHOLD", "70"))

VALIDATION_SYSTEM_PROMPT = """
تو یک ممیز مستقل هستی. یک "جواب نهایی" و خلاصه‌ای از "شواهد خام" (نتایج
واقعی ابزارهایی که صدا زده شدن) رو می‌گیری. فقط یک JSON با این فرمت
برگردون -- هیچ متن اضافه‌ای ننویس:

{
  "grounded": true|false,
  "match_score": <عدد صحیح ۰ تا ۱۰۰ -- چقدر جواب دقیقاً از شواهد پشتیبانی می‌شه>,
  "warnings": ["<هر ادعای عددی یا علّی در جواب که مستقیم از شواهد پشتیبانی نمی‌شه>"]
}

قوانین امتیازدهی match_score:
- ۱۰۰ یعنی هر ادعای جواب مستقیم از شواهد قابل‌استخراجه.
- هر ادعای عددی/آماری که در شواهد نیست، امتیاز رو به‌طور محسوس کم کن.
- هر رابطه‌ی علّی ("چون X، پس Y") که شواهد فقط هم‌بستگی نشون می‌ده نه
  علیت، امتیاز رو کم کن.
- اگه هیچ ابزاری صدا زده نشده ولی جواب مدعی داده‌ی خاصیه، امتیاز خیلی
  پایین (زیر ۳۰) بده.
- اگه جواب کاملاً بر اساس شواهد موجوده -> warnings خالی، grounded=true،
  match_score نزدیک ۱۰۰.
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


def validate_answer(final_answer: str, tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not VALIDATION_ENABLED:
        return {"skipped": True, "reason": "ENABLE_ANSWER_VALIDATION=false"}

    if not final_answer or not final_answer.strip():
        return {"grounded": False, "match_score": 0, "warnings": ["جواب نهایی خالی بود."]}

    user_prompt = f"جواب نهایی:\n{final_answer}\n\nخلاصه‌ی شواهد خام:\n{_summarize_trace(tool_trace)}"

    try:
        result = call_llm_json(VALIDATION_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001 - ممیزی نباید کل جواب رو خراب کنه
        logger.warning("validate_answer: LLM call failed: %s", exc)
        # match_score رو عمداً None می‌ذاریم (نه ۰ و نه ۱۰۰) تا
        # route_after_validate بفهمه این "امتیاز پایین" نیست، بلکه
        # "امتیازی نداریم" -- و در نتیجه سراغ اصلاح نره (فیل-سیف).
        return {"grounded": None, "match_score": None, "warnings": [], "error": f"validate_answer: {exc}"}

    return {
        "grounded": result.get("grounded"),
        "match_score": result.get("match_score"),
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
