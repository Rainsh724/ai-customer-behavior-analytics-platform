## PATH: app/graph/audit.py
"""
بررسی نرم (soft) صحت‌سنجی جواب نهایی -- جبران بخشی از قابلیت ممیزی‌ای که
با حذف نودهای صریح evidence_fusion/answer_validator (نسخه‌ی پایپ‌لاین
قبلی) از دست رفت.

این یک تماس LLM جدا و اضافیه که جواب نهایی رو در برابر خلاصه‌ی نتایج خام
ابزارهایی که واقعاً صدا زده شدن (tool_trace) می‌سنجه:
    - آیا عدد/آماری در جواب اومده که در شواهد نبوده؟
    - آیا یک رابطه‌ی علّی ادعا شده که شواهد فقط هم‌بستگی نشون می‌ده؟

مهم: این بررسی جواب رو تغییر نمی‌ده و مانع ارسالش به کاربر نمی‌شه --
فقط در state["validation"] برای لاگ/ممیزی بعدی ذخیره می‌شه. چون هزینه‌ی
یک تماس LLM اضافی به ازای هر سوال داره، با متغیر محیطی
ENABLE_ANSWER_VALIDATION قابل خاموش‌کردنه (پیش‌فرض: روشن).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .llm_client import call_llm_json

logger = logging.getLogger(__name__)

VALIDATION_ENABLED = os.getenv("ENABLE_ANSWER_VALIDATION", "true").strip().lower() in ("1", "true", "yes")

VALIDATION_SYSTEM_PROMPT = """
تو یک ممیز مستقل هستی. یک "جواب نهایی" و خلاصه‌ای از "شواهد خام" (نتایج
واقعی ابزارهایی که صدا زده شدن) رو می‌گیری. فقط یک JSON با این فرمت
برگردون -- هیچ متن اضافه‌ای ننویس:

{
  "grounded": true|false,
  "warnings": ["<هر ادعای عددی یا علّی در جواب که مستقیم از شواهد پشتیبانی نمی‌شه>"]
}

قوانین:
- اگه جواب یک عدد/آمار دقیق آورده که در شواهد نیست -> warning.
- اگه جواب یک رابطه‌ی علّی ("چون X، پس Y") ادعا کرده ولی شواهد فقط
  هم‌بستگی/هم‌زمانی نشون می‌ده نه علیت -> warning.
- اگه هیچ ابزاری صدا زده نشده ولی جواب مدعی داده‌ی خاصیه -> warning.
- اگه جواب کاملاً بر اساس شواهد موجوده -> warnings خالی و grounded=true.
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
        return {"grounded": False, "warnings": ["جواب نهایی خالی بود."]}

    user_prompt = f"جواب نهایی:\n{final_answer}\n\nخلاصه‌ی شواهد خام:\n{_summarize_trace(tool_trace)}"

    try:
        result = call_llm_json(VALIDATION_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001 - ممیزی نباید کل جواب رو خراب کنه
        logger.warning("validate_answer: LLM call failed: %s", exc)
        return {"grounded": None, "warnings": [], "error": f"validate_answer: {exc}"}

    return {
        "grounded": result.get("grounded"),
        "warnings": result.get("warnings", []),
    }
