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

from .llm_client import call_llm_json, CHAT_MODEL
from .fast_classifier import fast_validate_answer

logger = logging.getLogger(__name__)

VALIDATION_ENABLED = os.getenv("ENABLE_ANSWER_VALIDATION", "true").strip().lower() in ("1", "true", "yes")

# اگه faithfulness_score زیر این عدد باشه، correct_answer صدا زده می‌شه.
CORRECTION_THRESHOLD = int(os.getenv("VALIDATION_CORRECTION_THRESHOLD", "70"))

 
VALIDATION_SYSTEM_PROMPT = """
You are an independent auditor. You are given a "user question", a "final
answer", and a summary of "raw evidence" (the actual results of the tools
that were called). Return only a JSON object in this format -- write no
extra text:
 
{
  "grounded": true|false,
  "faithfulness_score": <integer 0-100 -- how precisely the answer is supported by the evidence>,
  "relevance_score": <integer 0-100 -- how much the answer actually answers the user's question, not a nearby/general topic>,
  "confidence_score": <integer 0-100 -- how confident you are that the available evidence is sufficient and unambiguous for this conclusion>,
  "warnings": ["<any numeric or causal claim in the answer that isn't directly supported by the evidence>"]
}
 
Scoring rules for faithfulness_score:
- 100 means every claim in the answer can be directly derived from the
  evidence.
- Any numeric/statistical claim not present in the evidence should
  noticeably lower the score.
- Any causal relationship ("because X, therefore Y") where the evidence
  only shows correlation, not causation, should lower the score.
- If no tool was called at all but the answer claims specific data, give a
  very low score (below 30).
- Conversational greetings, identity introductions ("نام من راهین است"), or explanations of capabilities are derived from system persona, NOT from database tools. If the question is about identity/greeting/capabilities and no database tool was required, give faithfulness_score near 100 with grounded=true and warnings empty.
- If the answer is fully based on the available evidence -> warnings empty,
  grounded=true, faithfulness_score near 100.
- If faithfulness_score is below 70, warnings must never be empty -- always
  write at least one specific claim (or the general lack of sufficient
  evidence) in warnings, otherwise correct_answer won't know exactly what
  to fix.
 
Scoring rules for relevance_score (independent of faithfulness):
- If the answer addresses exactly what the user asked, near 100.
- If part of the question is left unanswered, or the answer covers a
  nearby/tangential topic instead of the exact question, or it's too vague
  instead of a precise answer, noticeably lower the score.
- Note: an answer can be fully faithful (correct and well-supported) but
  have low relevance (e.g. it answered a different question), or vice
  versa.
 
Scoring rules for confidence_score (independent of faithfulness):
- This axis means "is the available evidence itself -- regardless of
  whether the answer is correctly built on it -- sufficient/unambiguous for
  this conclusion?"
- If the evidence is complete, has no internal contradictions, and has a
  sufficient sample size -> high score.
- If the evidence is incomplete (e.g. only part of the time range is
  covered), the sample size is small, or there's a contradiction between
  different tools (e.g. SQL and RAG) -> lower the score.
"""
 
CORRECTION_SYSTEM_PROMPT = """
You are correcting a final answer that the audit showed has some
unsupported claims. Return only a JSON object in this format -- write no
extra text:
 
{"corrected_answer": "<the corrected answer>"}
 
Rules:
- Only fix the claims identified in warnings; leave the rest of the answer
  as unchanged as possible.
- If there isn't enough evidence for a claim, explicitly say the data is
  insufficient -- don't silently remove it, and don't invent anything new.
- The output must still be Persian, fluent, and in the form of a
  managerial answer -- not a list of changes or an explanation of what
  changed.
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

    # سوالات هویتی، احوال‌پرسی یا قابلیت‌های سیستم که نیازی به ابزارهای دیتابیس ندارند
    if not tool_trace:
        q_norm = (question or "").strip().lower()
        identity_keywords = (
            "اسم", "نام", "کیستی", "کی هستی", "چیستی", "چه کاره", "چکار", "سلام", "درود", "خوبی", "قابلیت"
        )
        if any(kw in q_norm for kw in identity_keywords):
            logger.info("validate_answer: سوال هویتی/معرفی تشخیص داده شد؛ تایید بدون نیاز به ابزار دیتابیس.")
            return {
                "grounded": True,
                "faithfulness_score": 100,
                "relevance_score": 100,
                "confidence_score": 100,
                "warnings": [],
            }

    evidence_summary = _summarize_trace(tool_trace)

    # ممیزی سریع ۳۰ میلی‌ثانیه‌ای با مدل تصمیم‌گیری سبک (Decision Model)
    fast_val = fast_validate_answer(question, final_answer, evidence_summary)
    if fast_val is not None:
        logger.info(
            "validate_answer: ممیزی سریع با موفقیت انجام شد (faithfulness=%s, relevance=%s)",
            fast_val.get("faithfulness_score"),
            fast_val.get("relevance_score"),
        )
        return fast_val

    user_prompt = (
        f"سوال کاربر:\n{question}\n\n"
        f"جواب نهایی:\n{final_answer}\n\n"
        f"خلاصه‌ی شواهد خام:\n{evidence_summary}"
    )

    try:
        result = call_llm_json(VALIDATION_SYSTEM_PROMPT, user_prompt, model=CHAT_MODEL)
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
        result = call_llm_json(CORRECTION_SYSTEM_PROMPT, user_prompt, model=CHAT_MODEL)
        corrected = result.get("corrected_answer")
        return corrected if corrected and corrected.strip() else final_answer
    except Exception as exc:  # noqa: BLE001
        logger.warning("correct_answer: LLM call failed, جواب اصلی حفظ می‌شه: %s", exc)
        return final_answer
