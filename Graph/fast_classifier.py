## PATH: app/graph/fast_classifier.py
"""
ماژول اختصاصی برای مدل‌های تصمیم‌گیری سبک و فوق‌سریع (System 1 Decision Models)
مانند Jev (از طریق OpenRouter Decisions API) یا Laya (محلی).

این ماژول وظایف طبقه‌بندی بولین، روتینگ و امتیازدهی را در ۳۰ الی ۵۰ میلی‌ثانیه
بدون نیاز به تولید متن (non-autoregressive) انجام می‌دهد.

ویژگی‌های کلیدی:
1. کاملاً مستقل از ساختار گراف و به صورت Drop-in Replacement.
2. با متغیر محیطی USE_FAST_CLASSIFIER=true قابل فعال‌سازی است.
3. در صورت خاموش بودن، نبود کلید یا خطای شبکه، به صورت fail-open مقدار None برمی‌گرداند
   تا سیستم بلافاصله روی روش استاندارد قبلی (LLM سنگین) سوییچ کند و هیچ خطایی متوجه کاربر نشود.
"""
from __future__ import annotations

import logging
import os
from typing import Any
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# فلگ اصلی فعال‌سازی
USE_FAST_CLASSIFIER = os.getenv("USE_FAST_CLASSIFIER", "true").strip().lower() in ("1", "true", "yes")

# تنظیمات اتصال
FAST_CLASSIFIER_PROVIDER = os.getenv("FAST_CLASSIFIER_PROVIDER", "openrouter").strip().lower()
FAST_CLASSIFIER_API_KEY = (
    os.getenv("FAST_CLASSIFIER_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("API_KEY", "")
).strip()

FAST_CLASSIFIER_MODEL = os.getenv("FAST_CLASSIFIER_MODEL", "~typesafe/jev-latest").strip()
FAST_CLASSIFIER_BASE_URL = os.getenv(
    "FAST_CLASSIFIER_BASE_URL",
    "https://openrouter.ai/api/alpha/decisions",
).strip()
FAST_CLASSIFIER_TIMEOUT = float(os.getenv("FAST_CLASSIFIER_TIMEOUT", "25.0"))


def call_decision_api(
    state: dict[str, Any],
    questions: dict[str, Any],
) -> dict[str, Any] | None:
    """
    فراخوانی مستقیم Decisions API (مثل Jev روی OpenRouter).
    در صورت هرگونه خطا یا غیرفعال بودن، None برمی‌گرداند (fail-open).
    """
    if not USE_FAST_CLASSIFIER:
        return None

    if not FAST_CLASSIFIER_API_KEY:
        logger.debug("Fast classifier: کلید API مشخص نشده است؛ استفاده از مسیر پیش‌فرض LLM.")
        return None

    payload = {
        "model": FAST_CLASSIFIER_MODEL,
        "state": state,
        "questions": questions,
    }

    headers = {
        "Authorization": f"Bearer {FAST_CLASSIFIER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            FAST_CLASSIFIER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=FAST_CLASSIFIER_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("answers")
        
        logger.warning(
            "Fast classifier: درخواست با کد %d ناموفق بود: %s",
            response.status_code,
            response.text[:200],
        )
        return None
    except Exception as exc:  # noqa: BLE001 - fail-open intentional
        logger.warning("Fast classifier: خطا در ارتباط با سرویس: %s", exc)
        return None


def fast_is_follow_up(question: str, previous_answer: str) -> bool | None:
    """
    تشخیص سریع اینکه آیا سوال کاربر ادامه‌ی پاسخ قبلی است یا خیر.
    خروجی:
        - True / False در صورت تشخیص موفق توسط مدل سبک
        - None در صورت غیرفعال بودن یا خطا (برای fallback به LLM عادی)
    """
    if not question or not previous_answer:
        return None

    state = {
        "last_answer": previous_answer[:1500],
        "new_question": question[:500],
    }
    questions = {
        "is_follow_up": {
            "type": "noul",
            "instructions": (
                "Does the new question directly continue, refer to, or follow up on the previous assistant answer? "
                "Return true if it implicitly or explicitly refers to or builds upon that previous answer "
                "(e.g. asking why, asking for a chart, asking to summarize it, or referring to the same product/brand). "
                "Return false if it is a completely independent topic or question."
            ),
        }
    }

    answers = call_decision_api(state, questions)
    if not answers or "is_follow_up" not in answers:
        return None

    res = answers["is_follow_up"]
    # Jev مقدار احتمال noul (بین 0.0 تا 1.0) برمی‌گرداند
    noul_prob = float(res.get("noul", 0.0))
    is_fu = noul_prob >= 0.5
    logger.info("Fast classifier follow-up check: prob=%.2f -> %s", noul_prob, is_fu)
    return is_fu


def fast_is_multi_question(question: str) -> bool | None:
    """
    فیلتر سریع برای تشخیص اولیه سوالات چندبخشی مستقل.
    خروجی:
        - False: با قطعیت بالا تک‌بخشی است (مسیر مستقیم سریع بدون تماس LLM)
        - True: احتمالاً چندبخشی است (نیاز به تفکیک و بازنویسی توسط LLM)
        - None: در صورت غیرفعال بودن یا خطا (fallback به مسیر عادی)
    """
    if not question or not question.strip():
        return False

    state = {"question": question[:800]}
    questions = {
        "question_type": {
            "type": "choice",
            "instructions": "Classify whether the user prompt contains a single question or multiple distinct questions.",
            "criteria": {
                "single": "A single question or inquiry about one topic, product, or metric (even with multiple adjectives).",
                "multi": "Contains two or more distinct, independent questions or requests that should be split.",
            },
        }
    }

    answers = call_decision_api(state, questions)
    if not answers or "question_type" not in answers:
        return None

    res = answers["question_type"]
    choice = res.get("choice")
    is_multi = choice == "multi"
    confidence = res.get("confidence", 0.0)
    logger.info("Fast classifier multi-question check: choice=%s (conf=%.2f) -> is_multi=%s", choice, confidence, is_multi)
    return is_multi


def fast_validate_answer(
    question: str,
    final_answer: str,
    evidence_summary: str,
) -> dict[str, Any] | None:
    """
    ممیزی و اعتبارسنجی سریع پاسخ نهایی در برابر شواهد ابزارها.
    اگر پاسخ مستند و معتبر بود (امتیاز >= 70)، دیکشنری نتیجه را تحویل می‌دهد.
    اگر پاسخ دارای تناقض بود یا خطایی رخ داد، None برمی‌گرداند تا LLM اصلی
    جزئیات هشدارها (warnings) را استخراج و تصحیح کند.
    """
    if not final_answer or not evidence_summary:
        return None

    state = {
        "question": question[:600],
        "evidence": evidence_summary[:2000],
        "answer": final_answer[:2000],
    }

    questions = {
        "grounded": {
            "type": "noul",
            "instructions": (
                "Is every factual and numeric statement in the answer directly supported by the evidence?"
            ),
        },
        "faithfulness": {
            "type": "score",
            "instructions": "How faithful and grounded is the answer compared to the evidence?",
            "criteria": ["unsupported", "weakly supported", "partially supported", "fully supported"],
        },
        "relevance": {
            "type": "score",
            "instructions": "How relevant is the answer to the user question?",
            "criteria": ["irrelevant", "partially relevant", "relevant", "fully relevant"],
        },
    }

    answers = call_decision_api(state, questions)
    if not answers or "grounded" not in answers or "faithfulness" not in answers:
        return None

    grounded_prob = float(answers["grounded"].get("noul", 0.0))
    
    # مقیاس score از 0.0 تا 3.0 است (چهار گزینه criteria)
    faith_raw = float(answers["faithfulness"].get("score", 0.0))
    faithfulness_score = int(round((faith_raw / 3.0) * 100))

    rel_raw = float(answers.get("relevance", {}).get("score", 2.0))
    relevance_score = int(round((rel_raw / 3.0) * 100))

    is_grounded = grounded_prob >= 0.5 and faithfulness_score >= 70

    logger.info(
        "Fast validator check: grounded_prob=%.2f, faithfulness=%d, relevance=%d",
        grounded_prob,
        faithfulness_score,
        relevance_score,
    )

    if is_grounded:
        return {
            "grounded": True,
            "faithfulness_score": faithfulness_score,
            "relevance_score": relevance_score,
            "confidence_score": int(grounded_prob * 100),
            "warnings": [],
        }

    # در صورتی که نمره پایین باشد، اجازه می‌دهیم LLM اصلی جزئیات هشدارها را تولید کند
    return None
