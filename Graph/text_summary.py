## PATH: app/graph/text_summary.py
"""
خلاصه‌سازی استخراجی و کاملاً بدون LLM برای نظرات مشتری‌ها.

الگوریتم (کلاسیک، شبیه Luhn's method برای خلاصه‌سازی استخراجی):
    ۱. تمام نظرات رو توکنایز می‌کنیم و فرکانس کلمات (بدون stopword) رو
       می‌شماریم.
    ۲. هر نظر رو بر اساس میانگین فرکانس کلماتش امتیاز می‌دیم -- نظری که
       کلمات پرتکرارتر (یعنی نزدیک‌تر به موضوع اصلیِ مشترک بین همه‌ی
       نظرات) داره، امتیاز بالاتری می‌گیره.
    ۳. پرتکرارترین کلمات (به‌عنوان "مضامین تکرارشونده") و بالاترین‌
       امتیاز نظرات (به‌عنوان "نظرات نماینده") رو برمی‌گردونیم.

هیچ تماس LLM/API خارجی این‌جا نیست -- کاملاً محلی و رایگان. run_rag_tool
(در vector_retriever.py) این تابع رو روی نتایج خام جست‌وجوی برداری صدا
می‌زنه، قبل از این‌که چیزی به Agent برسه -- دقیقاً طبق همون چیزی که
خواسته شده بود: "قبل از رسیدن به LLM" خلاصه بشه.

محدودیت شناخته‌شده: توکنایزر فقط بر پایه‌ی regex ساده‌ست (بدون ریشه‌یابی/
نرمال‌سازی زبان‌شناختی فارسی مثل hazm) -- برای شروع کافیه، ولی اگه دقت
مهم شد می‌شه بعداً با hazm جایگزینش کرد بدون تغییر امضای تابع.
"""
from __future__ import annotations

import re
from collections import Counter

# لیست کوتاه و عمداً محافظه‌کارانه‌ی stopword های فارسی -- هدف حذف حروف
# ربط/اضافه‌ی بسیار پرتکراره، نه یک لیست کامل زبان‌شناختی.
PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "برای", "هم", "تا",
    "یک", "شود", "شد", "است", "بود", "می", "ها", "های", "رو", "یا",
    "اما", "ولی", "نیز", "دیگر", "هر", "همه", "بی", "بر", "آن", "اگر",
    "چون", "روی", "بین", "بعد", "قبل", "کرد", "کردم", "کردیم", "شده",
    "نمی", "خیلی", "چه", "من", "ما", "شما", "او", "آنها", "خود", "کنید",
}

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in PERSIAN_STOPWORDS]

def _normalize_comment(item: str | dict) -> dict:
    """
    ورودی summarize_comments می‌تونه یا رشته‌ی خام باشه (سازگاری با
    فراخوان‌های قدیمی مثل test_summary_manual.py) یا دیکشنری با متادیتا
    (comment_id, rate, likes, dislikes, ...). این تابع هر دو حالت رو به
    یک شکل یکسان تبدیل می‌کنه: یک دیکشنری که حداقل کلید "text" رو داره.
    """
    if isinstance(item, str):
        return {"text": item}
    return item

def summarize_comments(
    comments: list[str] | list[dict],
    top_n_keywords: int = 15,
    top_n_representative: int = 6,
) -> dict:
    """
    ورودی: لیست متن خام نظرات (list[str]) یا لیست دیکشنری با متادیتا
    (list[dict] -- هر دیکشنری حداقل کلید "text" رو باید داشته باشه؛ هر
    کلید دیگه‌ای مثل comment_id/rate/likes بدون تغییر همراهش نگه داشته
    می‌شه).
    خروجی: {"top_keywords": [...], "representative_comments": [...]}
    -- representative_comments همیشه لیستی از دیکشنریه (حتی اگه ورودی
    رشته‌ی خام بوده باشه)، تا فراخوان‌ها یکدست بمونن.
    """
    if not comments:
        return {"top_keywords": [], "representative_comments": []}

    normalized = [_normalize_comment(c) for c in comments]

    freq: Counter[str] = Counter()
    tokenized_comments: list[list[str]] = []
    for c in normalized:
        tokens = _tokenize(c["text"])
        tokenized_comments.append(tokens)
        freq.update(tokens)

    top_keywords = [word for word, _ in freq.most_common(top_n_keywords)]

    scored: list[tuple[float, dict]] = []
    for c, tokens in zip(normalized, tokenized_comments):
        if not tokens:
            continue
        score = sum(freq[t] for t in tokens) / len(tokens)
        scored.append((score, c))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    representative: list[dict] = []
    seen: set[str] = set()
    for _, c in scored:
        text = c["text"]
        if text in seen:
            continue
        seen.add(text)
        representative.append(c)
        if len(representative) >= top_n_representative:
            break

    return {
        "top_keywords": top_keywords,
        "representative_comments": representative,
    }