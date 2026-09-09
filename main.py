## PATH: app/main.py
from __future__ import annotations

from typing import Any
import json

from Graph.graph import get_graph
from Graph.dataset_time import get_reference_date
from Graph.llm_client import preload_embedding_model
import memory_store

# ============================================================
# پرامپت سیستمی Agent -- شخصیت "مدیر ارشد" طبق سند معماری.
# ============================================================

AGENT_SYSTEM_PROMPT = """
تو Agent ارشد یک سیستم تحلیل هوشمند کسب‌وکار هستی. وظیفه‌ات تحلیل
داده‌های فروشگاه آنلاین و ارائه‌ی پاسخ دقیق، فارسی و مدیریتی است.

ابزارها:
- tool_sql: اجرای SQL معتبر PostgreSQL برای تحلیل عددی و ساختاری.
- tool_rag: جست‌وجوی معنایی در نظرات مشتریان با search_topic و
  در صورت نیاز product_id؛ بدون metadata filtering.
- tool_chart: ساخت نمودار از SQL، فقط وقتی کاربر صریحاً نمودار بخواهد.

قوانین:

۱. حافظه و Follow-up
ابتدا تاریخچه‌ی همین مکالمه را بررسی کن. اگر پاسخ یا داده‌ی لازم قبلاً
در پیام‌ها یا نتایج ابزارها وجود دارد، دوباره همان داده را محاسبه نکن.

برای سوال‌های کوتاه مثل «چرا؟»، «دلیلش؟»، «چطور؟»، «همون محصول؟» و
مشابه آن، منظور را از آخرین نتیجه‌ی معتبر مکالمه استخراج کن.

در Follow-up باید این موارد را حفظ کنی:
- محصول و product_id
- metric
- بازه‌ی زمانی
- نتیجه‌ی قبلی

مثلاً اگر قبلاً SQL مشخص کرده محصول X در یک بازه پرفروش‌ترین بوده و
کاربر می‌پرسد «چرا؟»، محصول را عوض نکن، ranking جدید انجام نده و
بازه را تغییر نده. برای یافتن دلیل، مستقیماً سراغ RAG همان محصول برو.

۲. ترتیب ابزارها
سوال عددی/آماری → فقط SQL.

سوال ترکیبی عددی + کیفی → ابتدا SQL برای بخش عددی، سپس در صورت نیاز RAG.

سوال علّی درباره‌ی افزایش/کاهش/افت/رشد → ابتدا فقط SQL.
فقط اگر SQL تغییر موردنظر را واقعاً تأیید کرد، RAG را اجرا کن.

برای سوال‌هایی مثل «کدام محصول پرفروش‌تر بوده و چرا؟»:
SQL → تعیین محصول و product_id → RAG همان محصول.

SQL و RAG را برای یک سوال علّی به‌صورت هم‌زمان اجرا نکن.

۳. تعریف فروش
«پرفروش‌ترین»، «بیشترین فروش» و «محصولات پرفروش» به‌صورت پیش‌فرض
یعنی بیشترین تعداد خرید.

در user_behavior_logs هر purchase یک رخداد خرید است؛ بنابراین معیار
پیش‌فرض:

COUNT(*) AS units_sold

و ranking:

ORDER BY units_sold DESC

اگر کاربر صریحاً «مبلغ فروش»، «درآمد» یا «ارزش فروش» خواست، مبلغ را
محاسبه کن.

products.price قیمت فعلی محصول است؛ بنابراین price * units_sold
فقط estimated_sales است و نباید بدون توضیح به‌عنوان درآمد واقعی
تاریخی معرفی شود.


برای جلوگیری از fan-out، ابتدا purchaseها را بر اساس product_id
تجمیع کن و سپس به products JOIN شو. هرگز SUM(products.price) را
مستقیماً روی JOIN با purchase events اجرا نکن.

هر وقت ORDER BY + LIMIT برای رتبه‌بندی/انتخاب top-N استفاده می‌شود
(مثلاً «پرفروش‌ترین ۱۰ محصول»)، حتماً یک تای‌بریک قطعی (مثل
product_id ASC) به‌عنوان کلید دوم ORDER BY اضافه کن، حتی اگر معیار
اصلی units_sold/COUNT باشد. بدون این کار، وقتی چند محصول امتیاز
برابر دارند (tie)، هر اجرای جدید -- از جمله وقتی tool_chart همان
رتبه‌بندی را برای رسم نمودار دوباره می‌سازد -- می‌تواند ست متفاوتی
از محصولات هم‌امتیاز را برگرداند و باعث شود جدول SQL و نمودار برای
همان سؤال، محصولات متفاوتی نشان دهند. اگر کاربر پرسید چرا SQL و
نمودار نتایج متفاوتی دارند، همین علت (نبود tie-breaker) را به‌عنوان
دلیل احتمالی در نظر بگیر، نه یک اختلاف دیتای واقعی.

هر وقت رتبه‌بندی بر اساس یک نسبت/میانگین/درصد است (مثل conversion_rate،
avg_negative_pct، avg_rating)، نه یک شمارش خام، حتماً یک حداقل حجم
نمونه (مثلاً view_cnt >= 30 یا comment_cnt >= 5، بسته به سؤال) در
WHERE اعمال کن. بدون این فیلتر، محصولاتی با تعداد بازدید/نظر بسیار
کم (مثلاً ۱ بازدید یا ۱ نظر) به‌راحتی به مقادیر افراطی ۰٪ یا ۱۰۰٪
می‌رسند و رتبه‌بندی را با نویز آماری (نه سیگنال واقعی کسب‌وکار) پر
می‌کنند. اگر چنین فیلتری اعمال کردی، حتماً در پاسخ نهایی ذکر کن که
نتایج به محصولات با حداقل فلان مقدار بازدید/نظر محدود شده است.

۴. زمان
تاریخ مرجع تمام محاسبات نسبی همان DATASET REFERENCE DATE موجود در
system prompt است؛ از تاریخ واقعی امروز، NOW() یا CURRENT_DATE استفاده نکن.

«۷ روز اخیر»، «۳۰ روز اخیر»، «۳ ماه اخیر»، «۶ ماه اخیر» و مشابه آن
rolling نسبت به تاریخ مرجع هستند و طول بازه باید دقیقاً از عبارت
کاربر گرفته شود.

«ماه اخیر» = rolling یک ماه اخیر، نه ماه تقویمی قبلی.
«ماه قبل/ماه گذشته» در صورت اشاره‌ی تقویمی = ماه تقویمی قبلی.

اگر کاربر تاریخ دقیق داد، دقیقاً همان بازه را استفاده کن.

هرگز تاریخ شروع/پایان دقیق را خودت (در ذهن/متن) محاسبه نکن. همیشه
در خودِ SQL، عبارت را به‌صورت نسبی به تاریخ مرجع لفظی بنویس و بگذار
PostgreSQL محاسبه کند -- مثلاً به‌جای نوشتن مستقیم '2022-09-01'،
بنویس '<تاریخ مرجع>'::date - INTERVAL '6 months'. فقط محاسبه‌ی
PostgreSQL معتبر است، نه محاسبه‌ی دستی خودت.

تمام بازه‌ها با قرارداد [start, end) ساخته شوند: >= start AND < end.
وقتی end = تاریخ مرجع است (یعنی بازه باید تا خودِ تاریخ مرجع، شامل
همان روز، را پوشش دهد)، end واقعی در SQL باید
'<تاریخ مرجع>'::date + INTERVAL '1 day' باشد، نه خودِ تاریخ مرجع؛
وگرنه رویدادهای روز مرجع به‌اشتباه از بازه حذف می‌شوند. این قاعده را
در همه‌ی کوئری‌های مرتبط با یک سؤال (مثلاً هم در tool_sql و هم در
tool_chart برای همان بازه) یکسان اعمال کن.

در پاسخ نهایی، بازه را از SQL واقعی اجراشده استخراج کن؛ نه از حافظه
یا محاسبه‌ی مجدد. توجه کن که چون بازه [start, end) است، اگر SQL مثلاً:
timestamp >= '2022-12-01'
AND timestamp < '2023-03-02'
باشد، آخرین روز گزارش‌شده 2023-03-01 است، نه 2023-03-02 (چون end
همیشه exclusive است).

۵. سوالات علّی
برای «چرا فروش/امتیاز/بازدید X کم یا زیاد شده؟»:

الف) ابتدا SQL و محاسبه‌ی صریح دوره‌ی فعلی و دوره‌ی مقایسه‌ای.
ب) اگر SQL برای اثبات تغییر کافی نبود، SQL اصلاح‌شده اجرا کن.
ج) اگر تغییر تأیید نشد، متوقف شو و بگو داده‌ها ادعا را تأیید نمی‌کنند؛
   RAG اجرا نکن.
د) اگر تغییر تأیید شد، RAG را برای شواهد کیفی اجرا کن.
   کاهش → search_topic در جهت نارضایتی/شکایت.
   افزایش → search_topic در جهت رضایت/استقبال.
هـ) اگر product_id معتبر داری، حتماً همان product_id را ارسال کن.
و) مقدار و درصد تغییر را از SQL و مضامین کیفی را از RAG جدا کن.
   همبستگی را علت قطعی معرفی نکن.

اگر RAG شواهد کافی برای علت نداشت، صریحاً بگو شواهد برای تعیین علت
قطعی کافی نیست.

۶. محدودیت RAG
وقتی product_id مشخص است، آن محصول مرجع اصلی است.

اگر RAG با همان product_id مقدار hit_count=0 برگرداند:
- product_id را حذف نکن.
- برای پیدا کردن «evidence جایگزین» جست‌وجوی عمومی انجام نده.
- نظرات محصولات دیگر یا category-level را به محصول اصلی نسبت نده.
- نتیجه را به‌عنوان «شواهد مستقیم کافی پیدا نشد» گزارش کن.

فقط اگر صریحاً تصمیم گرفتی از context سطح دسته استفاده کنی، آن را
category-level context بنام، نه evidence مستقیم محصول.

۷. خطای ابزار
اگر ابزار خطا داد، با یک tool_call اصلاح‌شده دوباره تلاش کن.
موفق شدن SQL به‌تنهایی کافی نیست؛ نتیجه باید مستقیماً پاسخ سؤال را
پوشش دهد.

اگر چند تلاش ناموفق بود و داده‌ی معتبر به دست نیامد، محدودیت را
شفاف اعلام کن و حدس نزن.

۸. نمودار
tool_chart را فقط وقتی اجرا کن که کاربر صریحاً نمودار، چارت،
داشبورد یا visualization بخواهد.

۹. پاسخ نهایی
پاسخ همیشه فارسی، روان، مختصر و مدیریتی باشد.
دام خام JSON، SQL یا trace ابزارها را نمایش نده.

هرگز داده، علت، محصول، بازه یا نتیجه‌ای را که از ابزارها پشتیبانی
نمی‌شود حدس نزن.
"""

# ============================================================
# قانون ۸ -- فعال شد چون tool_knowledge_base الان با پلیس‌هولدر موقت
# در tools.py وصله (برای دیباگ گراف). وقتی نسخه‌ی واقعی جایگزین شد，
# چیزی در این پرامپت لازم نیست تغییر کنه.
# ============================================================
KNOWLEDGE_BASE_RULE = """
۸. قبل از دادن هرگونه پیشنهاد یا توصیه‌ی مدیریتی (نه فقط گزارش عدد/
   نظرات، بلکه وقتی کاربر می‌خواد بدونه "چیکار کنم؟")، حتماً اول
   tool_knowledge_base رو با موضوع مرتبط صدا بزن و پیشنهادت رو با
   ترکیب اون دانش آموزشی + دانش عمومی خودت بساز -- نه فقط از حافظه‌ی
   خودت. اگه پایگاه‌دانش چیز مرتبطی نداشت، صریح بگو و بر پایه‌ی دانش
   عمومی خودت پیش برو.
"""

AGENT_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT + KNOWLEDGE_BASE_RULE


# ============================================================
# FOLLOW-UP / ACTIVE CONTEXT
# ============================================================

FOLLOW_UP_PHRASES = {
    "چرا",
    "چرا؟",
    "دلیلش",
    "دلیلش؟",
    "دلیلش چیه",
    "دلیلش چیست",
    "چطور",
    "چطور؟",
    "چطور بوده",
    "همون محصول",
    "همون محصول؟",
    "اون محصول",
    "اون محصول؟",
    "منظورت همون محصوله؟",
}


def _normalize_question(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_follow_up_question(question: str) -> bool:
    normalized = _normalize_question(question)

    if normalized in FOLLOW_UP_PHRASES:
        return True

    short_followups = (
        "چرا ",
        "دلیل ",
        "چطور ",
        "همون ",
        "اون ",
        "این ",
    )

    if len(normalized.split()) <= 5:
        return normalized.startswith(short_followups)

    return False


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(
                    str(
                        item.get("text")
                        or item.get("content")
                        or item
                    )
                )
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """
    تلاش سبک برای استخراج JSONهای موجود در پیام‌های tool.

    هدف parser کامل SQL نیست؛ فقط پیدا کردن فیلدهای مهم
    برای active context است.
    """

    if not text:
        return []

    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return [obj]

        if isinstance(obj, list):
            return [
                item
                for item in obj
                if isinstance(item, dict)
            ]

    except Exception:
        pass

    return []


def _extract_active_context(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    آخرین context معتبر مکالمه را از history استخراج می‌کند.

    اولویت:
        1. آخرین SQL tool result
        2. آخرین assistant answer
        3. fallback از متن user

    این تابع برای Follow-up استفاده می‌شود و قرار نیست
    تحلیل جدید انجام دهد.
    """

    context: dict[str, Any] = {
        "is_follow_up": False,
        "product_id": None,
        "product_title": None,
        "metric": None,
        "metric_label": None,
        "period_start": None,
        "period_end": None,
        "period_label": None,
        "previous_result": None,
        "source": None,
    }

    # ---------------------------------------------------------
    # 1. آخرین tool resultهای معتبر
    # ---------------------------------------------------------

    for message in reversed(messages):

        if message.get("role") != "tool":
            continue

        name = message.get("name", "")

        if name != "tool_sql":
            continue

        content = _content_to_text(message.get("content"))

        objects = _extract_json_objects(content)

        for obj in objects:

            rows = obj.get("rows")

            if isinstance(rows, list) and rows:
                first_row = rows[0]

                if isinstance(first_row, dict):

                    # product_id
                    product_id = (
                        first_row.get("product_id")
                        or first_row.get("id")
                    )

                    if product_id is not None:
                        context["product_id"] = product_id

                    # product title
                    product_title = (
                        first_row.get("title_fa")
                        or first_row.get("product_title")
                        or first_row.get("title")
                    )

                    if product_title:
                        context["product_title"] = product_title

                    # metric
                    metric_candidates = (
                        "units_sold",
                        "purchase_cnt",
                        "purchase_count",
                        "sales",
                        "revenue",
                        "value",
                    )

                    for key in metric_candidates:
                        if key in first_row:
                            context["metric"] = key
                            context["previous_result"] = first_row[key]
                            break

            # اگر خود result یک summary داشت
            if obj.get("summary"):
                context["previous_result"] = obj["summary"]

            # period fields
            for key in (
                "period_start",
                "start_date",
                "from_date",
            ):
                if obj.get(key):
                    context["period_start"] = str(obj[key])
                    break

            for key in (
                "period_end",
                "end_date",
                "to_date",
            ):
                if obj.get(key):
                    context["period_end"] = str(obj[key])
                    break

    # ---------------------------------------------------------
    # 2. از assistant answer برای metric/title استفاده کن
    # ---------------------------------------------------------

    for message in reversed(messages):

        if message.get("role") != "assistant":
            continue

        if message.get("tool_calls"):
            continue

        content = _content_to_text(message.get("content"))

        if not content:
            continue

        if context["product_title"] is None:
            # اینجا عمداً title را از متن آزاد استخراج نمی‌کنیم.
            # چون احتمال hallucination وجود دارد.
            pass

        if context["metric"] is None:

            if "تعداد خرید" in content:
                context["metric"] = "units_sold"
                context["metric_label"] = "تعداد خرید"

            elif "فروش" in content:
                context["metric"] = "units_sold"
                context["metric_label"] = "تعداد خرید"

            elif "درآمد" in content:
                context["metric"] = "revenue"
                context["metric_label"] = "درآمد"

        break

    # ---------------------------------------------------------
    # 3. Label metric
    # ---------------------------------------------------------

    metric_labels = {
        "units_sold": "تعداد خرید",
        "purchase_cnt": "تعداد خرید",
        "purchase_count": "تعداد خرید",
        "revenue": "درآمد",
        "sales": "فروش",
        "value": "ارزش فروش",
    }

    if context["metric"]:
        context["metric_label"] = metric_labels.get(
            context["metric"],
            context["metric"],
        )

    # ---------------------------------------------------------
    # 4. آخرین user question
    # ---------------------------------------------------------

    for message in reversed(messages):

        if message.get("role") == "user":
            previous_question = _content_to_text(
                message.get("content")
            )

            if previous_question:
                context["previous_question"] = previous_question

            break

    return context


def _build_follow_up_system_context(
    context: dict[str, Any],
) -> str:
    """
    Context فعال را به‌صورت system message به Agent می‌دهد.

    این بخش عمداً explicit است تا LLM نتواند context را
    به‌صورت دلخواه reinterpret کند.
    """

    product_id = context.get("product_id")
    product_title = context.get("product_title")
    metric = context.get("metric")
    metric_label = context.get("metric_label")
    period_start = context.get("period_start")
    period_end = context.get("period_end")
    period_label = context.get("period_label")
    previous_result = context.get("previous_result")

    lines = [
        "[ACTIVE CONVERSATION CONTEXT]",
        "این context از نتیجه‌ی معتبر قبلی استخراج شده است.",
        "برای Follow-up باید دقیقاً همین context را حفظ کنی.",
    ]

    if product_id is not None:
        lines.append(f"product_id = {product_id}")

    if product_title:
        lines.append(f"product_title = {product_title}")

    if metric:
        lines.append(f"metric = {metric}")

    if metric_label:
        lines.append(f"metric_label = {metric_label}")

    if period_label:
        lines.append(f"period_label = {period_label}")

    if period_start:
        lines.append(f"period_start = {period_start}")

    if period_end:
        lines.append(f"period_end = {period_end}")

    if previous_result is not None:
        lines.append(f"previous_result = {previous_result}")

    lines.extend(
        [
            "",
            "قانون مهم:",
            "اگر سؤال فعلی Follow-up کوتاه است، context بالا را تغییر نده.",
            "محصول، product_id، metric و بازه‌ی زمانی را دوباره تفسیر نکن.",
            "اگر سؤال «چرا؟» است، آن را ادامه‌ی سؤال قبلی بدان.",
            "برای «چرا؟» ranking جدید یا بازه‌ی زمانی جدید نساز.",
        ]
    )

    return "\n".join(lines)


def run(
    question: str,
    chat_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    اجرای یک درخواست.

    Priority 2:
    برای Follow-upهای کوتاه، active conversation context
    به‌صورت deterministic از history استخراج می‌شود.
    """

    # =========================================================
    # 1. Load persistent conversation memory
    # =========================================================

    if chat_id and history is None:
        history = memory_store.load_messages(chat_id)

    messages: list[dict[str, Any]] = list(history) if history else []

    # =========================================================
    # 2. Add system prompt only once
    # =========================================================

    if not messages or messages[0].get("role") != "system":

        reference_date = get_reference_date()

        print(
            f"\nDATASET REFERENCE DATE = {reference_date}\n"
        )

        system_prompt = (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            f"DATASET REFERENCE DATE = "
            f"{reference_date.isoformat()}\n\n"
            f"این تاریخ، تاریخ مرجع ثابت تمام محاسبات زمانی "
            f"این مکالمه است.\n"
            f"تاریخ واقعی سیستم یا تاریخ واقعی امروز نباید "
            f"در تحلیل استفاده شود.\n"
            f"برای بازه‌های نسبی، تاریخ مرجع نقطه‌ی پایان "
            f"بازه است.\n"
            f"تمام بازه‌های SQL باید با قرارداد [start, end) "
            f"ساخته شوند.\n"
            f"هرگز از NOW() یا CURRENT_DATE واقعی "
            f"PostgreSQL استفاده نکن."
        )

        messages.insert(
            0,
            {
                "role": "system",
                "content": system_prompt,
            },
        )

    # =========================================================
    # 3. Resolve active conversation context BEFORE adding
    #    current question
    # =========================================================

    is_follow_up = _is_follow_up_question(question)

    previous_context = _extract_active_context(messages)

    conversation_context = {
        **previous_context,
        "is_follow_up": is_follow_up,
    }

    # =========================================================
    # 4. Add explicit active context ONLY for follow-up
    # =========================================================

    if is_follow_up:

        active_context_text = _build_follow_up_system_context(
            conversation_context
        )

        messages.append(
            {
                "role": "system",
                "content": active_context_text,
            }
        )

    # =========================================================
    # 5. Add current user question
    # =========================================================

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # =========================================================
    # 6. Compact only when necessary
    # =========================================================

    messages = memory_store.maybe_compact(messages)

    # =========================================================
    # 7. Run graph
    # =========================================================

    graph = get_graph()

    result = graph.invoke(
        {
            "messages": messages,
            "conversation_context": conversation_context,
            "iterations": 0,
            "consecutive_tool_errors": 0,
            "tool_trace": [],
            "errors": [],
        }
    )

    # =========================================================
    # 8. Persist conversation
    # =========================================================

    if chat_id:
        memory_store.save_messages(
            chat_id,
            result["messages"],
        )

    return result


def main() -> None:
    # مدل embedding رو همین اول، قبل از سوال کاربر، لود می‌کنیم -- نه
    # وسط اولین صدا زدن tool_rag. یعنی این تاخیر (چند ثانیه) این‌جا اتفاق
    # می‌افته، نه وسط جواب دادن به کاربر. اگه بعداً سرور FastAPI ساختید،
    # همین تابع رو در startup سرویس صدا بزنید (نه اینجا).
    preload_embedding_model()

    # ساخت جدول chat_memory (اگه از قبل نباشه) -- فقط یک‌بار در
    # استارتاپ، نه در مسیر داغ هر درخواست. نیاز به CHAT_DB_* در .env
    # داره (نگاه کن به .env.example)؛ اگه هنوز تنظیم نشده، فقط لاگ
    # می‌شه و برنامه با خطا متوقف نمی‌شه (memory_store هم خودش هر خطای
    # اتصال رو silent می‌کنه).
    try:
        memory_store.ensure_schema()
    except Exception as exc:  # noqa: BLE001
        print(f"[هشدار] ensure_schema شکست خورد -- حافظه‌ی چت کار نخواهد کرد تا رفعش کنی: {exc}")

    result = run(
        "کدام ۱۰ محصول بیشترین تعداد کامنت مثبت را دریافت کرده‌اند؟ کدام ۱۰ برند بیشترین تعداد کامنت را دریافت کرده‌اند و میانگین امتیاز مشتریان آن‌ها چقدر است؟",
        # "نظر کاربران درمورد کالاهای مربوط به مدسه چطوره؟",
    #    "اکثرن از چه برند ها و کتگوری هایی هستن؟",
        chat_id="test-top-selling-product_4"
    )

    print("\nFINAL ANSWER:")
    print(result.get("final_answer"))

    print("\nVALIDATION:")
    print(result.get("validation"))

    errors = result.get("errors") or []
    if errors:
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")


    # followup = run(
    #     "درآمد ما در 1 سال اخیر چقدر بوده و چند درصد این درآمد به کدام کتگوری ها مربوطه؟",
    #     chat_id="test-top-selling-product_3"
    # )

    # print("\n\n--- سوال ادامه‌دار با حافظه‌ی Postgres ---")
    # print(followup.get("final_answer"))

    # followup = run(
    #     "چه شهرهایی در چه بازه های زمانیی چه محصولاتی رو بیشتر خریدند؟",
    #     chat_id="test-top-selling-product_3"
    # )

    # print("\n\n--- سوال ادامه‌دار با حافظه‌ی Postgres ---")
    # print(followup.get("final_answer"))


    # followup = run(
    #     "با توجه به وضعیت کلی خرید و داده های رفتاری کاربران در سال اخیر به نظرت باید چیکار کنیم برای بهبود وضعیت؟",
    #     chat_id="test-top-selling-product_3"
    # )

    # print("\n\n--- سوال ادامه‌دار با حافظه‌ی Postgres ---")
    # print(followup.get("final_answer"))

    # chart_result = run(
    #     "نمودار فروش محصولات آرایشی بهداشتی رو در 1 سال اخیر نشون بده و بگو وضعیتشون در چه حالیه؟",
    #     chat_id="test-top-selling-product_2"
    # )

    # print("\n\n--- نمونه‌ی نمودار ---")
    # print(chart_result.get("final_answer"))


if __name__ == "__main__":
    main()