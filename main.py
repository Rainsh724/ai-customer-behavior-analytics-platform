## PATH: app/main.py
from __future__ import annotations

from typing import Any
import json
import logging

from Graph.graph import get_graph
from Graph.dataset_time import get_reference_date
from Graph.llm_client import preload_embedding_model, call_llm_json
import memory_store

logger = logging.getLogger(__name__)

# ============================================================
# پرامپت سیستمی Agent -- شخصیت "مدیر ارشد" طبق سند معماری.
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are the senior Agent of an intelligent business-analytics system. Your job
is to analyze the online store's data and produce accurate, Persian-language,
managerial answers.
 
Tools:
- tool_sql: run valid PostgreSQL for numeric/structural analysis.
- tool_rag: semantic search over customer reviews using search_topic and,
  if needed, product_id; no metadata filtering.
- tool_chart: build a chart from SQL, only when the user explicitly asks
  for a chart.
 
Rules:
 
1. Memory and follow-up
First check the history of this same conversation. If the needed answer or
data already exists in earlier messages or tool results, do not recompute it.
 
For short questions like "why?", "what's the reason?", "how?", "the same
product?" and similar, infer the intent from the last valid result in the
conversation.
 
In a follow-up you must preserve:
- the product and product_id
- the metric
- the time range
- the previous result
 
For example, if SQL already determined that product X was the best-seller in
a given range and the user asks "why?", do not change the product, do not
re-run the ranking, and do not change the range. To find the reason, go
straight to RAG for that same product.
 
2. Tool order
Numeric/statistical question -> SQL only.
 
Combined numeric + qualitative question -> SQL first for the numeric part,
then RAG if needed.
 
Causal question about an increase/decrease/drop/growth -> SQL only first.
Only run RAG if SQL actually confirms the change in question.
 
For questions like "which product sold better, and why?":
SQL -> determine the product and product_id -> RAG for that same product.
 
Never run SQL and RAG at the same time for one causal question.
 
3. Definition of "sales"
"Best-selling", "top sales" and "best-selling products" default to meaning
the highest number of purchases.
 
In user_behavior_logs, each purchase is one purchase event, so the default
metric is:
 
COUNT(*) AS units_sold
 
and the ranking:
 
ORDER BY units_sold DESC
 
If the user explicitly asks for "sales amount", "revenue" or "sales value",
calculate the monetary amount instead.
 
products.price is the product's *current* price, so price * units_sold is
only an estimated_sales figure and must not be presented as actual
historical revenue without explanation.
 
To avoid fan-out, first aggregate purchases by product_id and only then JOIN
to products. Never run SUM(products.price) directly on a JOIN with purchase
events.
 
Whenever ORDER BY + LIMIT is used for ranking / selecting a top-N (e.g. "top
10 best-selling products"), always add a deterministic tie-breaker (such as
product_id ASC) as the second ORDER BY key, even if the primary metric is
units_sold/COUNT. Without this, when several products are tied, each new
execution -- including when tool_chart rebuilds the same ranking to draw the
chart -- can return a different set of tied products, causing the SQL table
and the chart to show different products for the same question. If the user
asks why SQL and the chart disagree, treat this (missing tie-breaker) as the
likely cause, not an actual data discrepancy.
 
Whenever ranking is based on a ratio / average / percentage (e.g.
conversion_rate, avg_negative_pct, avg_rating) rather than a raw count,
always apply a minimum sample-size filter (e.g. view_cnt >= 30 or
comment_cnt >= 5, depending on the question) in the WHERE clause. Without
this filter, products with very few views/comments (e.g. 1 view or 1
comment) easily hit extreme values of 0% or 100% and fill the ranking with
statistical noise rather than a real business signal. If you apply such a
filter, state in the final answer that results are limited to products with
at least that minimum number of views/comments.
 
4. Time
The reference date for all relative calculations is the DATASET REFERENCE
DATE given in the system prompt; never use today's real date, NOW(), or
CURRENT_DATE.
 
"Last 7 days", "last 30 days", "last 3 months", "last 6 months" and similar
are rolling windows relative to the reference date, and the length of the
window must be taken exactly from the user's wording.
 
"Last month" = a rolling one-month window, not the previous calendar month.
"Previous / last month" when referring to the calendar = the previous
calendar month.
 
If the user gives an exact date, use exactly that range.
 
Never compute the exact start/end date yourself (mentally or in text).
Always write the expression inside the SQL itself, relative to the literal
reference date, and let PostgreSQL compute it -- e.g. instead of writing
'2022-09-01' directly, write '<reference date>'::date - INTERVAL '6 months'.
Only PostgreSQL's own computation is valid, never your own manual one.
 
All ranges must be built with the [start, end) convention:
>= start AND < end. When end = the reference date (i.e. the range must cover
through the reference date itself, inclusive), the real end in SQL must be
'<reference date>'::date + INTERVAL '1 day', not the reference date itself;
otherwise events on the reference day are wrongly excluded. Apply this rule
consistently across every query tied to one question (e.g. in both tool_sql
and tool_chart for the same range).
 
In the final answer, extract the range from the SQL that actually ran -- not
from memory or by recomputing it. Note that because the range is
[start, end), if the SQL is e.g.:
timestamp >= '2022-12-01'
AND timestamp < '2023-03-02'
then the last reported day is 2023-03-01, not 2023-03-02 (end is always
exclusive).
 
5. Causal questions
For "why did sales/rating/views go up or down?":
 
a) First, SQL with an explicit computation of the current period and the
   comparison period.
b) If the SQL doesn't sufficiently prove the change, run a corrected SQL.
c) If the change is not confirmed, stop and say the data doesn't support the
   claim; do not run RAG.
d) If the change is confirmed, run RAG for qualitative evidence.
   Decrease -> search_topic toward dissatisfaction/complaints.
   Increase -> search_topic toward satisfaction/positive reception.
e) If you have a valid product_id, always pass that same product_id.
f) Keep the value/percentage of the change (from SQL) separate from the
   qualitative themes (from RAG). Do not present correlation as a definite
   cause.
 
If RAG doesn't have enough evidence for a cause, say explicitly that the
evidence is not sufficient to determine a definite cause.
 
6. RAG limitation
When product_id is specified, that product is the primary reference.
 
If RAG returns hit_count=0 for that same product_id:
- Do not drop the product_id.
- Do not run a general search to find "alternative evidence."
- Do not attribute other products' or category-level reviews to the main
  product.
- Report the result as "no sufficient direct evidence found."
 
Only if you explicitly decide to use category-level context, label it as
category-level context, never as direct product evidence.
 
7. Tool errors
If a tool returns an error, retry once with a corrected tool_call.
SQL succeeding alone isn't enough -- the result must directly answer the
question asked.
 
If several attempts fail and no valid data is obtained, clearly state the
limitation and do not guess.
 
8. Charts
Only run tool_chart when the user explicitly asks for a chart, graph,
dashboard, or visualization. The frontend renders the chart automatically and graphically.
Never output any JSON, chart configuration, code blocks, or technical chart markup in your text response.
Your response must only contain title, table (if helpful), trend analysis, and managerial suggestions.

9. Final answer
The answer must always be in Persian, fluent, concise, and managerial.
Never show raw JSON, SQL, or tool traces.
 
Never guess at data, cause, product, range, or a result that isn't backed by
the tools.
"""

# ============================================================
# قانون ۸ -- فعال شد چون tool_knowledge_base الان با پلیس‌هولدر موقت
# در tools.py وصله (برای دیباگ گراف). وقتی نسخه‌ی واقعی جایگزین شد，
# چیزی در این پرامپت لازم نیست تغییر کنه.
# ============================================================
KNOWLEDGE_BASE_RULE = """
8. Before giving any suggestion or managerial recommendation (not just
   reporting numbers/reviews, but whenever the user wants to know "what
   should I do?"), you must first call tool_knowledge_base with the relevant
   topic and build your suggestion by combining that trained knowledge with
   your own general knowledge -- not from your own memory alone. If the
   knowledge base has nothing relevant, say so explicitly and proceed based
   on your own general knowledge.
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


# ============================================================
# تشخیص follow-up با LLM (fallback روی whitelist ثابت بالا)
# ============================================================
# مشکل نسخه‌ی قبلی: _is_follow_up_question فقط یک whitelist ثابت از
# عبارت‌ها («چرا؟»، «دلیلش؟»، ...) و چند پیشوند («چرا »، «همون »، ...)
# رو follow-up می‌شناخت. یک عبارت کاملاً معمولی مثل «همین جوابتو
# خلاصه‌تر بهم بده» هیچ‌کدوم از این‌ها رو نداره -- پس is_follow_up=False
# می‌شد، active context هیچ‌وقت به‌عنوان یک پیام system صریح تزریق
# نمی‌شد، و مدل فقط با تاریخچه‌ی خام (و محدود) تنها می‌موند.
#
# راه‌حل: whitelist سریع/رایگان بالا رو به‌عنوان fast-path نگه می‌داریم
# (اکثر follow-upهای رایج رو بدون هیچ تماس اضافه‌ی LLM تشخیص می‌ده)، ولی
# وقتی هیچ‌کدوم مچ نشد و یک پاسخ قبلی معتبر در تاریخچه هست، یک تماس سبک
# LLM (دقیقاً همون الگوی call_llm_json که در Graph/nodes.py برای تشخیص
# سوال چندبخشی استفاده می‌شه) می‌پرسه که آیا این سوال واقعاً ادامه‌ی
# همون پاسخ قبلیه یا نه. این باعث می‌شه پارافریزهای غیرمنتظره هم درست
# تشخیص داده بشن، نه فقط عبارت‌های از پیش پیش‌بینی‌شده.
# ============================================================

FOLLOW_UP_CLASSIFIER_PROMPT = """
You must determine whether the user's "new question" is a direct
continuation/follow-up of the assistant's "last answer" in this same
conversation, or a completely new, independent question.
 
Return only a JSON object in this format -- write no extra text:
 
{"is_follow_up": true|false}
 
Rules -- decide conservatively:
- If the new question implicitly or explicitly refers to that same previous
  answer (e.g. "summarize it more", "show the same thing as a chart",
  "why?", "say the same for another brand", "change its unit") -> true.
- If the new question is fully independent and understandable without
  knowing the previous answer (even if the topic is similar) -> false.
- If the previous answer was empty/irrelevant, or the new question is a
  completely new topic -> false.
"""


def _classify_follow_up_llm(question: str, previous_answer: str) -> bool:
    if not previous_answer or not previous_answer.strip():
        return False

    try:
        result = call_llm_json(
            FOLLOW_UP_CLASSIFIER_PROMPT,
            (
                f"آخرین پاسخ دستیار:\n{previous_answer}\n\n"
                f"سوال جدید کاربر:\n{question}"
            ),
        )
        return bool(isinstance(result, dict) and result.get("is_follow_up"))
    except Exception as exc:  # noqa: BLE001 - fail-open: نمونه‌ی معمولی/تک‌سوالی
        logger.warning(
            "follow-up classification failed, defaulting to False: %s",
            exc,
        )
        return False


def _is_follow_up_question(question: str, previous_answer: str = "") -> bool:
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

    if len(normalized.split()) <= 5 and normalized.startswith(short_followups):
        return True

    # fast-path هیچی رو مچ نکرد -- اگه پاسخ قبلی معتبری داریم، یک تماس
    # سبک LLM بپرس (به‌جای اینکه بی‌قید‌وشرط False برگردونیم).
    return _classify_follow_up_llm(question, previous_answer)


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


def _extract_last_assistant_answer(
    messages: list[dict[str, Any]],
) -> str:
    """
    آخرین پاسخ متنیِ واقعیِ دستیار (نه پیامی که فقط tool_call بوده) رو
    از تاریخچه پیدا می‌کنه -- برای دادن context به
    _classify_follow_up_llm استفاده می‌شه.
    """
    for message in reversed(messages):

        if message.get("role") != "assistant":
            continue

        if message.get("tool_calls"):
            continue

        content = _content_to_text(message.get("content"))

        if content:
            return content

    return ""


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

    # ---------------------------------------------------------
    # ریشه‌ای‌سازی: به‌جای اینکه previous_answer/previous_context از
    # کل تاریخچه‌ی چت (از اول مکالمه، حتی turnهای قدیمیِ کاملاً بی‌ربط
    # یا خطادار) استخراج بشه، فقط از "ترید فعال" استفاده می‌کنیم --
    # یعنی از اولین turn موفق (grounded + relevance بالا در eval_log)
    # که با جست‌وجوی معکوس پیدا می‌شه، تا انتها.
    #
    # اگه هیچ turn موفقی توی این chat_id نبود، active_slice خالی
    # می‌مونه -> previous_answer="" -> _is_follow_up_question هم خودبه‌خود
    # False برمی‌گردونه (حتی بدون زدن تماس LLM classify، چون
    # _classify_follow_up_llm از قبل چک می‌کنه previous_answer خالی
    # نباشه) -> هیچ context بی‌ربطی هم تزریق نمی‌شه.
    # ---------------------------------------------------------

    turn_ok_map = (
        memory_store.get_turn_ok_map(chat_id) if chat_id else {}
    )
    _, existing_turns = memory_store.split_into_turns(messages)
    current_turn_index = len(existing_turns)

    active_slice: list[dict[str, Any]] = []
    for idx in range(len(existing_turns) - 1, -1, -1):
        if turn_ok_map.get(idx):
            active_slice = existing_turns[idx]
            break

    previous_answer = _extract_last_assistant_answer(active_slice)
    is_follow_up = _is_follow_up_question(question, previous_answer)

    previous_context = _extract_active_context(active_slice)

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

    # =========================================================
    # 9. Log evaluation metrics (faithfulness / relevance / confidence)
    #    برای محاسبه‌ی تجمعیِ calibration در آینده (نگاه کن به
    #    memory_store.py::log_evaluation/compute_calibration). این کار
    #    برای مسیر تک‌سوالی و چندبخشی یکسانه، چون هر دو مسیر نهایتاً از
    #    همون validate_node مشترک state["validation"] رو پر می‌کنن.
    #    Best-effort -- شکستش هیچ‌وقت نباید جواب کاربر رو خراب کنه.
    # =========================================================

    try:
        memory_store.log_evaluation(
            chat_id,
            question,
            result.get("validation", {}),
            turn_index=current_turn_index,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_evaluation failed: %s", exc)

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

    try:
        memory_store.ensure_eval_schema()
    except Exception as exc:  # noqa: BLE001
        print(f"[هشدار] ensure_eval_schema شکست خورد -- لاگ ارزیابی/calibration کار نخواهد کرد تا رفعش کنی: {exc}")

    result = run(
        " در شهرهایی که بیشترین خرید رو داشتند در چه بازه های زمانیی چه محصولاتی رو بیشتر خریدند؟",
        chat_id="test-top-selling-product_9"
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
    #     "",
    #     chat_id=""
    # )

    # print("\n\n--- سوال ادامه‌دار با حافظه‌ی Postgres ---")
    # print(followup.get("final_answer"))

    # followup = run(
    #     "",
    #     chat_id=""
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