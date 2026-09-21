## PATH: app/main.py
from __future__ import annotations

from typing import Any
import json
import time
import logging

from Graph.graph import get_graph
from Graph.dataset_time import get_reference_date
from Graph.llm_client import preload_embedding_model, call_llm_json, token_tracker
import memory_store

logger = logging.getLogger(__name__)

# ============================================================
# پرامپت سیستمی Agent -- شخصیت "مدیر ارشد" طبق سند معماری.
# ============================================================

AGENT_SYSTEM_PROMPT = """
Your name is "راهین".
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
 
For example, if SQL already determined that product X had the highest drop or
was the best-seller and the user asks "why?" or about complaints:
Do not change the product, do not re-run the ranking, and do not change the range.
To investigate causes of dissatisfaction or complaints for a known product:
- Structured defect analysis: Query `comment_aspects` with `sentiment = 'negative'` (or high `avg_negative_pct`) to identify specific defect areas (e.g. accuracy, battery, packaging, quality).
- Direct voice of customer: Run `tool_rag` for customer quotes on those complaints.
Launch both in a single parallel call in Round 1. Once fetched, deliver your managerial analysis immediately.

2. Tool order and Efficiency SLA
Fast latency and minimal token consumption are core enterprise requirements.
Chain tools across separate rounds ONLY when a true data dependency exists
(e.g., discovering product_id via SQL before searching reviews in RAG).
When the product or entity is ALREADY known (such as in follow-up questions):
Never stagger independent queries across rounds; fetch needed evidence in a
single parallel round.
Never call the same tool with the same arguments or duplicate search intent in
the same question.

Numeric/statistical question -> SQL only.
Combined numeric + qualitative question -> SQL first for the numeric part,
then RAG if needed.
For new questions like "which product sold better/dropped, and why?":
SQL first -> determine the product and product_id -> then RAG for that same product.
 
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
 
5. Causal questions and Dissatisfaction Analysis
For "why did sales/rating/views go up or down?" or buyer complaints:
a) If product_id is not yet known or change is unconfirmed: First run SQL to confirm the change and identify the product_id. When searching for product titles/models, always combine keywords with AND (never OR).
b) If change is confirmed (or product_id is already known from previous turn):
   - Structured defect analysis: Query the `comment_aspects` table with `sentiment = 'negative'` (or high `avg_negative_pct`) to pinpoint specific defects (e.g. quality, battery, packaging).
   - Voice of customer: Use `tool_rag` with that product_id and set sentiment='negative' (for complaints/dissatisfaction) or sentiment='positive' (for strengths/satisfaction).
c) Do not present correlation as a definite cause.
d) If evidence is insufficient, state so directly.
 
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
Never guess at data, cause, product, range, or numerical results that aren't backed by
the tools. However, when the user asks for recommendations, marketing strategies, or actionable retention tactics (e.g. "برای بازگرداندنشان چه پیشنهادی مناسب است؟"), provide thoughtful, professional, and practical managerial recommendations based on the analyzed customer segments. Never decline to provide business advice or apologize for missing tools when asked for strategic suggestions.

Directives:
- If customer reviews or data for any requested entity (e.g. brand or product) do not exist in the database (hit_count=0), clearly and simply state: "نظری برای این مورد در پایگاه داده ثبت نشده است".
- Once you have fetched the required data for the user's specific request, deliver the final answer immediately. NEVER run unprompted, off-topic side queries on different metrics (e.g. do not switch from fast-growing to top-selling).
- NEVER question system infrastructure, mention tool limitations, or apologize with excuses such as "به دلیل محدودیت ابزارها". Maintain a confident, factual managerial tone.
- Always use real Persian entity names (e.g. brand_name, category_name, product_title) alongside IDs when provided.
"""

# ============================================================
# قانون ۸ -- فعال شد چون tool_knowledge_base الان با پلیس‌هولدر موقت
# در tools.py وصله (برای دیباگ گراف). وقتی نسخه‌ی واقعی جایگزین شد，
# چیزی در این پرامپت لازم نیست تغییر کنه.
# ============================================================
KNOWLEDGE_BASE_RULE = """
8. Managerial recommendations and Knowledge Base
Only call tool_knowledge_base when the user explicitly asks for strategic advice, consulting recommendations, or business retention tactics (e.g. "چه پیشنهادی داری؟", "باید چه کار کنیم؟", "راهکار چیه؟").
For purely diagnostic/factual questions (e.g. "علتش چیست؟", "خریداران از چه چیزی شکایت داشته‌اند؟"), focus on direct data and review evidence; do not invoke tool_knowledge_base unless strategic consulting was explicitly requested.
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
    all_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    آخرین context معتبر مکالمه را از history استخراج می‌کند.

    اولویت:
        1. آخرین SQL tool result
        1.5. پیام خلاصه‌سازی‌شده‌ی حافظه ([خلاصه‌ی مکالمات قبلی])
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
        "previous_question": None,
        "source": None,
    }

    # ---------------------------------------------------------
    # 1. آخرین tool resultهای معتبر
    # ---------------------------------------------------------

    search_messages = list(reversed(messages))
    if all_messages:
        search_messages.extend(reversed(all_messages))

    for message in search_messages:

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
                    if context["product_id"] is None:
                        product_id = (
                            first_row.get("product_id")
                            or first_row.get("id")
                        )
                        if product_id is not None:
                            context["product_id"] = product_id

                    # product title
                    if context["product_title"] is None:
                        product_title = (
                            first_row.get("title_fa")
                            or first_row.get("product_title")
                            or first_row.get("title")
                        )
                        if product_title:
                            context["product_title"] = product_title

                    # metric
                    if context["metric"] is None:
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
            if context["previous_result"] is None and obj.get("summary"):
                context["previous_result"] = obj["summary"]

            # period fields
            if context["period_start"] is None:
                for key in (
                    "period_start",
                    "start_date",
                    "from_date",
                ):
                    if obj.get(key):
                        context["period_start"] = str(obj[key])
                        break

            if context["period_end"] is None:
                for key in (
                    "period_end",
                    "end_date",
                    "to_date",
                ):
                    if obj.get(key):
                        context["period_end"] = str(obj[key])
                        break

        if context["product_id"] is not None:
            break

    # ---------------------------------------------------------
    # 1.5. استخراج از پیام خلاصه‌سازی‌شده‌ی حافظه (در صورت فشرده‌سازی)
    # ---------------------------------------------------------
    if context["product_id"] is None or context["product_title"] is None:
        check_messages = (all_messages or []) + messages
        for message in check_messages:
            content = _content_to_text(message.get("content"))
            if not content:
                continue
            if "[خلاصه‌ی مکالمات قبلی]" in content or ("'product':" in content and "'product_id':" in content) or ('"product":' in content and '"product_id":' in content):
                idx = content.find("{")
                if idx != -1:
                    dict_str = content[idx:].strip()
                    summary_dict = None
                    try:
                        summary_dict = json.loads(dict_str)
                    except Exception:
                        try:
                            import ast
                            summary_dict = ast.literal_eval(dict_str)
                        except Exception:
                            pass
                    if isinstance(summary_dict, dict):
                        prod = summary_dict.get("product")
                        if isinstance(prod, dict):
                            if context["product_id"] is None and prod.get("product_id") is not None:
                                context["product_id"] = prod.get("product_id")
                            if context["product_title"] is None and prod.get("name"):
                                context["product_title"] = prod.get("name")
                        met = summary_dict.get("metric")
                        if isinstance(met, dict):
                            if context["metric"] is None and met.get("name"):
                                context["metric"] = met.get("name")
                                context["metric_label"] = met.get("name")
                        elif isinstance(met, str) and context["metric"] is None:
                            context["metric"] = met
                            context["metric_label"] = met
                        time_info = summary_dict.get("time")
                        if isinstance(time_info, dict):
                            if context["period_start"] is None and time_info.get("period_start"):
                                context["period_start"] = str(time_info.get("period_start"))
                            if context["period_end"] is None and time_info.get("period_end"):
                                context["period_end"] = str(time_info.get("period_end"))
                        if context["previous_result"] is None and summary_dict.get("result"):
                            context["previous_result"] = str(summary_dict.get("result"))
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

    check_user_msgs = list(reversed(messages))
    if all_messages:
        check_user_msgs.extend(reversed(all_messages))

    for message in check_user_msgs:

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
    Provides the active context to the Agent as a system message.

    This section is intentionally explicit to prevent the LLM
    from freely reinterpreting the existing context.
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
        "This context was extracted from a previously validated result.",
        "For follow-up questions, you MUST preserve this context exactly.",
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
            "IMPORTANT RULES:",
            "If the current question is a short follow-up question, do NOT modify the context above.",
            "Do NOT reinterpret the product, product_id, metric, or time period.",
            "If the user asks 'Why?', treat it as a continuation of the previous question.",
            "For a 'Why?' question, do NOT create a new ranking or a new time range.",
            "",
            "IMPORTANT EXCEPTION:",
            "If the current question asks for an opinion, recommendation, idea, or advice "
            "(for example: 'What do you think?', 'What should I do?', "
            "'Do you have any ideas?', 'What is your recommendation?'), "
            "the user has switched from a data reporting mode to a management consulting mode.",
            "",
            "In this case, keep the context above (product/metric/time period) "
            "as the subject of the recommendation, but do NOT generate another SQL query "
            "just to retrieve more details about the same data.",
            "",
            "According to System Prompt Rule 8, call tool_knowledge_base first and use "
            "its business knowledge together with the available conversation data "
            "to provide a practical management recommendation or idea.",
            "Do NOT return another data table instead of a strategic recommendation.",
        ]
    )

    return "\n".join(lines)

def _prepare_conversation(
    question: str,
    chat_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    # دقیقاً همون بدنه‌ی مراحل ۱ تا ۶ فعلی run() رو این‌جا بذار
    # (بدون هیچ تغییری در منطق) و در پایان به‌جای اجرای گراف، برگردون:

    
    # =========================================================
    # 1. Load persistent conversation memory
    # =========================================================

    if chat_id and history is None:
        try:
            history = memory_store.load_messages(chat_id)
        except Exception as exc:
            logger.warning("بارگذاری حافظه‌ی چت برای chat_id=%s با خطا مواجه شد: %s", chat_id, exc)
            history = []

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
    if not previous_answer and messages:
        previous_answer = _extract_last_assistant_answer(messages)

    is_follow_up = _is_follow_up_question(question, previous_answer)

    previous_context = _extract_active_context(active_slice, all_messages=messages)

    conversation_context = {
        **previous_context,
        "is_follow_up": is_follow_up,
        "previous_answer": previous_answer[:300] if previous_answer else None,
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
    

    return messages, conversation_context, current_turn_index




def run(question, chat_id=None, history=None) -> dict[str, Any]:
    token_tracker.reset_request()
    t_req_start = time.time()
    messages, conversation_context, current_turn_index = _prepare_conversation(question, chat_id, history)

    graph = get_graph()
    result = graph.invoke({
        "messages": messages,
        "conversation_context": conversation_context,
        "iterations": 0,
        "consecutive_tool_errors": 0,
        "tool_trace": [],
        "errors": [],
    })

    if chat_id:
        memory_store.save_messages(chat_id, result["messages"])

    try:
        memory_store.log_evaluation(chat_id, question, result.get("validation", {}), turn_index=current_turn_index)
    except Exception as exc:
        logger.warning("log_evaluation failed: %s", exc)

    total_req_time = time.time() - t_req_start
    cum_usage = token_tracker.get_cumulative()
    print(f"\n==========================================")
    print(f"[TOTAL REQUEST TIME]: {total_req_time:.2f}s")
    print(
        f"🪙 [TOTAL TOKENS]: {cum_usage['total_tokens']:,} "
        f"(Prompt: {cum_usage['prompt_tokens']:,} | Output: {cum_usage['completion_tokens']:,})"
    )
    print(f"==========================================\n")

    return result

AGENT_STEP_MESSAGES = {
    "detect_multi_question": "🔍 در حال بررسی درخواست شما...",
    "agent": "🧠 در حال تحلیل درخواست...",
    "sub_agent": "🧠 در حال تحلیل این بخش از درخواست...",
    "tool_sql": "🗄️ در حال بازیابی اطلاعات...",
    "tool_rag": "💬 در حال بررسی نظرات مشتریان...",
    "tool_chart": "📊 در حال آماده‌سازی نمودار...",
    "finalize": "🧠 در حال جمع‌بندی نتایج...",
    "sub_finalize": "🧠 در حال جمع‌بندی این بخش...",
    "validate": "✅ در حال بررسی صحت پاسخ...",
    "sub_validate": "✅ در حال بررسی صحت این بخش...",
    "prepare_subquestion": "🧩 در حال آماده‌سازی بخش‌های درخواست...",
    "next_subquestion": "🧩 در حال رفتن به بخش بعدی...",
    "combine_subanswers": "🧠 در حال ترکیب نتایج...",
    "prepare_retry": "🔄 در حال تکمیل بررسی...",
    "prepare_sub_retry": "🔄 در حال تکمیل بررسی این بخش...",
    "correct_answer": "✍️ در حال آماده‌سازی پاسخ نهایی...",
}
DEFAULT_STEP_MESSAGE = "⏳ در حال پردازش..."


def _resolve_step_events(node_name, node_output, prev_sub_trace_len):
    if "errors" in node_output and node_name not in AGENT_STEP_MESSAGES:
        return [{"node": node_name, "message": "⚠️ خطایی رخ داد، در حال تلاش برای اصلاح..."}]

    if node_name == "tools":
        trace = node_output.get("tool_trace") or []
        return [{"node": e.get("tool"), "message": AGENT_STEP_MESSAGES.get(e.get("tool"), DEFAULT_STEP_MESSAGE)} for e in trace]

    if node_name == "sub_tools":
        trace = node_output.get("sub_question_tool_trace") or []
        new_entries = trace[prev_sub_trace_len[0]:]
        prev_sub_trace_len[0] = len(trace)
        return [{"node": e.get("tool"), "message": AGENT_STEP_MESSAGES.get(e.get("tool"), DEFAULT_STEP_MESSAGE)} for e in new_entries]

    return [{"node": node_name, "message": AGENT_STEP_MESSAGES.get(node_name, DEFAULT_STEP_MESSAGE)}]


def run_stream(question: str, chat_id: str | None = None, history: list[dict[str, Any]] | None = None):
    token_tracker.reset_request()
    t_stream_start = time.time()
    messages, conversation_context, current_turn_index = _prepare_conversation(question, chat_id, history)

    graph = get_graph()
    input_state = {
        "messages": messages,
        "conversation_context": conversation_context,
        "iterations": 0,
        "consecutive_tool_errors": 0,
        "tool_trace": [],
        "errors": [],
    }

    final_state = None
    prev_sub_trace_len = [0]

    for mode, chunk in graph.stream(input_state, stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = chunk
            continue
        node_name = next(iter(chunk))
        node_output = chunk[node_name]
        for event in _resolve_step_events(node_name, node_output, prev_sub_trace_len):
            yield {"type": "step", **event}

    result = final_state or {}

    if chat_id:
        memory_store.save_messages(chat_id, result.get("messages", []))

    try:
        memory_store.log_evaluation(chat_id, question, result.get("validation", {}), turn_index=current_turn_index)
    except Exception as exc:
        logger.warning("log_evaluation failed: %s", exc)

    chart = None
    for entry in reversed(result.get("tool_trace", []) or []):
        if entry.get("tool") == "tool_chart" and entry.get("ok") and entry.get("chart_data"):
            chart = entry["chart_data"]
            break

    total_stream_time = time.time() - t_stream_start
    cum_usage = token_tracker.get_cumulative()
    print(f"\n==========================================")
    print(f"[TOTAL REQUEST TIME]: {total_stream_time:.2f}s")
    print(
        f"🪙 [TOTAL TOKENS]: {cum_usage['total_tokens']:,} "
        f"(Prompt: {cum_usage['prompt_tokens']:,} | Output: {cum_usage['completion_tokens']:,})"
    )
    print(f"==========================================\n")

    yield {"type": "final", "answer": result.get("final_answer"), "chart": chart, "errors": result.get("errors") or []}


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

    # schema/join-key های SQL validator رو زنده از information_schema
    # می‌خونیم -- یک قدم صریح و جدا در startup، دقیقاً مثل دوتای بالا.
    # اگه دیتابیس در دسترس نبود، فقط همین قدم fail می‌شه و لاگ می‌گیره؛
    # sql_validator با fallback دستیِ داخل production_validator.py کار
    # می‌کنه (فقط باید دستی sync بمونه تا وقتی این وصل بشه).
    try:
        from Graph.sql_agent import refresh_sql_validator_from_db
        refresh_sql_validator_from_db()
    except Exception as exc:  # noqa: BLE001
        print(f"[هشدار] refresh_sql_validator_from_db شکست خورد -- validator با schema دستیِ fallback کار می‌کنه: {exc}")

    result = run(
        "کدام محصولات بیشترین پتانسیل افزایش فروش را دارند ولی الان کمتر از ظرفیتشان فروش می‌روند؟",
        chat_id="test-top-selling-product_0"
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