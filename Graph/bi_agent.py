## PATH: app/graph/bi_agent.py
"""
پیاده‌سازی واقعی مسیر BI (نمودار / داشبورد) -- جایگزین استاب‌های
bi_planner و bi_builder در nodes.py.

چرا BI یک "route" چهارم موازی sql/rag/hybrid نیست؟
----------------------------------------------------
BI هیچ‌وقت به‌تنهایی داده تولید نمی‌کنه؛ همیشه روی داده‌ای سوار می‌شه که
یا مسیر SQL آورده (state["sql_result"]) یا مسیر RAG/ABSA آورده
(state["aspect_statistics"] / state["aspect_trends"] / state["ranked_aspects"])
یا هر دو (hybrid). اگر BI رو یک route چهارم مدل می‌کردیم، مجبور بودیم
ترکیب‌هاش با sql/rag/hybrid رو به‌صورت route های جدا (sql_bi, rag_bi,
hybrid_bi, ...) تعریف کنیم -- که هم duplicate-logic ایجاد می‌کنه (یک
sql_agent دومِ مخصوصِ حالت "sql_bi") هم نگه‌داری‌ش سخت می‌شه.

در عوض:
    - state["wants_chart"] در همون لحظه‌ای که query_router مسیر داده
      (sql/rag/hybrid) رو تعیین می‌کنه، ست می‌شه (نگاه کن به nodes.py).
    - evidence_normalizer از قبل نقطه‌ی تلاقی هر ترکیبی از sql/rag/hybrid
      هست (sql-only، rag-only، hybrid-sequential، hybrid-parallel -- همه
      قبل از رسیدن به evidence_fusion از evidence_normalizer عبور
      می‌کنن). پس دقیقاً همون‌جا بهترین نقطه برای شاخه‌شدن به سمت BI است:
      داده‌ی لازم (sql_result / aspect_statistics / aspect_trends) از قبل
      در state موجوده، فارغ از این‌که از کدوم route اومده باشه.
    - graph.py با همون تکنیک fan-out که برای hybrid parallel استفاده شده
      (برگردوندن یک لیست از label ها به‌جای یک رشته)، وقتی wants_chart
      True باشه، هم زنجیره‌ی روایت (evidence_fusion -> insight_generator)
      و هم زنجیره‌ی BI (bi_planner -> bi_builder) رو در همون سوپراستپ فعال
      می‌کنه. نتیجه: BI به‌صورت خودکار با هر سه‌ی sql/rag/hybrid "ترکیب"
      می‌شه، بدون نسخه‌ی دوم از هیچ نودی.

دو نود این فایل:
    bi_planner (LLM):
        سوال کاربر + متادیتای تحلیلی (metrics/dimensions/time_range) +
        شکل داده‌ی موجود (ستون‌های sql_result، وجود aspect_statistics) رو
        می‌گیره و یک "chart_request" ساختاریافته برمی‌گردونه: نوع نمودار،
        عنوان، محور X/Y، و این‌که کدوم منبع (sql/absa/both) باید نمودار
        بشه. این تنها جایی‌ست که در مسیر BI از LLM استفاده می‌شه.

    bi_builder (Agent، نه LLM):
        بر اساس chart_request، مستقیماً از sql_result / aspect_statistics
        / aspect_trends / ranked_aspects یک یا چند "chart_spec" واقعی
        می‌سازه -- یک JSON سبک و مستقل از فرانت‌اند (شبیه Vega-Lite ساده‌شده:
        {chart_type, title, x_field, y_field, data}) که هر فرانت‌اندی
        (React/Plotly/ECharts/...) می‌تونه راحت رندرش کنه. اگه هم داده‌ی
        SQL هم داده‌ی ABSA موجود باشه (حالت hybrid+BI)، بیش از یک چارت
        می‌سازه و همه رو در state["dashboard"] برمی‌گردونه.
"""
from __future__ import annotations

import logging
from typing import Any

from .llm_client import call_llm_json
from .state import GraphState

logger = logging.getLogger(__name__)

# ============================================================
# BI PLANNER (LLM)
# ============================================================

BI_PLANNER_SYSTEM_PROMPT = """
تو یک برنامه‌ریز داشبورد/BI هستی. سوال مدیر + اطلاعات زمینه رو بگیر و
فقط یک JSON با این فرمت برگردون -- هیچ متن اضافه‌ای ننویس:

{
  "chart_type": "bar" | "line" | "pie" | "area" | "scatter" | "table",
  "title": "<عنوان کوتاه فارسی برای نمودار>",
  "x_field": "<نام ستون/فیلدی که باید محور X باشه، یا null>",
  "y_field": "<نام ستون/فیلدی که باید محور Y باشه، یا null>",
  "data_source": "sql" | "absa" | "both",
  "reason": "<یک جمله‌ی کوتاه توضیح چرا این نوع نمودار>"
}

قوانین:
- اگه سوال روند زمانی می‌خواد (مثلاً "روند فروش در چند ماه اخیر") ->
  chart_type = "line" یا "area".
- اگه سوال مقایسه‌ی چند دسته/برند/محصول می‌خواد -> chart_type = "bar".
- اگه سوال سهم/درصد از کل می‌خواد -> chart_type = "pie".
- اگه فقط داده‌ی SQL موجوده -> data_source = "sql".
- اگه فقط داده‌ی جنبه‌های نظرات (ABSA) موجوده -> data_source = "absa".
- اگه هر دو موجودن و سوال هم به داده‌ی عددی هم به نظرات مشتری اشاره داره
  -> data_source = "both" (یک داشبورد دو-نموداره ساخته می‌شه).
- x_field/y_field رو فقط از بین ستون‌های واقعی‌ای که در «ستون‌های نمونه‌ی
  داده‌ی SQL» بهت داده شده انتخاب کن؛ اگه معلوم نیست null بذار (bi_builder
  خودش یک fallback منطقی انتخاب می‌کنه).
"""


def _sample_sql_columns(sql_result: dict[str, Any]) -> list[str]:
    rows = (sql_result or {}).get("rows") or []
    if not rows:
        return []
    return list(rows[0].keys())


def bi_planner(state: GraphState) -> dict[str, Any]:
    question = state.get("question", "")
    metrics = state.get("metrics", [])
    dimensions = state.get("dimensions", [])
    time_range = state.get("time_range", {})

    sql_result = state.get("sql_result") or {}
    aspect_statistics = state.get("aspect_statistics") or {}
    aspect_trends = state.get("aspect_trends") or {}

    has_sql = bool(sql_result.get("rows"))
    has_absa = bool(aspect_statistics) or bool(aspect_trends)

    user_prompt = (
        f"سوال کاربر: {question}\n"
        f"متریک‌های شناسایی‌شده: {metrics}\n"
        f"بعدهای شناسایی‌شده: {dimensions}\n"
        f"بازه‌ی زمانی: {time_range}\n"
        f"آیا داده‌ی SQL موجوده: {has_sql}\n"
        f"ستون‌های نمونه‌ی داده‌ی SQL: {_sample_sql_columns(sql_result)}\n"
        f"آیا داده‌ی جنبه‌های نظرات (ABSA) موجوده: {has_absa}\n"
    )

    try:
        plan = call_llm_json(BI_PLANNER_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001 - safe_node هم می‌گیردش، ولی fallback محلی بهتره
        logger.warning("bi_planner: LLM call failed, falling back to defaults: %s", exc)
        plan = {}

    # normalize + fallback های امن -- اگه LLM چیزی نده یا چیز نامعتبر بده،
    # bi_builder نباید crash کنه.
    data_source = plan.get("data_source")
    if data_source not in ("sql", "absa", "both"):
        if has_sql and has_absa:
            data_source = "both"
        elif has_absa:
            data_source = "absa"
        else:
            data_source = "sql"

    chart_type = plan.get("chart_type")
    if chart_type not in ("bar", "line", "pie", "area", "scatter", "table"):
        chart_type = "bar"

    chart_request = {
        "chart_type": chart_type,
        "title": plan.get("title") or "",
        "x_field": plan.get("x_field"),
        "y_field": plan.get("y_field"),
        "data_source": data_source,
        "reason": plan.get("reason", ""),
    }

    return {"chart_request": chart_request}


# ============================================================
# BI BUILDER (Agent -- deterministic, no LLM)
# ============================================================

def _chart_from_sql(chart_request: dict[str, Any], sql_result: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = (sql_result or {}).get("rows") or []
    if not rows:
        return None

    columns = list(rows[0].keys())

    x_field = chart_request.get("x_field")
    if x_field not in columns:
        # fallback: اولین ستونی که عددی نیست رو به‌عنوان محور X (دسته/برچسب) بردار
        x_field = next((c for c in columns if not isinstance(rows[0][c], (int, float))), columns[0])

    y_field = chart_request.get("y_field")
    if y_field not in columns:
        # fallback: اولین ستون عددیِ متفاوت از x_field
        y_field = next(
            (c for c in columns if c != x_field and isinstance(rows[0][c], (int, float))),
            None,
        )

    if y_field is None:
        return None

    data = [{"x": row.get(x_field), "y": row.get(y_field)} for row in rows]

    return {
        "source": "sql",
        "chart_type": chart_request.get("chart_type", "bar"),
        "title": chart_request.get("title") or f"{y_field} بر اساس {x_field}",
        "x_field": x_field,
        "y_field": y_field,
        "data": data,
    }


def _chart_from_absa(
    chart_request: dict[str, Any],
    aspect_statistics: dict[str, Any],
    aspect_trends: dict[str, Any],
    ranked_aspects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if aspect_statistics:
        data = [{"x": aspect, "y": value} for aspect, value in aspect_statistics.items()]
    elif ranked_aspects:
        data = [
            {
                "x": item.get("term") or item.get("aspect") or "",
                "y": item.get("score", item.get("frequency", 0)),
            }
            for item in ranked_aspects
        ]
    elif aspect_trends:
        # هر جنبه: {"previous": ..., "current": ...} -> برای نمودار روند
        data = [
            {"x": aspect, "y": trend.get("current", trend.get("growth", 0))}
            for aspect, trend in aspect_trends.items()
        ]
    else:
        return None

    if not data:
        return None

    chart_type = chart_request.get("chart_type", "bar")
    if chart_type == "table":
        chart_type = "table"
    elif chart_request.get("data_source") == "absa" and chart_type not in ("bar", "pie"):
        # روند زمانی معنی‌داری در توزیع ساده‌ی جنبه‌ها وجود نداره؛ مگر از
        # aspect_trends اومده باشه که در اون صورت line هم منطقیه.
        chart_type = chart_request.get("chart_type", "bar")

    return {
        "source": "absa",
        "chart_type": chart_type,
        "title": chart_request.get("title") or "توزیع جنبه‌های نظرات مشتریان",
        "x_field": "aspect",
        "y_field": "value",
        "data": data,
    }


def bi_builder(state: GraphState) -> dict[str, Any]:
    chart_request = state.get("chart_request") or {}
    data_source = chart_request.get("data_source", "sql")

    sql_result = state.get("sql_result") or {}
    aspect_statistics = state.get("aspect_statistics") or {}
    aspect_trends = state.get("aspect_trends") or {}
    ranked_aspects = state.get("ranked_aspects") or []

    charts: list[dict[str, Any]] = []

    if data_source in ("sql", "both"):
        chart = _chart_from_sql(chart_request, sql_result)
        if chart:
            charts.append(chart)

    if data_source in ("absa", "both"):
        chart = _chart_from_absa(chart_request, aspect_statistics, aspect_trends, ranked_aspects)
        if chart:
            charts.append(chart)

    if not charts:
        return {
            "chart_spec": {},
            "dashboard": [],
            "errors": ["bi_builder: داده‌ی کافی برای ساخت نمودار در state موجود نبود"],
        }

    return {
        "chart_spec": charts[0],
        "dashboard": charts,
    }
