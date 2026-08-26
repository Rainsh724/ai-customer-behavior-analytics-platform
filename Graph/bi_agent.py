## PATH: app/graph/bi_agent.py
"""
پیاده‌سازی واقعی Tool_BI.

تغییر بنیادی نسبت به نسخه‌ی قبلی: دیگه لینک Power BI نمی‌سازیم. نمودار
مستقیم با کد پایتون از روی نتیجه‌ی یک کوئری SQL ساخته می‌شه.

تصمیم فنی: بدون وابستگی به پکیج plotly.
------------------------------------------------------------
Chart.js و ECharts کتابخانه‌های جاوااسکریپت‌ان -- «پیاده‌سازی‌شون در
پایتون» یعنی فقط ساختن همون JSON پیکربندی‌ای که خودشون در فرانت‌اند
انتظار دارن، نه اجرای واقعی این کتابخانه‌ها در پایتون. حتی برای Plotly
(که پکیج پایتونی هم داره)، خروجی نهایی‌ای که فرانت‌اند نیاز داره چیزی
جز یک دیکشنری {"data":[...], "layout":{...}} نیست. پس به‌جای اضافه کردن
یک وابستگی خارجی (plotly) فقط برای ساختن یک دیکشنری ساده، هر سه فرمت رو
مستقیم و دستی از روی داده‌ی خام می‌سازیم -- سبک‌تر، بدون وابستگی، و
قطعی‌تر برای تست.

خروجی این ابزار هر سه فرمت رو هم‌زمان برمی‌گردونه (chartjs_config /
echarts_option / plotly_figure) تا فرانت‌اند هرکدوم از این سه کتابخانه
رو که استفاده می‌کنه، مستقیم بتونه بدون تبدیل اضافه رندرش کنه.

مثل tool_sql، اینجا هم خودِ Agent مستقیم SQL می‌نویسه (نه یک زیرایجنت
پنهان) -- description ابزار tool_bi فقط یک ارجاع کوتاه به قوانین/
اسکیمای tool_sql داره (نه تکرار کامل SCHEMA_CONTEXT)، چون هر دو تعریف
ابزار در یک تماس واحد به مدل داده می‌شن و مدل از قبل اون اسکیما رو در
همون تماس دیده -- تکرارش فقط توکن اضافه مصرف می‌کنه.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .sql_agent import run_sql_tool

logger = logging.getLogger(__name__)

VALID_CHART_TYPES = {"bar", "line", "pie", "scatter", "area"}


def _json_safe(value: Any) -> Any:
    """psycopg2 مقادیر Decimal/date/datetime برمی‌گردونه که خودشون
    مستقیم JSON-serializable نیستن؛ اینجا به float/رشته‌ی ISO تبدیل
    می‌شن."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _extract_labels_values(
    rows: list[dict[str, Any]],
    x_field: str | None,
    y_field: str | None,
) -> tuple[str, str, list[Any], list[Any]]:
    if not rows:
        raise ValueError("داده‌ای برای رسم نمودار برنگشت.")

    columns = list(rows[0].keys())

    if x_field not in columns:
        # fallback: اولین ستونی که عددی نیست -> برچسب/دسته
        x_field = next((c for c in columns if not isinstance(rows[0][c], (int, float, Decimal))), columns[0])

    if y_field not in columns:
        # fallback: اولین ستون عددیِ متفاوت از x_field
        y_field = next(
            (c for c in columns if c != x_field and isinstance(rows[0][c], (int, float, Decimal))),
            None,
        )

    if y_field is None:
        raise ValueError("ستون عددی مناسب برای محور Y پیدا نشد.")

    labels = [_json_safe(row.get(x_field)) for row in rows]
    values = [_json_safe(row.get(y_field)) for row in rows]
    return x_field, y_field, labels, values


# ============================================================
# سه سازنده‌ی فرمت -- هرکدوم یک دیکشنری خالص پایتون، بدون هیچ import
# خارجی. هر سه از روی همون labels/values ساخته می‌شن.
# ============================================================

def _build_chartjs_config(chart_type: str, title: str, x_field: str, y_field: str,
                            labels: list[Any], values: list[Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "type": chart_type if chart_type != "area" else "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": title,
                "data": values,
                **({"fill": True} if chart_type == "area" else {}),
            }],
        },
        "options": {
            "plugins": {"title": {"display": True, "text": title}},
        },
    }
    if chart_type not in ("pie",):
        config["options"]["scales"] = {
            "x": {"title": {"display": True, "text": x_field}},
            "y": {"title": {"display": True, "text": y_field}},
        }
    return config


def _build_echarts_option(chart_type: str, title: str, x_field: str, y_field: str,
                            labels: list[Any], values: list[Any]) -> dict[str, Any]:
    if chart_type == "pie":
        return {
            "title": {"text": title},
            "series": [{
                "type": "pie",
                "data": [{"name": label, "value": value} for label, value in zip(labels, values)],
            }],
        }

    echarts_type = "scatter" if chart_type == "scatter" else ("line" if chart_type in ("line", "area") else "bar")
    series_entry: dict[str, Any] = {"type": echarts_type, "data": values}
    if chart_type == "area":
        series_entry["areaStyle"] = {}

    return {
        "title": {"text": title},
        "xAxis": {"type": "category", "name": x_field, "data": labels},
        "yAxis": {"type": "value", "name": y_field},
        "series": [series_entry],
    }


def _build_plotly_figure(chart_type: str, title: str, x_field: str, y_field: str,
                           labels: list[Any], values: list[Any]) -> dict[str, Any]:
    if chart_type == "pie":
        trace = {"type": "pie", "labels": labels, "values": values}
        layout: dict[str, Any] = {"title": title}
    elif chart_type in ("line", "area"):
        trace = {"type": "scatter", "mode": "lines+markers", "x": labels, "y": values}
        if chart_type == "area":
            trace["fill"] = "tozeroy"
        layout = {"title": title, "xaxis": {"title": x_field}, "yaxis": {"title": y_field}}
    elif chart_type == "scatter":
        trace = {"type": "scatter", "mode": "markers", "x": labels, "y": values}
        layout = {"title": title, "xaxis": {"title": x_field}, "yaxis": {"title": y_field}}
    else:  # bar (پیش‌فرض)
        trace = {"type": "bar", "x": labels, "y": values}
        layout = {"title": title, "xaxis": {"title": x_field}, "yaxis": {"title": y_field}}

    return {"data": [trace], "layout": layout}


def run_bi_tool(
    sql: str,
    chart_type: str = "bar",
    title: str | None = None,
    x_field: str | None = None,
    y_field: str | None = None,
) -> dict[str, Any]:
    """
    ورودی: sql (کوئری SELECT که خودِ Agent -- طبق همون قوانین tool_sql --
    نوشته و داده‌ی نمودار رو برمی‌گردونه)، chart_type، و اختیاری
    title/x_field/y_field (اگه ندی، از روی ستون‌های نتیجه حدس زده می‌شه).
    خروجی: dict شامل هر سه فرمت (chartjs_config / echarts_option /
    plotly_figure) -- مستقیم به‌صورت JSON در پیام "tool" به Agent
    برمی‌گرده و از همون‌جا به کاربر/فرانت‌اند می‌رسه.
    """
    if chart_type not in VALID_CHART_TYPES:
        logger.warning("run_bi_tool: chart_type نامعتبر '%s' -- fallback به 'bar'", chart_type)
        chart_type = "bar"

    sql_result = run_sql_tool(sql)
    if "error" in sql_result:
        return {"error": f"داده‌ی نمودار قابل‌دریافت نبود: {sql_result['error']}"}

    rows = sql_result.get("rows") or []
    if not rows:
        return {"error": "کوئری هیچ ردیفی برای رسم نمودار برنگردوند."}

    try:
        x_field, y_field, labels, values = _extract_labels_values(rows, x_field, y_field)
    except ValueError as exc:
        return {"error": str(exc)}

    chart_title = title or f"{y_field} بر اساس {x_field}"

    return {
        "chart_type": chart_type,
        "title": chart_title,
        "x_field": x_field,
        "y_field": y_field,
        "row_count": len(rows),
        "chartjs_config": _build_chartjs_config(chart_type, chart_title, x_field, y_field, labels, values),
        "echarts_option": _build_echarts_option(chart_type, chart_title, x_field, y_field, labels, values),
        "plotly_figure": _build_plotly_figure(chart_type, chart_title, x_field, y_field, labels, values),
    }
