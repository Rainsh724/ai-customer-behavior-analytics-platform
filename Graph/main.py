## PATH: app/main.py
from __future__ import annotations

from app.graph.graph import get_graph
from app.graph.state import GraphState


def run(question: str) -> GraphState:
    graph = get_graph()

    initial_state: GraphState = {
        "question": question,
        "errors": [],
    }

    return graph.invoke(initial_state)


def main() -> None:
    # نمونه‌ی بدون درخواست نمودار (رفتار قبلی، بدون تغییر)
    result = run("چرا فروش محصول X در ماه جاری کاهش پیدا کرده؟")

    print("\nFINAL ANSWER:")
    print(result.get("final_answer"))

    print("\nROUTE:")
    print(result.get("route"))

    print("\nHYBRID MODE:")
    print(result.get("hybrid_mode"))

    print("\nWANTS CHART:")
    print(result.get("wants_chart"))

    print("\nVALIDATION:")
    print(result.get("validation"))

    errors = result.get("errors") or []
    if errors:
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")

    # نمونه‌ی BI: همون سوال، ولی این‌بار درخواست نمودار هم شده -- query_router
    # علاوه بر route (sql/rag/hybrid)، wants_chart=True هم تشخیص می‌ده و
    # graph.py بعد از evidence_normalizer به‌صورت موازی مسیر BI رو هم فعال
    # می‌کنه.
    bi_result = run("نمودار روند فروش محصول X در ۶ ماه اخیر رو نشون بده")

    print("\n\n--- نمونه‌ی BI ---")
    print("\nWANTS CHART:")
    print(bi_result.get("wants_chart"))

    print("\nCHART SPEC:")
    print(bi_result.get("chart_spec"))

    print("\nDASHBOARD:")
    print(bi_result.get("dashboard"))


if __name__ == "__main__":
    main()
