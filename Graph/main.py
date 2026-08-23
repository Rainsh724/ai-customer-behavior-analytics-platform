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
    result = run("چرا فروش محصول X در ماه جاری کاهش پیدا کرده؟")

    print("\nFINAL ANSWER:")
    print(result.get("final_answer"))

    print("\nROUTE:")
    print(result.get("route"))

    print("\nHYBRID MODE:")
    print(result.get("hybrid_mode"))

    print("\nVALIDATION:")
    print(result.get("validation"))

    errors = result.get("errors") or []
    if errors:
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
