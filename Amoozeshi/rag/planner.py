import json
from functools import lru_cache
from typing import TypedDict, List
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from tools import (
    search_wikipedia,
    get_page_summary,
    compare_topics,
    list_available_pages,
)
from memory import (
    save_search,
    save_summary,
    save_comparison,
)


OLLAMA_MODEL = "qwen2.5:7b"
MAX_PLAN_STEPS = 6


class PlannerState(TypedDict, total=False):
    task: str
    plan: List[dict]
    current_step: int
    step_results: List[dict]
    final_answer: str
    done: bool


@lru_cache(maxsize=1)
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_ctx=8192,
    )


PLANNER_SYSTEM_PROMPT = """تو یک برنامه‌ریز دقیق هستی.
وظیفه کاربر را به مراحل ساده تقسیم کن.
هر مرحله باید شامل یک ابزار و آرگومان‌هایش باشد.
فقط و فقط یک آرایه معتبر برگردان.
هیچ متن اضافه‌ای ننویس.

فرمت هر مرحله:
{"step": 1, "tool": "tool_name", "args": {...}, "description": "توضیح فارسی"}

ابزارهای موجود:
- search_wikipedia: جستجو در ویکی‌پدیا (آرگومان: query, top_k)
- get_page_summary: خلاصه یک صفحه (آرگومان: page_title)
- compare_topics: مقایسه دو موضوع (آرگومان: topic1, topic2)
- list_available_pages: لیست صفحات موجود (بدون آرگومان)
- summarize_text: خلاصه‌سازی متن (آرگومان: text)
- save_to_file: ذخیره در فایل (آرگومان: text, filename)
- final_answer: پاسخ نهایی (آرگومان: answer)

نکات:
- اگر نتیجه یک ابزار برای مرحله بعد لازم است، در آرگومان‌ها از {{prev_result}} استفاده کن.
- مرحله آخر باید پاسخ نهایی یا ذخیره فایل باشد.
- حداکثر ۵ مرحله بساز."""


def planner_node(state: PlannerState):
    task = state.get("task", "")

    response = get_llm().invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"وظیفه: {task}"),
    ])

    raw = response.content if isinstance(response.content, str) else str(response.content)
    
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            parsed = [{"step": 1, "tool": "final_answer", "args": {"answer": raw}, "description": "پاسخ مستقیم"}]
    except json.JSONDecodeError:
        import re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = [{"step": 1, "tool": "final_answer", "args": {"answer": raw}, "description": "پاسخ مستقیم"}]
        else:
            parsed = [{"step": 1, "tool": "final_answer", "args": {"answer": raw}, "description": "پاسخ مستقیم"}]

    print(f"\n برنامه ساخته شد ({len(parsed)} مرحله):")
    for step in parsed:
        print(f"   {step.get('step', '?')}. {step.get('description', '?')} → {step.get('tool', '?')}")

    return {
        "plan": parsed,
        "current_step": 0,
        "step_results": [],
        "done": False,
    }


def executor_node(state: PlannerState):
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    step_results = list(state.get("step_results", []))

    if current_step >= len(plan):
        return {
            "done": True,
            "final_answer": "All Steps Done",
        }

    step = plan[current_step]
    tool_name = step.get("tool", "final_answer")
    args = step.get("args", {})
    description = step.get("description", "")

    print(f"\ اجرای مرحله {current_step + 1}: {description}")
    print(f"   ابزار: {tool_name}")
    print(f"   آرگومان‌ها: {args}")

    if step_results:
        prev_result = step_results[-1].get("result", "")
        args_str = json.dumps(args, ensure_ascii=False)
        if "{{prev_result}}" in args_str:
            args_str = args_str.replace("{{prev_result}}", prev_result)
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                pass

    if tool_name == "final_answer":
        answer = args.get("answer", "") if isinstance(args, dict) else str(args)
        if step_results and "{{prev_result}}" in str(answer):
            prev_result = step_results[-1].get("result", "")
            answer = str(answer).replace("{{prev_result}}", prev_result)

        step_results.append({
            "step": current_step + 1,
            "tool": tool_name,
            "description": description,
            "result": answer,
        })

        return {
            "step_results": step_results,
            "current_step": current_step + 1,
            "done": True,
            "final_answer": answer,
        }

    result = ""
    
    if tool_name == "search_wikipedia":
        result = search_wikipedia.invoke(args)
        save_search(
            query=args.get("query", ""),
            results_summary=result[:300],
            page_titles=[],
        )
    elif tool_name == "get_page_summary":
        result = get_page_summary.invoke(args)
        save_summary(
            topic=args.get("page_title", ""),
            summary=result[:300],
            source_titles=[],
        )
    elif tool_name == "compare_topics":
        result = compare_topics.invoke(args)
        save_comparison(
            topic1=args.get("topic1", ""),
            topic2=args.get("topic2", ""),
            result=result[:300],
        )
    elif tool_name == "list_available_pages":
        result = list_available_pages.invoke(args)
    elif tool_name == "summarize_text":
        text = args.get("text", "")
        if step_results and "{{prev_result}}" in text:
            text = step_results[-1].get("result", "")
        
        summary_response = get_llm().invoke([
            SystemMessage(content="تو یک خلاصه‌ساز فارسی هستی. متن را در ۳ جمله خلاصه کن."),
            HumanMessage(content=text[:2000]),
        ])
        result = summary_response.content
    elif tool_name == "save_to_file":
        text = args.get("text", "")
        filename = args.get("filename", "report.txt")
        if step_results and "{{prev_result}}" in str(text):
            text = step_results[-1].get("result", "")
        
        output_dir = Path("agent_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        output_path.write_text(str(text), encoding="utf-8")
        result = f"فایل ذخیره شد: {output_path}"
    else:
        result = f"ابزار {tool_name} شناخته نشد."

    print(f"   نتیجه: {str(result)[:200]}...")

    step_results.append({
        "step": current_step + 1,
        "tool": tool_name,
        "description": description,
        "result": str(result),
    })

    return {
        "step_results": step_results,
        "current_step": current_step + 1,
        "done": False,
    }


def summarizer_node(state: PlannerState):
    step_results = state.get("step_results", [])
    task = state.get("task", "")

    if state.get("final_answer"):
        return {"done": True}

    if not step_results:
        return {"final_answer": "هیچ نتیجه‌ای تولید نشد.", "done": True}

    results_text = "\n".join(
        f"مرحله {r['step']} ({r['description']}): {r['result'][:300]}"
        for r in step_results
    )

    response = get_llm().invoke([
        SystemMessage(content=(
            "تو یک جمع‌بندی‌کننده فارسی هستی. "
            "نتایج مراحل مختلف تحقیق را به یک گزارش منسجم تبدیل کن."
        )),
        HumanMessage(content=(
            f"وظیفه اصلی: {task}\n\n"
            f"نتایج مراحل:\n{results_text}\n\n"
            f"گزارش نهایی:"
        )),
    ])

    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "final_answer": answer.strip(),
        "done": True,
    }


def should_continue(state: PlannerState):
    if state.get("done"):
        return "summarize"
    return "executor"


def create_planner_graph():
    workflow = StateGraph(PlannerState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("summarize", summarizer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")

    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "summarize": "summarize",
        },
    )

    workflow.add_edge("summarize", END)

    return workflow.compile()