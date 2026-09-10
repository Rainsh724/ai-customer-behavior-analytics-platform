from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from tools import (
    search_wikipedia,
    get_page_summary,
    compare_topics,
    list_available_pages,
)
from memory import (
    save_search,
    save_summary,
    save_conversation,
    get_memory_context,
    search_in_memory,
)

@tool
def recall_previous_searches(query: str) -> str:
    results = search_in_memory(query)
    
    if not results:
        return f"موضوع «{query}» در حافظه پیدا نشد."
    
    output = f"نتایج یافت شده در حافظه برای «{query}»:\n\n"
    for r in results:
        if r["type"] == "search":
            output += f" جستجوی قبلی: {r['query']}\n"
            output += f"   خلاصه: {r['summary']}\n\n"
        elif r["type"] == "summary":
            output += f" خلاصه قبلی: {r['topic']}\n"
            output += f"   {r['summary']}\n\n"
    
    return output


@tool
def get_conversation_history() -> str:
    from memory import get_recent_conversations
    
    conversations = get_recent_conversations(10)
    
    if not conversations:
        return "هیچ مکالمه‌ای ذخیره نشده است."
    
    output = "مکالمات اخیر:\n"
    for conv in conversations:
        role_label = " کاربر" if conv["role"] == "user" else " دستیار"
        output += f"{role_label}: {conv['content'][:100]}\n"
    
    return output


ALL_TOOLS = [
    search_wikipedia,
    get_page_summary,
    compare_topics,
    list_available_pages,
    recall_previous_searches,
    get_conversation_history,
]

TOOL_MAP = {
    "search_wikipedia": search_wikipedia,
    "get_page_summary": get_page_summary,
    "compare_topics": compare_topics,
    "list_available_pages": list_available_pages,
    "recall_previous_searches": recall_previous_searches,
    "get_conversation_history": get_conversation_history,
}


def run_memory_agent(user_input: str, max_steps: int = 5):
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        num_ctx=4096,
    )

    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    memory_context = get_memory_context()
    
    system_content = (
        "تو یک دستیار فارسی متخصص در موضوعات هوش مصنوعی، یادگیری ماشین "
        "و پردازش زبان طبیعی هستی.\n"
        "از ابزارهای جستجو در ویکی‌پدیا و ابزارهای حافظه استفاده کن.\n"
    )
    
    if memory_context:
        system_content += f"\n اطلاعات از حافظه بلندمدت:\n{memory_context}\n"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]

    save_conversation("user", user_input)

    print(f" کاربر: {user_input}")

    for step in range(max_steps):
        print(f"\n--- مرحله {step + 1} ---")

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            final_answer = response.content
            print(f" دستیار: {final_answer}")
            
            save_conversation("assistant", final_answer)
            
            return final_answer

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f" ابزار: {tool_name}")
            print(f"   آرگومان‌ها: {tool_args}")

            tool_func = TOOL_MAP.get(tool_name)
            if tool_func:
                try:
                    result = tool_func.invoke(tool_args)
                except Exception as e:
                    result = f"خطا در اجرا: {e}"
            else:
                result = f"ابزار {tool_name} پیدا نشد."

            if tool_name == "search_wikipedia":
                save_search(
                    query=tool_args.get("query", ""),
                    results_summary=result[:300],
                    page_titles=[],
                )
            elif tool_name == "get_page_summary":
                save_summary(
                    topic=tool_args.get("page_title", ""),
                    summary=result[:300],
                    source_titles=[],
                )

            print(f"   نتیجه: {result[:200]}...")

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "Max Reached"


def main():
    print("Agent with memory")
    print("برای خروج: exit")

    while True:
        user_input = input("\n شما: ").strip()

        if user_input.lower() in ["exit", "quit", "خروج"]:
            break

        if not user_input:
            continue

        try:
            run_memory_agent(user_input)
        except Exception as e:
            print(f"خطا: {e}")


if __name__ == "__main__":
    main()