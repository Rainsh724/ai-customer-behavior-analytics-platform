from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from advanced_session1_wiki_tools import (
    WIKI_TOOLS,
    search_wikipedia,
    get_page_summary,
    compare_topics,
    list_available_pages,
)


TOOL_MAP = {
    "search_wikipedia": search_wikipedia,
    "get_page_summary": get_page_summary,
    "compare_topics": compare_topics,
    "list_available_pages": list_available_pages,
}


def run_wiki_agent(user_task: str, max_steps: int = 5):
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        num_ctx=8192,
    )

    llm_with_tools = llm.bind_tools(WIKI_TOOLS)

    messages = [
        SystemMessage(content=(
            "تو یک دستیار فارسی متخصص در موضوعات هوش مصنوعی، یادگیری ماشین "
            "و پردازش زبان طبیعی هستی. برای پاسخ به سؤالات از ابزارهای "
            "جستجو در ویکی‌پدیا استفاده کن."
        )),
        HumanMessage(content=user_task),
    ]

    print(f"سؤال: {user_task}")

    for step in range(max_steps):
        print(f"\n--- مرحله {step + 1} ---")

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            print(f"پاسخ نهایی:\n{response.content}")
            return response.content

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"ابزار: {tool_name}")
            print(f"آرگومان‌ها: {tool_args}")

            tool_func = TOOL_MAP.get(tool_name)
            if tool_func:
                try:
                    result = tool_func.invoke(tool_args)
                except Exception as e:
                    result = f"خطا در اجرا: {e}"
            else:
                result = f"ابزار {tool_name} پیدا نشد."

            print(f"   نتیجه: {result[:200]}...")

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "Reach Max"


def main():
    tasks = [
        "چه صفحاتی در دیتاست وجود دارد؟",
        "هوش مصنوعی چیست؟",
        "یادگیری ماشین چه ارتباطی با هوش مصنوعی دارد؟",
        "پردازش زبان طبیعی را توضیح بده.",
        "هوش مصنوعی و یادگیری ماشین را مقایسه کن.",
    ]

    for task in tasks:
        run_wiki_agent(task)
        print("\n")


if __name__ == "__main__":
    main()