from planner import create_planner_graph


def main():
    graph = create_planner_graph()

    # سناریوهای مختلف روی دیتاست ویکی‌پدیا
    tasks = [
        "درباره هوش مصنوعی تحقیق کن و نتیجه را خلاصه کن.",
        "هوش مصنوعی و یادگیری ماشین را مقایسه کن و گزارش بنویس.",
        "درباره پردازش زبان طبیعی جستجو کن، خلاصه کن و در نلپ_گزارش.txt ذخیره کن.",
        "لیست صفحات موجود را بگو و درباره هر کدام یک جمله خلاصه بنویس.",
    ]

    for task in tasks:
        print(f" وظیفه: {task}")

        try:
            result = graph.invoke({
                "task": task,
                "plan": [],
                "current_step": 0,
                "step_results": [],
                "final_answer": "",
                "done": False,
            })

            print(f" گزارش نهایی:")
            print(result.get("final_answer", ""))
            print(f"\n تعداد مراحل اجرا شده: {len(result.get('step_results', []))}")

            for step in result.get("step_results", []):
                print(f"\n  مرحله {step['step']}: {step['description']}")
                print(f"  ابزار: {step['tool']}")
                print(f"  نتیجه: {step['result'][:150]}...")

        except Exception as e:
            print(f"\n خطا: {e}")
            import traceback
            traceback.print_exc()

        print()


if __name__ == "__main__":
    main()