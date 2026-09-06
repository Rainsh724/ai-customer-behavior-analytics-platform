
    followup = run(
        "چرا؟",
        chat_id="test-top-selling-product"
    )

    print("\n\n--- سوال ادامه‌دار با حافظه‌ی Postgres ---")
    print(followup.get("final_answer"))


    chart_result = run(
        "نمودار فروش ماهانه ده محصول پرفروش‌تر را در 6 ماه اخیر نشان بده",
        chat_id="test-top-selling-product"
    )

    print("\n\n--- نمونه‌ی نمودار ---")
    print(chart_result.get("final_answer"))
