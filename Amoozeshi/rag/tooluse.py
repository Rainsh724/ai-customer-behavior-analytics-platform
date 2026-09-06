from langchain_core.tools import tool

@tool
def get_today_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    import ast
    import operator as op

@tool
def weather(city: str, temperature: str) -> str:
    city = city.strip()
    temp = temperature.strip()
    return f"آب و هوای{city} {temp} درجه سانتیگراد است."

TOOLS = [get_today_date, calculator, weather]
