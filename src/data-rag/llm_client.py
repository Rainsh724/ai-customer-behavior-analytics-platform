## PATH: app/graph/llm_client.py
"""
یک wrapper نازک روی OpenAI-compatible API (خود OpenAI، یا هر endpoint دیگه‌ای
که همین فرمت رو پیاده کنه). اگه از provider دیگه‌ای استفاده می‌کنی (مثلاً
Anthropic مستقیم)، فقط همین دو تابع رو عوض کن -- بقیه‌ی گراف بهش وابسته نیست.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


CHAT_MODEL = os.getenv("SQL_LLM_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def call_llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """
    یک تماس LLM که مجبورش می‌کنیم فقط JSON خروجی بده (response_format json_object).
    خروجی رو parse می‌کنه؛ اگه parse نشد یک JSONDecodeError بالا میره که
    safe_node در nodes.py می‌گیردش و به state["errors"] اضافه می‌کنه.
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def embed_text(text: str) -> list[float]:
    client = get_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding
