## PATH: app/graph/llm_client.py
"""
یک wrapper نازک روی OpenAI-compatible API برای تولید متن/JSON (فقط برای
sql_agent و retrieval_planner استفاده می‌شه). اگه از provider دیگه‌ای
استفاده می‌کنی (مثلاً Anthropic)، فقط `call_llm_json` رو عوض کن.

نکته‌ی مهم درباره‌ی embed_text:
تمام کامنت‌های موجود در comments_embedding با مدل محلی
"intfloat/multilingual-e5-base" (۷۶۸بعدی، با پیشوند "query: "/"passage: ")
امبد شدن (notebooks/6-Embed_Comments.ipynb). embed_text اینجا عمداً از
همون مدل محلی استفاده می‌کنه، نه از OpenAI embeddings API -- چون این دو
embedding space قابل‌قیاس نیستن و اگه اینجا از یک مدل دیگه استفاده بشه،
جست‌وجوی شباهت یا اصلاً اجرا نمی‌شه (dimension mismatch در pgvector) یا
نتایج کاملاً بی‌معنی برمی‌گردونه.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


CHAT_MODEL = os.getenv("SQL_LLM_MODEL", "gpt-4.1")

# باید دقیقاً همون مدلی باشه که comments_embedding باهاش ساخته شده.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")


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


_embedding_model = None
_embedding_model_lock = threading.Lock()


def _get_embedding_model():
    """Lazy-load the local sentence-transformers model (heavy import, keep it optional)."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """
    Embeds a SEARCH QUERY (not a document) using the same local model the
    comment corpus was embedded with. The "query: " prefix matches the E5
    convention used in notebooks/6-Embed_Comments.ipynb (is_query=True) --
    dropping it would still run, but would degrade retrieval quality
    because the model was trained expecting that prefix.
    """
    model = _get_embedding_model()
    vec = model.encode(
        "query: " + text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.tolist()
