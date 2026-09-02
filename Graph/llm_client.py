## PATH: app/graph/llm_client.py
"""
یک wrapper نازک روی OpenAI-compatible API.

سه نوع تماس:
    call_llm_json:       خروجی اجباراً JSON (برای مولد SQL و امثالش).
    call_llm_with_tools:  تماس اصلی "مغز" ایجنت -- پیام‌ها + تعریف ابزارها
                          رو می‌ده، مدل یا مستقیم جواب متنی می‌ده یا
                          tool_calls (حتی چندتایی/موازی) برمی‌گردونه.
    embed_text:           امبدینگ محلی برای جست‌وجوی برداری (بدون تغییر).

نکته‌ی مهم درباره‌ی embed_text:
تمام کامنت‌های موجود در comments_embedding با مدل محلی
"intfloat/multilingual-e5-base" (۷۶۸بعدی، با پیشوند "query: "/"passage: ")
امبد شدن (notebooks/6-Embed_Comments.ipynb). embed_text اینجا عمداً از
همون مدل محلی استفاده می‌کنه، نه از OpenAI embeddings API -- چون این دو
embedding space قابل‌قیاس نیستن.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


CHAT_MODEL = os.getenv(
    "AGENT_LLM_MODEL",
    "openai/gpt-oss-120b",
)

# باید دقیقاً همون مدلی باشه که comments_embedding باهاش ساخته شده.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-base",
)

# اگه Groq بگه "دوباره تلاش کن" (429)، به‌جای خطا دادن فوری، چند بار با
# فاصله‌ی افزایشی (exponential backoff) دوباره امتحان می‌کنیم -- چون
# rate limit یک خطای *موقت* و رایجه (خصوصاً روی tier رایگان/on_demand)،
# نه یک خطای واقعی در کد یا کوئری. اگه بعد از همه‌ی تلاش‌ها بازم رد بشه،
# خطا رو بالا می‌فرستیم تا safe_node (در nodes.py) بگیرتش.
MAX_RATE_LIMIT_RETRIES = int(os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "4"))
RATE_LIMIT_BASE_DELAY_SECONDS = float(os.getenv("LLM_RATE_LIMIT_BASE_DELAY", "2.0"))


def _call_with_rate_limit_retry(fn: Callable[[], _T]) -> _T:
    last_exc: RateLimitError | None = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return fn()
        except RateLimitError as exc:
            last_exc = exc
            if attempt >= MAX_RATE_LIMIT_RETRIES:
                break
            wait_seconds = RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning(
                "Rate limit از Groq (تلاش %d/%d) -- %.1f ثانیه صبر می‌کنیم و دوباره امتحان می‌کنیم...",
                attempt + 1,
                MAX_RATE_LIMIT_RETRIES + 1,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc


def call_llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """
    یک تماس LLM که مجبورش می‌کنیم فقط JSON خروجی بده (response_format json_object).
    برای ابزارهای داخلی (مثل مولد SQL) استفاده می‌شه، نه برای خودِ Agent.
    """
    client = get_client()

    def _do_call():
        return client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    resp = _call_with_rate_limit_retry(_do_call)
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def message_to_dict(msg: Any) -> dict[str, Any]:
    """
    یک ChatCompletionMessage (شیء OpenAI SDK) رو به یک dict ساده و
    JSON-serializable تبدیل می‌کنه، دقیقاً به همون فرمتی که خودِ OpenAI
    API برای پیام‌های ورودی بعدی انتظار داره -- چون این پیام دوباره به
    state["messages"] اضافه می‌شه و در تماس بعدی به مدل پس داده می‌شه.
    """
    out: dict[str, Any] = {
        "role": msg.role,
        "content": msg.content,
    }
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def call_llm_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """
    تماس اصلی نود Agent. tool_choice="auto" یعنی مدل خودش تصمیم می‌گیره
    صفر، یک، یا چند ابزار رو (به‌صورت موازی) صدا بزنه یا مستقیم جواب
    متنی نهایی بده -- دقیقاً همون مکانیزمی که "اجرای موازی" و "حلقه‌ی
    اصلاح خطا" در سند معماری رو، بدون هیچ سیم‌کشی دستی اضافه‌ای در گراف،
    پیاده می‌کنه. parallel_tool_calls=True (پیش‌فرض API) اجازه می‌ده در
    یک پاسخ چند tool_call هم‌زمان برگرده (مثلاً هم SQL هم RAG).

    tool_choice="none" برای finalize اجباری استفاده می‌شه: وقتی سقف
    iterations رد شده و می‌خوایم مدل مجبور به جمع‌بندی متنی بشه، نه
    درخواست ابزار جدید.

    خروجی از message_to_dict عبور کرده -- یعنی از قبل به فرمت dict
    ساده‌ست، آماده برای append شدن به state["messages"].
    """
    client = get_client()
    kwargs: dict[str, Any] = dict(
        model=CHAT_MODEL,
        temperature=0,
        messages=messages,
        tool_choice=tool_choice,
    )

    if tool_choice != "none":
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = True

    resp = _call_with_rate_limit_retry(lambda: client.chat.completions.create(**kwargs))

    return message_to_dict(
        resp.choices[0].message
    )

_embedding_model = None
_embedding_model_lock = threading.Lock()


def _get_embedding_model():
    """
    Lazy-load the local sentence-transformers model (heavy import, keep it
    optional).

    نکته درباره‌ی کند بودن اولین بار: sentence-transformers حتی وقتی وزن‌های
    مدل قبلاً روی دیسک cache شدن، پیش‌فرض یک تماس شبکه‌ای به Hugging Face
    Hub می‌زنه (برای چک کردن نسخه/متادیتا) -- دقیقاً همون چیزی که در لاگ
    دیدید ("unauthenticated requests to HF Hub"). این تماس شبکه‌ای هم کند
    و هم به rate limit عمومی HF محدوده. دو راه‌حل:

        ۱. HF_TOKEN رو در .env بذار -- باعث میشه این تماس‌ها authenticated
           باشن (سریع‌تر و بدون محدودیت نرخ عمومی).
        ۲. بعد از اولین اجرای موفق (که مدل واقعاً دانلود و cache شده)،
           HF_HUB_OFFLINE=1 رو ست کن -- کاملاً از زدن هر تماس شبکه‌ای صرف‌نظر
           می‌کنه و مستقیم از cache محلی می‌خونه (سریع‌ترین حالت ممکنه).

    نکته‌ی مهم‌تر درباره‌ی "هر بار طول می‌کشه": توی یک پردازه‌ی زنده (مثلاً
    وقتی روی FastAPI/uvicorn بالا میاد)، این تابع فقط یک‌بار واقعاً مدل رو
    لود می‌کنه (globals + lock بالا دقیقاً برای همینه) -- دفعات بعد از
    همون شیء در حافظه استفاده می‌شه. اگه همچنان "هر بار" کند حس می‌شه،
    یعنی دارید هر بار یک پردازه‌ی پایتون تازه اجرا می‌کنید (مثلاً هر بار
    python3 main.py می‌زنید) -- راه‌حلش preload_embedding_model() پایینه:
    یک‌بار در شروع سرویس (نه در وسط اولین درخواست کاربر) صداش بزن تا
    کاربر منتظر لود مدل نمونه.
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def preload_embedding_model() -> None:
    """
    فراخوانی اختیاری برای لود کردن زودهنگام مدل embedding -- مثلاً در
    startup سرویس FastAPI (نگاه کن به app/api.py) یا در ابتدای main.py،
    تا لود کردن مدل (که چند ثانیه طول می‌کشه) وسط اولین سوال کاربر اتفاق
    نیفته و کاربر منتظر نمونه.
    """
    _get_embedding_model()


def embed_text(text: str) -> list[float]:
    """
    Embeds a SEARCH QUERY (not a document) using the same local model the
    comment corpus was embedded with. The "query: " prefix matches the E5
    convention used in notebooks/6-Embed_Comments.ipynb (is_query=True).
    """
    model = _get_embedding_model()
    vec = model.encode(
        "query: " + text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.tolist()
