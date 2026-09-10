import json
from pathlib import Path
from functools import lru_cache
from typing import Optional
from langchain_core.tools import tool

@lru_cache(maxsize=1)
def load_chunks() -> list:
    chunks_file = Path("dataset/chunks.jsonl")
    if not chunks_file.exists():
        return []
    
    chunks = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


@lru_cache(maxsize=1)
def load_pages() -> list:
    pages_file = Path("data/pages.jsonl")
    if not pages_file.exists():
        return []
    
    pages = []
    with pages_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


@lru_cache(maxsize=1)
def get_vectorstore():
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        faiss_dir = Path("faiss")
        if not faiss_dir.exists():
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={"normalize_embeddings": True},
        )

        return FAISS.load_local(
            str(faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None

@tool
def search_wikipedia(query: str, top_k: int = 3) -> str:
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        return "first run build_index"

    try:
        docs = vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        return f"error in search: {e}"

    if not docs:
        return "Nothing Foubd"

    results = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")
        text = doc.page_content[:400]
        results.append(
            f"[{i}] عنوان: {title}\n"
            f"منبع: {source}\n"
            f"متن: {text}"
        )

    return "\n\n---\n\n".join(results)


@tool
def get_page_summary(page_title: str) -> str:
    pages = load_pages()
    
    for page in pages:
        title = page.get("title", "")
        if page_title.strip() in title or title in page_title.strip():
            text = page.get("text", "")
            summary = text[:500]
            return (
                f"عنوان: {title}\n"
                f"منبع: {page.get('source', '')}\n"
                f"خلاصه:\n{summary}"
            )
    
    chunks = load_chunks()
    for chunk in chunks:
        if page_title.strip() in chunk.get("title", ""):
            return (
                f"عنوان: {chunk.get('title', '')}\n"
                f"متن: {chunk.get('text', '')[:400]}"
            )
    
    return f"not page found «{page_title}» "


@tool
def compare_topics(topic1: str, topic2: str) -> str:
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        return "first run build index"

    try:
        docs1 = vectorstore.similarity_search(topic1, k=2)
        docs2 = vectorstore.similarity_search(topic2, k=2)
    except Exception as e:
        return f"خطا در جستجو: {e}"

    result = f"مقایسه «{topic1}» و «{topic2}»:\n\n"

    result += f"--- {topic1} ---\n"
    for doc in docs1:
        result += f"عنوان: {doc.metadata.get('title', '')}\n"
        result += f"متن: {doc.page_content[:200]}\n\n"

    result += f"--- {topic2} ---\n"
    for doc in docs2:
        result += f"عنوان: {doc.metadata.get('title', '')}\n"
        result += f"متن: {doc.page_content[:200]}\n\n"

    return result


@tool
def list_available_pages() -> str:
    pages = load_pages()
    
    if not pages:
        return "no pages found."
    
    result = "صفحات موجود در دیتاست:\n"
    for i, page in enumerate(pages, 1):
        title = page.get("title", "")
        text_len = len(page.get("text", ""))
        result += f"{i}. {title} ({text_len} کاراکتر)\n"
    
    return result


WIKI_TOOLS = [
    search_wikipedia,
    get_page_summary,
    compare_topics,
    list_available_pages,
]