import json
from pathlib import Path
from typing import TypedDict, List
from functools import lru_cache
from hazm import word_tokenize
from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

BASE_DIR = Path(__file__).resolve().parent
ADV_DIR = BASE_DIR / "advance"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "qwen-local"

class RAGState(TypedDict, total=False):
    question: str
    context: List[dict]
    answer: str

def persian_preprocess(text: str):
    return word_tokenize(text)

# @lru_cache(maxsize=1)
# def get_hybrid_retriever():
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
#     faiss_vs = FAISS.load_local(
#     str(ADV_DIR / "faiss"),
#     embeddings,
#     allow_dangerous_deserialization=True
#     )
#     faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": 3})
    
#     docs = []
#     with (ADV_DIR / "chunks_with_meta.jsonl").open("r", encoding="utf-8") as f:
#         for line in f:
#             data = json.loads(line)
#             from langchain_core.documents import Document
#             docs.append(Document(page_content=data["text"], metadata=data["metadata"]))
            
#     bm25_retriever = BM25Retriever.from_documents(
#         docs, 
#         preprocess_func=persian_preprocess, 
#         k=3
#     )
    
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, faiss_retriever],
#         weights=[0.4, 0.6] # وزن بیشتر برای معنایی، اما کلیدواژه هم تاثیر دارد
#     )
#     return ensemble_retriever

@lru_cache(maxsize=1)
def get_retriever():
    docs = []

    chunks_path = ADV_DIR / "advancechunks.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_path}"
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            docs.append(
                Document(
                    page_content=data["text"],
                    metadata=data["metadata"]
                )
            )

    if not docs:
        raise ValueError("No documents found in advancechunks.jsonl")

    return BM25Retriever.from_documents(
        docs,
        preprocess_func=persian_preprocess,
        k=3
    )

# @lru_cache(maxsize=1)
# def get_chain():
#     llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1, num_ctx=4096)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "تو یک دستیار هوشمند فارسی هستی. فقط بر اساس زمینه پاسخ بده. "
#                    "در انتهای پاسخ خود، حتماً «منبع» (عنوان و لینک) را ذکر کن."),
#         ("human", "زمینه:\n{context}\n\nسؤال:\n{question}\n\nپاسخ:")
#     ])
#     return prompt | llm | StrOutputParser()

def call_ollama(context: str, question: str) -> str:
    system_prompt = (
        "تو یک دستیار هوشمند فارسی هستی. "
        "فقط و فقط بر اساس زمینه ارائه‌شده پاسخ بده. "
        "اگر اطلاعات کافی در زمینه وجود ندارد، صادقانه بگو اطلاعات کافی پیدا نشد. "
        "در انتهای پاسخ، منبع را با عنوان و لینک ذکر کن."
    )

    user_prompt = (
        f"زمینه:\n{context}\n\n"
        f"سؤال:\n{question}\n\n"
        "پاسخ:"
    )

    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096
            }
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]

# def retrieve(state: RAGState):
#     question = state["question"]
#     retriever = get_hybrid_retriever()
    
#     docs = retriever.invoke(question)
    
#     context = []
#     for doc in docs:
#         meta = doc.metadata
#         context.append({
#             "text": doc.page_content,
#             "title": meta.get("title", "نامشخص"),
#             "url": meta.get("source_url", "#")
#         })
        
#     return {"context": context}

def retrieve(state: RAGState):
    question = state["question"]

    retriever = get_retriever()

    docs = retriever.invoke(question)

    context = []

    for doc in docs:
        meta = doc.metadata

        context.append({
            "text": doc.page_content,
            "title": meta.get("title", "نامشخص"),
            "url": meta.get("source_url", "#")
        })

    return {"context": context}

def format_context_with_metadata(context: List[dict]) -> str:
    parts = []
    for i, item in enumerate(context, 1):
        parts.append(
            f"[{i}] عنوان: {item['title']}\n"
            f"لینک: {item['url']}\n"
            f"متن: {item['text']}"
        )
    return "\n\n".join(parts)

def generate(state: RAGState):
    context = state.get("context", [])
    question = state["question"]
    
    if not context:
        return {"answer": "پاسخی یافت نشد."}
        
    context_text = format_context_with_metadata(context)
    # answer = get_chain().invoke({"context": context_text, "question": question})
    answer = call_ollama(
    context=context_text,
    question=question
    )
    return {"answer": answer.strip()}

def create_advanced_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()