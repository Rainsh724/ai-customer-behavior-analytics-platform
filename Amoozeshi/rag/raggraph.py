from pathlib import Path
from typing import TypedDict, List
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langgraph.graph import StateGraph, START, END

FAISS_DIR = Path("Amoozeshi/rag/dataset/faiss_index")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "qwen-local"

class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    context: List[str]
    source: List[str]
    answer: str

@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return vectorstore

@lru_cache(maxsize=1)
def get_chain() -> ChatOllama:
    chain = ChatOllama(model=OLLAMA_MODEL, temperature=0.3, num_ctx=4096)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "تو یک دستیار هوشمند هستی که به سوالات کاربران پاسخ می‌دهی. پاسخ‌هایت باید دقیق، مختصر و مفید و فقط براساس پاسخ در زمینه باشند. اگر اطلاعات کافی برای پاسخ دادن به سوالی نداشتی، صادقانه بگو که نمی‌دانی."),
        ("human", "{question}\n\nزمینه:\n{context}\n\nمنابع:\n{source}\n\nپاسخ:")
    ])
    return prompt | chain | StructuredOutputParser()

def retrieve(state: RAGState) -> RAGState:
    vectorstore = get_vectorstore()
    question = state.get("question", "")
    top_k = state.get("top_k", 3)
    docs = vectorstore.similarity_search(question, k=top_k)
    context = []
    source = []
    for doc in docs:
        context.append(
            {
                "title": doc.metadata.get("title", ""),
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content
            }
        )
        source.append(
            {
                "title": doc.metadata.get("title", ""),
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content[:150] + "..."
            }
        )
    return {"context": context, "source": source}

def format_context(context: List[dict]) -> str:
    formatted_context = []
    for i, item in enumerate(context):
        formatted_context.append(f"عنوان: {item.get('title', '')}\nمنبع: {item.get('source', '')}\nمتن: {item.get('text', '')}")
    return "\n\n".join(formatted_context)

def generate_answer(state: RAGState) -> RAGState:
    chain = get_chain()
    question = state.get("question", "")
    context = state.get("context", [])
    if not context:
        return {"answer": "متاسفم، زمینه‌ای برای پاسخ دادن وجود ندارد."}
    source = state.get("source", [])
    formatted_context = format_context(context)
    # formatted_source = format_context(source)
    answer = chain.run(question=question, context=formatted_context)
    return {"answer": answer}

def create_rag_graph() -> StateGraph:
    # workflow = StateGraph()
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieved", retrieve)
    workflow.add_node("generate", generate_answer)

    workflow.add_edge(START, "retrieved")
    workflow.add_edge("retrieved", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
