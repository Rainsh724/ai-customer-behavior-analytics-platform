from fastapi import FastAPI, HTTPException
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from raggraphadvance import create_advanced_graph

app = FastAPI(title="Advanced RAG API")

BASE_DIR = Path(__file__).resolve().parent
UI_FILE = BASE_DIR / "ui.html"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag_graph = create_advanced_graph()
    print("RAG graph initialized successfully.")
except Exception as e:
    rag_graph = None
    print(f"Failed to initialize RAG graph: {e}")


@app.get("/")
def serve_ui():
    with open(UI_FILE, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/ask")
def ask_endpoint(payload: dict):

    if rag_graph is None:
        raise HTTPException(
            status_code=503,
            detail="RAG graph is not initialized."
        )

    question = payload.get("question", "").strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question is required."
        )

    try:
        result = rag_graph.invoke({
            "question": question,
            "context": [],
            "answer": ""
        })

        sources = [
            {
                "title": item.get("title", "نامشخص"),
                "url": item.get("url", "#")
            }
            for item in result.get("context", [])
        ]

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": sources,
            "search_type": "BM25"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )