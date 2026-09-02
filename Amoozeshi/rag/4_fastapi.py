from fastapi import FastAPI, HTTPException
from raggraph import create_rag_graph, retrieve, OLLAMA_MODEL

app = FastAPI(title="RAG API", description="A simple RAG API using FastAPI and LangChain", version="1.0.0")

try :
    rag_graph = create_rag_graph()
    print(f"RAG graph created successfully with model: {OLLAMA_MODEL}")
except Exception as e:
    rag_graph = None
    print(f"Error creating RAG graph: {e}")

@app.get("/retrieve")
def root():
    return {"message": "Welcome to the RAG API. Use the /retrieve endpoint to get answers.",
            "docs": "/docs",
            "model": OLLAMA_MODEL
            }

@app.post("/retrieve")
def retrieve_endpoint(playload: dict):
    question = playload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    top_k = playload.get("top_k", 3)
    result = retrieve({"question": question, "top_k": top_k})
    return {"question": question, "sources": result.get("source", []), "result": result}