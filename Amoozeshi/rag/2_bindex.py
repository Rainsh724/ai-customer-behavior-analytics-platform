import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_DIR = Path("Amoozeshi/rag/dataset/chunks.jsonl")
FAISS_DIR = Path("Amoozeshi/rag/dataset/faiss_index")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    # Load documents from the JSONL file
    documents = []
    with CHUNK_DIR.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            item = json.loads(line)
            documents.append(Document(
                page_content=item.get("text"), metadata={
                    "id": item.get("id"),
                    "doc_id": item.get("doc_id"),
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "chunk_index": item.get("chunk_index")
                }
            ))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"FAISS index saved to {FAISS_DIR}")

if __name__ == "__main__":
    main()