import json
from pathlib import Path
from hazm import Normalizer, sent_tokenize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# الان اینجا داریم پیشرفته تر چانگ این رو انجام میدیم

IN_FILE=Path("Amoozeshi/rag/data/pages.jsonl")
OUT_DIR=Path("advance")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_FILES=OUT_DIR / "advanscechunks.jsonl"
FAISS_DIR=OUT_DIR
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

normalizer=Normalizer()

def sentence_chunker(text,chunk_size=600,overlap_sentence=1):

    sentences=sent_tokenize(text)
    chunks=[]
    current_chunk=[]
    current_len=0
    for sent in sentences :
        if current_len + len(sent) > chunk_size  and current_chunk :
            chunks.append(" ".join(current_chunk))
            current_chunk=current_chunk[-overlap_sentence:]
            current_len = sum(len(s) for s in current_chunk)
        current_chunk.append(sent)
        current_len+=len(sent)

    if current_chunk :
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    docs=[]
    with IN_FILE.open("r",encoding="utf-8") as f_in, CHUNKS_FILES.open("w",encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            page=json.loads(line)
            clean_text= normalizer.normalize(page["text"])
            chunks=sentence_chunker(clean_text)

            for i,chunk in enumerate(chunks):
                metadata={
                    "chunk_id":f"{page['pageid']}-{i}",
                    "title":page["title"],
                    "source":page["source"],
                    "chunk_index":i
                }

                doc= Document(page_content=chunk , metadata=metadata)
                docs.append(doc)
                f_out.write(json.dumps({
                    "text": chunk,
                    "metadata": metadata,


                }, ensure_ascii=False) + "\n")
            print(f"{page['title']} -> {len(chunk)} chunks Done")

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(str(FAISS_DIR))
    print("FAISS DONE")
if __name__=="__main__":
    main()