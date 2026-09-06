import re
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hazm import Normalizer

IN_FILE = Path("Amoozeshi/rag/data/pages.jsonl")
OUT_DIR = Path("Amoozeshi/rag/dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chunks.jsonl"

normalizer = Normalizer()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", ":", "!", "?"]
)

def clean_text(text:str) -> str:
    text = normalizer.normalize(text)
    text = text.replace("\u200f", "").replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def main():
    chunk_count = 0
    with IN_FILE.open("r", encoding="utf-8") as infile, OUT_FILE.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            title = data.get("title")
            source = data.get("source")
            pageid = data.get("pageid")
            text = data.get("text")
            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue
            chunks = text_splitter.split_text(cleaned_text)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_record = {
                    "id": f"fa-wiki-{pageid}-{i:04d}",
                    "doc_id": f"fa-wiki-{pageid}",
                    "title": title,
                    "source": source,
                    "chunk_index": i,
                    "text": chunk
                }
                outfile.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                chunk_count += 1
            print(f"{title} -> {chunk_count} chunks")
    print(f"Processed {title} with {chunk_count} chunks.")

if __name__ == "__main__":
    main()