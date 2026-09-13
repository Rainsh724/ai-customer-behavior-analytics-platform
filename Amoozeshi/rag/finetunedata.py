import json
import random
from pathlib import Path

CHUNKS_FILE = Path("advance/chunks.jsonl")
OUT_DIR = Path("data/finetune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_chunks():
    chunks = []
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("فایل چانک‌ها پیدا نشد.")
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def build_classification_data(chunks):
    data = []
    for chunk in chunks:
        title = chunk.get("title", "")
        text = chunk.get("text", "").strip()
        if not text or len(text) < 50:
            continue
        if "هوش مصنوعی" in title:
            label = "هوش مصنوعی"
        elif "پردازش زبان" in title:
            label = "پردازش زبان طبیعی"
        elif "یادگیری ماشین" in title:
            label = "یادگیری ماشین"
        else:
            label = "عمومی"
        data.append({
            "instruction": "این متن را به یکی از دسته‌های زیر طبقه‌بندی کن: هوش مصنوعی، پردازش زبان طبیعی، یادگیری ماشین",
            "input": text[:500],
            "output": label,
            "source": chunk.get("source", ""),
        })
    return data

def build_qa_data(chunks):
    data = []
    for chunk in chunks:
        title = chunk.get("title", "")
        text = chunk.get("text", "").strip()
        if not text or len(text) < 100:
            continue
        question = f"درباره {title} چه می‌دانید؟"
        answer = text[:200]
        data.append({
            "instruction": "به سؤال زیر بر اساس متن داده شده پاسخ بده.",
            "input": f"متن: {text[:400]}\n\nسؤال: {question}",
            "output": answer,
            "source": chunk.get("source", ""),
        })
    return data

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": data[:train_end],
        "validation": data[train_end:val_end],
        "test": data[val_end:],
    }

def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    chunks = load_chunks()
    classification_data = build_classification_data(chunks)
    qa_data = build_qa_data(chunks)

    cls_split = split_data(classification_data)
    save_jsonl(cls_split["train"], OUT_DIR / "classification_train.jsonl")
    save_jsonl(cls_split["validation"], OUT_DIR / "classification_val.jsonl")
    save_jsonl(cls_split["test"], OUT_DIR / "classification_test.jsonl")

    qa_split = split_data(qa_data)
    save_jsonl(qa_split["train"], OUT_DIR / "qa_train.jsonl")
    save_jsonl(qa_split["validation"], OUT_DIR / "qa_val.jsonl")
    save_jsonl(qa_split["test"], OUT_DIR / "qa_test.jsonl")

if __name__ == "__main__":
    main()