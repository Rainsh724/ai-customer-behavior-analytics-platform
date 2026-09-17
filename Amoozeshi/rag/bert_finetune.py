import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import evaluate

MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
DATA_DIR = Path("data/finetune")
OUTPUT_DIR = Path("my_finetuned_model")

LABELS = ["هوش مصنوعی", "پردازش زبان طبیعی", "یادگیری ماشین", "عمومی"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def prepare_dataset(data):
    texts = []
    labels = []
    for item in data:
        text = item.get("input", "")
        label = item.get("output", "")
        if text and label in LABEL2ID:
            texts.append(text)
            labels.append(LABEL2ID[label])
    return Dataset.from_dict({"text": texts, "label": labels})

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=256,
    )

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_data = load_jsonl(DATA_DIR / "classification_train.jsonl")
    val_data = load_jsonl(DATA_DIR / "classification_val.jsonl")

    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)

    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        fp16=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

if __name__ == "__main__":
    main()