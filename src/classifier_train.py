### -------------------------------------------------------------------------------------------------
## Team: 사푸트라 (Saputra Rizky Johan), 바트오르식 (Butemj Bat-Orshikh), 쉬슈잔 (Shu Xian Chow)
## Institution: Seoul National University, South Korea
## Course: 2025-2 Introduction to Natural Language Processing (001)
## Instructors: 황승원 (Prof), 김종윤(TA), 한상은 (TA)
## Project: Classifier-Guided Politeness Rewriting via Span Detection and Controlled Text Generation
## Corpus: Stanford Politeness Corpus (Convokit)
## Note: If additional implementations are to be made, please document them at the README file
### -------------------------------------------------------------------------------------------------

# ------------------------------
# SPECIFICATIONS
# ------------------------------
"""
classifier_train.py — Train a politeness classifier on Stanford JSON
--------------------------------------------------------------------
- Robust loader for Stanford politeness JSON (and JSONL / custom schemas)
- Tokenization with Hugging Face transformers
- Train/val split with stratification
- Metrics: accuracy, macro F1
- Saves model and tokenizer to out/classifier/model
"""

# ------------------------------
# SETUP
# ------------------------------
import os, sys, json, random, math, argparse
from typing import List, Dict, Any, Tuple

# Ensure src import works even if run from different cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from src.config import MODEL_CLS, MAX_LEN

# ------------------------------
# REPRODUCIBILITY
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ------------------------------
# DATA LOADERS
# ------------------------------
COMMON_TEXT_KEYS = ["request", "Request", "text", "utterance", "sentence"]
COMMON_LABEL_KEYS = ["label", "Binary", "binary", "polite", "is_polite", "y"]
COMMON_SCORE_KEYS = ["score", "politeness", "politeness_score"]

def _infer_text(item: Dict[str, Any]) -> str:
    for k in COMMON_TEXT_KEYS:
        if k in item and isinstance(item[k], str):
            return item[k]
    # fallback: join strings
    for v in item.values():
        if isinstance(v, str) and len(v.split()) > 2:
            return v
    return ""

def _infer_label(item: Dict[str, Any], score_threshold: float = 0.0) -> int:
    # 1) explicit binary label
    for k in COMMON_LABEL_KEYS:
        if k in item:
            v = item[k]
            if isinstance(v, (bool, int)):
                return int(v)
            if isinstance(v, str):
                if v.lower() in ["polite", "true", "yes", "1"]:
                    return 1
                if v.lower() in ["impolite", "false", "no", "0"]:
                    return 0
                try:
                    return 1 if float(v) > 0 else 0
                except:
                    pass
    # 2) score threshold
    for k in COMMON_SCORE_KEYS:
        if k in item:
            try:
                return 1 if float(item[k]) > score_threshold else 0
            except:
                continue
    # default: neutral → impolite (0)
    return 0

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def normalize_examples(raw: Any, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Convert various Stanford politeness formats into a unified list
    of dicts with {'text': str, 'label': 0/1}.
    """
    rows = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        # Some dumps use {"data": [...]} or {"examples": [...]}
        if "data" in raw and isinstance(raw["data"], list):
            items = raw["data"]
        elif "examples" in raw and isinstance(raw["examples"], list):
            items = raw["examples"]
        else:
            # assume dict-of-dicts
            items = list(raw.values())
    else:
        items = []

    for it in items:
        text = _infer_text(it)
        if not text:
            continue
        label = _infer_label(it, score_threshold=score_threshold)
        label = 0 if int(label) < 0 else int(label)
        rows.append({"text": text, "label": int(label)})
    return rows

# ------------------------------
# DATASET PIPELINE
# ------------------------------
def build_dataset(data_path: str, score_threshold: float = 0.0) -> Dataset:
    if data_path.endswith(".jsonl"):
        raw = load_jsonl(data_path)
    elif data_path.endswith(".json"):
        raw = load_json(data_path)
    else:
        raise ValueError("Unsupported data file format. Use .json or .jsonl")

    examples = normalize_examples(raw, score_threshold=score_threshold)
    if not examples:
        raise ValueError("No valid examples parsed from dataset.")
    return Dataset.from_list(examples)

def stratified_split(ds: Dataset, val_ratio: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset]:
    # simple stratified split by label
    y = ds["label"]
    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]
    random.Random(seed).shuffle(idx0)
    random.Random(seed).shuffle(idx1)
    n0 = max(1, int(len(idx0) * val_ratio))
    n1 = max(1, int(len(idx1) * val_ratio))
    val_idx = set(idx0[:n0] + idx1[:n1])
    train_idx = [i for i in range(len(ds)) if i not in val_idx]
    return ds.select(train_idx), ds.select(list(val_idx))

# ------------------------------
# TOKENIZATION AND METRICS
# ------------------------------
def tokenize_batch(tokenizer, batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# ------------------------------
# TRAINING PIPELINE
# ------------------------------
def train_classifier(data_path: str,
                     save_dir: str = "out/classifier/model",
                     val_ratio: float = 0.1,
                     epochs: int = 3,
                     batch_size: int = 16,
                     lr: float = 5e-5,
                     score_threshold: float = 0.0,
                     seed: int = 42):
    set_seed(seed)

    print(f"Loading dataset: {data_path}")
    ds = build_dataset(data_path, score_threshold=score_threshold)
    print(f"Total examples: {len(ds)} (labels: {dict(zip(*np.unique(ds['label'], return_counts=True)))} )")

    train_ds, val_ds = stratified_split(ds, val_ratio=val_ratio, seed=seed)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CLS)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CLS, num_labels=2)

    train_ds = train_ds.map(lambda b: tokenize_batch(tokenizer, b), batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(lambda b: tokenize_batch(tokenizer, b), batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="out/classifier/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        eval_steps=200,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()
    print("Evaluating...")
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving best model to: {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("🏁 Done.")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True,
                    help="Path to Stanford politeness JSON/JSONL")
    ap.add_argument("--save_dir", type=str, default="out/classifier/model")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--score_threshold", type=float, default=0.0,
                    help="If dataset has 'score', threshold > this => polite=1")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    train_classifier(
        data_path=args.data_path,
        save_dir=args.save_dir,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        score_threshold=args.score_threshold,
        seed=args.seed
    )

# ------------------------------
# DEMO AND TESTING
# ------------------------------
if __name__ == "__main__":
    main()

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------