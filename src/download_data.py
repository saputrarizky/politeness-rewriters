### -------------------------------------------------------------------------------------------------
## Team: 사푸트라 (Saputra Rizky Johan), 바트오르식 (Butemj Bat-Orshikh), 쉬슈잔 (Shu Xian Chow)
## Institution: Seoul National University, South Korea
## Course: 2025-2 Introduction to Natural Language Processing (001)
## Instructors: 황승원 (Prof), 김종윤(TA), 한상은 (TA)
## Project: Classifier-Guided Politeness Rewriting via Span Detection and Controlled Text Generation
## Corpus: Stanford Politeness Corpus (Convokit)
## Note: If additional implementations are to be made, please document them at the README file
### -------------------------------------------------------------------------------------------------

# --------------------------------------------------------------
# SPECIFICATIONS
# --------------------------------------------------------------
"""
download_data.py — Stanford Politeness Corpus downloader and formatter
----------------------------------------------------------------------
Fetches the official Stanford Politeness dataset via ConvoKit,
extracts (text, label, score) triplets, and writes JSONL splits
for training the classifier.

Output structure (default ./data):
    data/
      ├── stanford_politeness.jsonl   # full dataset
      ├── train.jsonl
      ├── val.jsonl
      └── test.jsonl
"""

# --------------------------------------------------------------
# SETUP
# --------------------------------------------------------------
import os, json, random
from convokit import Corpus, download

# --------------------------------------------------------------
# CONFIGURATIONS
# --------------------------------------------------------------
OUT_DIR = "data"           # directory for all dataset files
SPLIT = (0.8, 0.1, 0.1)    # (train, val, test) proportions

# --------------------------------------------------------------
# DATASET LOADER
# --------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("📥 Downloading Stanford Politeness Corpus via ConvoKit...")

    # 1️⃣ Download + load corpus
    corpus = Corpus(download("stanford-politeness-corpus"))

    # 2️⃣ Extract usable utterances
    rows = []
    for utt in corpus.utterances.values():
        text = (utt.text or "").strip()
        meta = utt.meta or {}
        if not text:
            continue
        if "Binary" not in meta:
            continue
        try:
            label = int(meta["Binary"])         # 1 = polite, 0 = impolite
        except Exception:
            continue
        score = float(meta.get("Continuous", 0.0))  # optional continuous score
        rows.append({"text": text, "label": label, "score": score})

    random.shuffle(rows)
    print(f"✅ Collected {len(rows)} valid utterances.")

    # 3️⃣ Save full dataset
    all_path = os.path.join(OUT_DIR, "stanford_politeness.jsonl")
    with open(all_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 Wrote complete dataset: {len(rows)} → {all_path}")

    # 4️⃣ Train / val / test split
    n = len(rows)
    n_tr = int(SPLIT[0] * n)
    n_val = int(SPLIT[1] * n)
    train, val, test = rows[:n_tr], rows[n_tr:n_tr + n_val], rows[n_tr + n_val:]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        p = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"📄 {name}: {len(part)} → {p}")

    print("🏁 Done! Dataset ready for classifier_train.py")

# --------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    main()

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------