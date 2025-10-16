### -------------------------------------------------------------------------------------------------
## Team: 사푸트라 (Saputra Rizky Johan), 바트오르식 (Butemj Bat-Orshikh), 쉬슈잔 (Shu Xian Chow)
## Institution: Seoul National University, South Korea
## Course: 2025-2 Introduction to Natural Language Processing (001)
## Instructors: 황승원 (Prof), 김종윤(TA), 한상은 (TA)
## Project: Classifier-Guided Politeness Rewriting via Span Detection and Controlled Text Generation
## Corpus: Stanford Politeness Corpus (Convokit)
## Note: If additional implementations are to be made, please document them at the README file
### -------------------------------------------------------------------------------------------------

# -------------------------------------------------------
# SPECIFICATIONS
# -------------------------------------------------------
"""
classifier_infer.py — Enhanced politeness classifier inference
--------------------------------------------------------------
Loads the fine-tuned classifier (out/classifier/model) and computes politeness
probabilities, batch outputs, and strategy detection.

Usage:
    from src.classifier_infer import score, batch_score
    print(score("Could you please send the file?"))
"""

# -------------------------------------------------------
# SETUP
# -------------------------------------------------------
import sys, os, re, json, math
from typing import List, Dict, Union
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure src import works from any cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MAX_LEN
# Optional: use the same lexicons from baseline_rules
try:
    from src.baseline_rules import LEXICONS
except ImportError:
    LEXICONS = {}

# -------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------
MODEL_DIR = "out/classifier/model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# LOAD MODEL + TOKENIZER
# -------------------------------------------------------
try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.to(DEVICE)
    mdl.eval()
except Exception as e:
    print(f"⚠️ Failed to load model from {MODEL_DIR}: {e}")
    tok, mdl = None, None

# -------------------------------------------------------
# STRATEGY DETECTION (LIGHTWEIGHT LEXICAL CHECK)
# -------------------------------------------------------
def detect_strategies(text: str) -> List[str]:
    found = []
    if not LEXICONS:
        return found
    t = " " + text.lower() + " "
    for name, words in LEXICONS.items():
        for w in words:
            if w in t:
                found.append(name)
                break
    return found

# -------------------------------------------------------
# CORE SCORING
# -------------------------------------------------------
def _predict(texts: List[str]) -> List[Dict[str, float]]:
    if not mdl or not tok:
        raise RuntimeError("Model or tokenizer not loaded.")
    enc = tok(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    results = [{"impolite": float(p[0]), "polite": float(p[1])} for p in probs]
    return results

def _confidence_label(polite: float, low=0.45, high=0.65) -> str:
    if polite < low:
        return "impolite"
    elif polite > high:
        return "polite"
    else:
        return "uncertain"

# -------------------------------------------------------
# SINGULAR TEXT SCORING
# -------------------------------------------------------
def score(text: str) -> Dict[str, Union[float, str]]:
    text = text.strip()
    if not text:
        return {"impolite": 0.0, "polite": 0.0, "label": "neutral"}
    result = _predict([text])[0]
    result["label"] = _confidence_label(result["polite"])
    result["strategies"] = detect_strategies(text)
    return result

# -------------------------------------------------------
# BATCH SCORING
# -------------------------------------------------------
def batch_score(texts: Union[List[str], pd.Series], batch_size: int = 32,
                export_path: str = None) -> pd.DataFrame:
    """
    Score multiple texts and optionally export to CSV/JSON.

    Args:
        texts: list or pandas.Series of sentences
        batch_size: inference batch size
        export_path: optional CSV/JSON output path

    Returns:
        pandas DataFrame with text, polite_prob, label, strategies
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        probs = _predict(batch)
        for t, p in zip(batch, probs):
            label = _confidence_label(p["polite"])
            strategies = detect_strategies(t)
            results.append({
                "text": t,
                "polite_prob": round(p["polite"], 4),
                "impolite_prob": round(p["impolite"], 4),
                "label": label,
                "strategies": ", ".join(strategies)
            })

    df = pd.DataFrame(results)
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        if export_path.endswith(".json"):
            df.to_json(export_path, orient="records", indent=2)
        else:
            df.to_csv(export_path, index=False)
        print(f"💾 Saved predictions to {export_path}")
    return df

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Politeness classifier inference tool")
    ap.add_argument("--text", type=str, help="Single sentence to score")
    ap.add_argument("--input_file", type=str, help="File (.txt, .csv) for batch scoring")
    ap.add_argument("--export_path", type=str, help="Save results to this file (CSV/JSON)")
    args = ap.parse_args()

    if args.text:
        print(json.dumps(score(args.text), indent=2))
    elif args.input_file:
        if args.input_file.endswith(".csv"):
            df = pd.read_csv(args.input_file)
            texts = df.iloc[:,0].tolist()
        else:
            with open(args.input_file, "r", encoding="utf-8") as f:
                texts = [ln.strip() for ln in f if ln.strip()]
        df = batch_score(texts, export_path=args.export_path)
        print(df.head())
    else:
        print("Please provide --text or --input_file")

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------