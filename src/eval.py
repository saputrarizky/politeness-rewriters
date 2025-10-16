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
eval.py — Evaluation script for the Politeness Rewriter
-------------------------------------------------------
Evaluates the pipeline on a subset of test samples (default: data/test.jsonl).

Metrics:
  • Counts proportion of samples whose politeness score improved
  • Optionally saves detailed per-sample CSV for inspection

Usage:
    python eval.py --n 200 --save_csv results/eval_results.csv
"""

# --------------------------------------------------------------
# SETUP
# --------------------------------------------------------------
import os, sys, json, csv, random, argparse
from statistics import mean

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import rewrite

# --------------------------------------------------------------
# DATA LOADER
# --------------------------------------------------------------
def load_jsonl(path: str):
    """Load JSONL dataset file line-by-line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# --------------------------------------------------------------
# EVALUATE PIPELINE
# --------------------------------------------------------------
def evaluate(n: int = 200,
             test_path: str = "data/test.jsonl",
             save_csv: str = ""):
    """
    Evaluate the pipeline on N random samples from test set.

    Args:
        n (int): number of examples to test
        test_path (str): path to test dataset
        save_csv (str): optional path to save detailed results

    Returns:
        dict with aggregated metrics
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Load test samples
    data = list(load_jsonl(test_path))
    random.shuffle(data)
    subset = data[:n]

    results = []
    improved_count = 0

    # Evaluate
    print(f"Evaluating on {len(subset)} samples...")
    for d in subset:
        x = d["text"]
        try:
            r = rewrite(x)
        except Exception as e:
            print(f"Skipped due to error: {e}")
            continue

        before = r.get("polite_prob_before", 0.0)
        after  = r.get("polite_prob_after", 0.0)
        improved = after > before
        improved_count += int(improved)

        results.append({
            "text": x,
            "output": r.get("output", ""),
            "rule_based": r.get("rule_based", ""),
            "polite_prob_before": round(before, 3),
            "polite_prob_after": round(after, 3),
            "improved": improved
        })

    # Aggregate stats
    total = len(results)
    win_rate = improved_count / total if total > 0 else 0.0
    mean_before = mean(r["polite_prob_before"] for r in results)
    mean_after  = mean(r["polite_prob_after"] for r in results)

    print(f"Improved politeness on {improved_count}/{total} samples ({win_rate:.1%})")
    print(f"Avg polite_prob_before = {mean_before:.3f}, after = {mean_after:.3f}")
    print(f"avg improvement = {mean_after - mean_before:.3f}")

    # Optional save
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved detailed results → {save_csv}")

    return {
        "n": total,
        "improved": improved_count,
        "win_rate": win_rate,
        "avg_before": mean_before,
        "avg_after": mean_after,
        "avg_delta": mean_after - mean_before
    }

# --------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the Politeness Rewriter on test data")
    ap.add_argument("--n", type=int, default=200, help="Number of test samples to evaluate")
    ap.add_argument("--test_path", type=str, default="data/test.jsonl", help="Path to test dataset")
    ap.add_argument("--save_csv", type=str, default="", help="Optional CSV path to save detailed results")
    args = ap.parse_args()

    stats = evaluate(n=args.n, test_path=args.test_path, save_csv=args.save_csv)
    print(json.dumps(stats, indent=2))

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------