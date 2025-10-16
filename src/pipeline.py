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
pipeline.py — Core Politeness Rewriter Orchestrator
---------------------------------------------------
Blends multiple components into one unified rewrite system:
  • Pre-check politeness with classifier_infer.score()
  • Optional rule-based rewrite via baseline_rules.rewrite_rule_based()
  • Neural generation using rewrite_t5.paraphrase()
  • Multi-objective rerank via rerank.rank_by_politeness_and_similarity()
  • Strategy-aware bonus and final scoring

Returns rich output for app.py visualization, including:
  output, politeness scores (before/after), sim, scores, bonus,
  detected strategies, negative markers, and top candidate diagnostics.
"""

# --------------------------------------------------------------
# SETUP
# --------------------------------------------------------------
import sys, os, re, math, time, json
from typing import List, Dict, Tuple

# Ensure src import works from any cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_rules import rewrite_rule_based
from src.classifier_infer import score
from src.rewrite_t5 import paraphrase
from src.rerank import rank_by_politeness_and_similarity
from src.config import POLITE_THRESHOLD, W_POLITE, W_SIM

# --------------------------------------------------------------
# STRATEGY LEXICONS
# --------------------------------------------------------------
# Add more strategy lexicons
STRATEGY_LEXICONS: Dict[str, List[str]] = {
    "gratitude": ["thank you", "thanks", "much appreciated", "really appreciate", "appreciated"],
    "apology": ["sorry", "i apologize", "my apologies", "excuse me", "pardon me"],
    "request_softener": ["please", "could you", "would you", "would you mind", "can you", "may i"],
    "hedges": ["maybe", "perhaps", "a bit", "somewhat", "slightly", "i think", "i feel", "it seems", "kind of", "sort of"],
    "deference": ["sir", "madam", "dear", "kindly"],
    "greetings": ["hello", "hi ", "good morning", "good afternoon", "good evening"],
    "indirect": ["would it be possible", "would it be okay", "if you have a moment", "when you get a chance"]
}

# Add more negative markers
NEGATIVE_MARKERS = {
    "profanity": re.compile(r"\b(damn|hell|stupid|idiot|fuck|shit|wtf|bastard|moron)\b", re.I),
    "demandy":  re.compile(r"\b(asap|immediately|right away|now|at once)\b", re.I),
    "blame":    re.compile(r"\byou (forgot|didn't|failed to)\b", re.I),
}

# --------------------------------------------------------------
# STRATEGY / NEGATIVITY DETECTION UTILITIES
# --------------------------------------------------------------
def detect_strategies(text: str) -> List[str]:
    """Return list of strategy names found in text."""
    t = " " + text.lower() + " "
    found = []
    for name, words in STRATEGY_LEXICONS.items():
        for w in words:
            if w in t:
                found.append(name)
                break
    return found

def count_negatives(text: str) -> int:
    """Count how many negative politeness markers are found."""
    return sum(1 for _, rx in NEGATIVE_MARKERS.items() if rx.search(text))

def explain_candidates_with_strategies(cands: List[str]) -> List[Dict]:
    """Annotate candidates with strategy coverage and negative marker count."""
    annotated = []
    for c in cands:
        strategies = detect_strategies(c)
        negatives = count_negatives(c)
        annotated.append({"text": c, "strategies": strategies, "negatives": negatives})
    return annotated

def strategy_bonus(strategies: List[str], negatives: int) -> float:
    """
    Compute a small additive bonus informed by strategy coverage.
    + Gratitude, apologies, hedges, requests → small positive weight
    – Negative markers → penalty
    """
    POS_W = {
        "gratitude": 0.02, "apology": 0.02, "request_softener": 0.03,
        "hedges": 0.01, "deference": 0.01, "greetings": 0.01, "indirect": 0.02
    }
    base = sum(POS_W.get(s, 0.0) for s in strategies) - 0.03 * negatives
    return max(-0.1, min(0.1, base))  # Clamp small bonus to ±0.1

# --------------------------------------------------------------
# INTERNAL PIPELINE HELPERS
# --------------------------------------------------------------
def _skip_if_already_polite(text: str) -> Tuple[bool, float]:
    """Return (skip, polite_prob) if text already above threshold."""
    before = score(text)["polite"]
    return (before >= POLITE_THRESHOLD, before)

def _apply_rules(text: str, enabled: bool) -> str:
    """Run rule-based pre-pass if enabled."""
    return rewrite_rule_based(text)["output"] if enabled else text

def _generate_candidates(text: str, n: int) -> List[str]:
    """Generate neural paraphrase candidates."""
    n = max(1, int(n or 4))
    return paraphrase(text, num_return_sequences=n)

def _rerank(orig: str, cands: List[str]) -> Tuple[str, List[Tuple[str, float, float, float]]]:
    """
    Use SBERT + classifier reranker.
    Returns (best_text, scored_list) with (text, polite, sim, score).
    """
    best_text, scored = rank_by_politeness_and_similarity(orig, cands)
    scored.sort(key=lambda x: x[3], reverse=True)
    return best_text or cands[0], scored

def _apply_strategy_bonus(scored: List[Tuple[str, float, float, float]]) -> List[Tuple[str, float, float, float, float]]:
    """
    Add bonus informed by strategy detection for tie-breaking.
    Returns list of tuples: (text, polite, sim, score, score_plus_bonus).
    """
    ann = {a["text"]: a for a in explain_candidates_with_strategies([s[0] for s in scored])}
    out = []
    for text, polite, sim, score in scored:
        a = ann.get(text, {"strategies": [], "negatives": 0})
        bonus = strategy_bonus(a["strategies"], a["negatives"])
        out.append((text, polite, sim, score, score + bonus))
    out.sort(key=lambda x: x[4], reverse=True)
    return out

# --------------------------------------------------------------
# OUTPUT ADJUSTMENTS AND REMOVAL OF UNNECESSARY PROMPTS
# --------------------------------------------------------------
def _strip_instruction_prefix(t: str) -> str:
    s = (t or "").strip()
    low = s.lower()
    # Triggers if it starts with an instruction
    prefixes = [
        "please write the following sentence",
        "please read the following sentence",
        "rewrite the following sentence",
        "input,"
    ]
    for p in prefixes:
        if low.startswith(p):
            # Remove the first sentence 
            parts = s.split('.', 1)
            if len(parts) > 1:
                s = parts[1].strip()
            break
    return s

# --------------------------------------------------------------
# MAIN ENTY POINT
# --------------------------------------------------------------
def rewrite(text: str,
            target_tone: str = "polite",
            rule_based: bool = True,
            n_candidates: int = 4,
            return_debug: bool = True) -> Dict:
    """
    Main rewrite pipeline called by the Gradio app.

    Steps:
        Classifier pre-check (skip if already polite)
        Optional rule-based cleanup
        T5 neural paraphrase generation
        SBERT + classifier rerank
        Strategy-aware bonus adjustment
    """
    original = (text or "").strip()
    if not original:
        return {"output": "", "polite_prob_before": 0.0, "polite_prob_after": 0.0, "rule_based": rule_based}

    # Pre-check
    skip, before = _skip_if_already_polite(original)
    if skip:
        return {
            "output": original,
            "polite_prob_before": before,
            "polite_prob_after": before,
            "sim": 1.0,
            "score": W_POLITE * before + W_SIM * 1.0,
            "score_plus_bonus": W_POLITE * before + W_SIM * 1.0,
            "rule_based": False,
            "strategies": detect_strategies(original),
            "negatives": count_negatives(original),
            "timestamp": time.time()
        }

    # Rule-based rewrite
    text2 = _apply_rules(original, enabled=rule_based)

    # Generate candidates
    cands = _generate_candidates(text2, n_candidates)

    # Rerank
    best_text, scored = _rerank(original, cands)

    # Apply strategy-aware bonus
    scored_bonus = _apply_strategy_bonus(scored)
    f_text, f_pol, f_sim, f_score, f_plus = scored_bonus[0]

    # Build final result
    result = {
        "output": f_text,
        "polite_prob_before": before,
        "polite_prob_after": f_pol,
        "sim": f_sim,
        "score": f_score,
        "score_plus_bonus": f_plus,
        "rule_based": rule_based,
        "strategies": detect_strategies(f_text),
        "negatives": count_negatives(f_text),
        "timestamp": time.time()
    }

    # Optional debug info
    if return_debug:
        result["top_candidates"] = [
            {"rank": i + 1, "text": t, "polite": round(pp, 3), "sim": round(ss, 3), "score": round(sc, 3)}
            for i, (t, pp, ss, sc) in enumerate(scored[:3])
        ]

    # Keep raw for audit, return cleaned for UI/eval
    raw_out = result.get("output", "")
    result["output_raw"] = raw_out
    result["output"] = _strip_instruction_prefix(raw_out)

    return result


# --------------------------------------------------------------
# SMOKE TEST
# --------------------------------------------------------------
if __name__ == "__main__":
    sample = "send me the data asap"
    out = rewrite(sample, target_tone="polite", rule_based=True, n_candidates=4, return_debug=True)
    print(json.dumps(out, indent=2))

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------