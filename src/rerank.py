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
rerank.py — Multi-objective reranker with SBERT caching
-------------------------------------------------------
Combines:
  - semantic similarity (SBERT cosine)
  - politeness probability (classifier_infer.score)
  - simple fluency/quality heuristics (length & repetition penalties)
  - optional strategy bonus (supplied externally by pipeline)

Exports:
  - rank_by_politeness_and_similarity(orig, candidates)
  - rank_candidates(orig, candidates, w_polite, w_sim, w_quality)

Returns (best_text, scored_list) where:
  scored_list = [(text, polite_prob, sim, score), ...]  sorted desc by score
"""

# -------------------------------------------------------
# SETUP
# -------------------------------------------------------
import sys, os, re, math
from typing import List, Tuple, Dict

# Ensure src import works from any cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer, util
from src.classifier_infer import score as cls_score
from src.config import SBERT_MODEL, W_POLITE, W_SIM

# ----------------------------------------------
# SBERT SINGLETON MODEL + EMBEDDING CACHE
# ----------------------------------------------
_SBERТ = None
_EMB_CACHE: Dict[str, 'torch.Tensor'] = {}

def _get_sbert():
    global _SBERТ
    if _SBERТ is None:
        _SBERТ = SentenceTransformer(SBERT_MODEL)
    return _SBERТ

def _embed(text: str):
    """Embed a single string with cache."""
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    m = _get_sbert()
    emb = m.encode([text], convert_to_tensor=True)
    _EMB_CACHE[text] = emb
    return emb

def _embed_many(texts: List[str]):
    out = []
    missing = []
    for t in texts:
        if t in _EMB_CACHE:
            out.append(_EMB_CACHE[t])
        else:
            out.append(None)
            missing.append(t)
    if missing:
        m = _get_sbert()
        embs = m.encode(missing, convert_to_tensor=True)
        j = 0
        for i, t in enumerate(texts):
            if out[i] is None:
                _EMB_CACHE[t] = embs[j:j+1]
                out[i] = _EMB_CACHE[t]
                j += 1
    return out

# ----------------------------------------------
# SIMPLE QUALITY HEURESTICS
# ----------------------------------------------
_vowel_re = re.compile(r"[aeiou]", re.I)
_multi_space = re.compile(r"\s{2,}")

def _quality_score(text: str) -> float:
    """
    Heuristic quality between ~0.0 and ~1.0
      + length sweet-spot
      + has vowels
      - repeated punctuation / characters
      - too short or too long
    """
    t = text.strip()
    n = len(t)

    # Length target around 40–140 chars for a sentence
    if n < 15:
        len_term = 0.2
    elif n > 220:
        len_term = 0.3
    else:
        # bell-ish curve around 80
        center = 80.0
        len_term = max(0.0, 1.0 - abs(n - center) / 140.0)

    # Vowel presence (avoid all-caps acronyms etc.)
    vowels = 1.0 if _vowel_re.search(t) else 0.2

    # Repeated punctuation penalty
    rep_punct = 1.0
    if "!!" in t or "??" in t or "..." in t:
        rep_punct -= 0.2
    if _multi_space.search(t):
        rep_punct -= 0.1
    # Character repetition like cooooool
    if re.search(r"(.)\1{3,}", t):
        rep_punct -= 0.2

    raw = max(0.0, min(1.0, 0.5 * len_term + 0.3 * vowels + 0.2 * rep_punct))
    return raw

# ----------------------------------------------
# SCORING AND RANKING
# ----------------------------------------------
def _score_candidate(orig: str, cand: str, w_polite: float, w_sim: float, w_quality: float) -> Tuple[float, float, float, float]:
    """
    Returns (polite_prob, sim, quality, final_score)
    final_score = w_polite * polite + w_sim * sim + w_quality * quality
    """
    polite_prob = cls_score(cand)["polite"]
    orig_emb = _embed(orig)
    cand_emb = _embed(cand)
    sim = util.cos_sim(orig_emb, cand_emb)[0, 0].item()
    quality = _quality_score(cand)
    final = w_polite * polite_prob + w_sim * sim + w_quality * quality
    return polite_prob, sim, quality, final

def rank_candidates(orig: str,
                    candidates: List[str],
                    w_polite: float = W_POLITE,
                    w_sim: float = W_SIM,
                    w_quality: float = 0.15) -> Tuple[str, List[Tuple[str, float, float, float]]]:
    """
    Rank candidate rewrites using a multi-objective score.
    Returns best_text and a sorted list of (text, polite, sim, score).
    """
    if not candidates:
        return orig, [(orig, cls_score(orig)["polite"], 1.0, w_polite * 1.0 + w_sim * 1.0)]

    # Pre-embed to reduce repeated calls
    _embed(orig)
    _embed_many(candidates)

    scored = []
    for c in candidates:
        polite, sim, quality, sc = _score_candidate(orig, c, w_polite, w_sim, w_quality)
        scored.append((c, polite, sim, sc))

    scored.sort(key=lambda x: x[3], reverse=True)
    best = scored[0][0]
    return best, scored

def rank_by_politeness_and_similarity(orig: str, candidates: List[str]) -> Tuple[str, List[Tuple[str, float, float, float]]]:
    """
    Backward-compatible wrapper (used by your pipeline before).
    Uses default weights from config and a small quality term.
    """
    return rank_candidates(orig, candidates, w_polite=W_POLITE, w_sim=W_SIM, w_quality=0.15)

# ----------------------------------------------
# DEMO AND TESTING
# ----------------------------------------------
if __name__ == "__main__":
    origin = "send me the report asap"
    cands = [
        "Could you please send me the report when you have a moment? Thank you.",
        "Send the report now.",
        "Please send the report.",
        "When you get a chance, could you share the report?"
    ]
    best, scored = rank_candidates(origin, cands)
    print("Best:", best)
    for row in scored:
        print(f"- {row[0]} | polite={row[1]:.3f} sim={row[2]:.3f} score={row[3]:.3f}")

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------