### -------------------------------------------------------------------------------------------------
## Team: 사푸트라 (Saputra Rizky Johan), 바트오르식 (Butemj Bat-Orshikh), 쉬슈잔 (Shu Xian Chow)
## Institution: Seoul National University, South Korea
## Course: 2025-2 Introduction to Natural Language Processing (001)
## Instructors: 황승원 (Prof), 김종윤(TA), 한상은 (TA)
## Project: Classifier-Guided Politeness Rewriting via Span Detection and Controlled Text Generation
## Corpus: Stanford Politeness Corpus (Convokit)
## Note: If additional implementations are to be made, please document them at the README file
### -------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# SPECIFICATIONS
# ---------------------------------------------------------------------
"""
rewrite_t5.py — Controlled polite paraphrasing with T5/BART-style models
------------------------------------------------------------------------
- Compatible with existing pipeline signature:
      paraphrase(text, num_return_sequences=4, num_beams=4, max_new_tokens=80)
- Adds:
    * Politeness conditioning via prompt templates
    * Tunable decoding: beam search OR nucleus sampling
    * Repetition / n-gram penalties, min length, stopwords trimming
    * Candidate de-duplication (semantic & textual lite)
    * Gentle post-processing (casing, punctuation)
- Reads MODEL_PARA from src.config (e.g., "Ateeqq/Text-Rewriter-Paraphraser")

Usage:
    from src.rewrite_t5 import paraphrase
    outs = paraphrase("send me the report asap", num_return_sequences=5, num_beams=5, max_new_tokens=96)
"""

# ---------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------
import sys, os, math, re, random
from typing import List, Dict, Any, Tuple

# Ensure src import works from various cwd entries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import MODEL_PARA

try:
    from src.config import PARAPHRASE_PROMPT_STYLE
except Exception:
    PARAPHRASE_PROMPT_STYLE = "concise"

# ---------------------------------------------------------------------
# SET GLOBALS FOR LOADED TOKENIZERS AND MODELS
# ---------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tok = None
_mdl = None

def _load_model():
    global _tok, _mdl
    if _tok is None or _mdl is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_PARA)
        _mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PARA)
        _mdl.to(_DEVICE)
        _mdl.eval()
    return _tok, _mdl


# ---------------------------------------------------------------------
# PROMPTING AND TEMPLATES (POLITENESS CONDITIONING)
# ---------------------------------------------------------------------
_PROMPTS = {
    "concise:polite": "Rewrite this sentence politely: ",
    "concise:extra-polite": "Rewrite this sentence politely and formally: ",
    "verbose:polite": (
        "Rewrite the following sentence so it is polite, respectful, concise, "
        "and keeps the same meaning. Avoid adding new facts.\nInput: "
    ),
    "verbose:extra-polite": (
        "Rewrite the sentence to be very polite, respectful, and appreciative "
        "while preserving meaning. Keep it brief and natural.\nInput: "
    ),
}

def _normalize_tone(tone: str) -> str:
    t = (tone or "polite").strip().lower()
    return "extra-polite" if t in ("extra-polite", "very-polite", "formal") else "polite"

def _build_prompt(text: str, tone: str = "polite", style: str = None, strategy_hints: List[str] = None) -> str:
    """
    Backward compatible:
      - prefers `tone` ("polite" | "extra-polite")
      - accepts legacy `style` param if callers pass it (ignored for content,
        we select from PARAPHRASE_PROMPT_STYLE instead).
      - `strategy_hints` kept for compatibility; currently unused.
    """
    # select prompt family from config
    style_key = (PARAPHRASE_PROMPT_STYLE or "concise").strip().lower()
    tone_key = _normalize_tone(tone)
    key = f"{style_key}:{tone_key}"
    base = _PROMPTS.get(key, _PROMPTS["concise:polite"])
    return f"{base}{(text or '').strip()}"

# ---------------------------------------------------------------------
# DECODING CONFIGURATIONS
# ---------------------------------------------------------------------
def _generation_kwargs(
    num_return_sequences: int = 4,
    num_beams: int = 4,
    max_new_tokens: int = 64,
    min_new_tokens: int = 8,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> Dict[str, Any]:
    """
    Consolidate decoding parameters.
    - If num_beams > 1 and do_sample=False → deterministic beam search.
    - If do_sample=True → nucleus sampling; num_beams still valid for diverse beam sampling.
    - Use max_new_tokens/min_new_tokens to limit only the generated text length (avoids cutting off the prompt tokens).
    """
    kw = dict(
        num_beams=max(1, int(num_beams)),
        num_return_sequences=max(1, int(num_return_sequences)),
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        length_penalty=float(length_penalty),
        early_stopping=bool(early_stopping),
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(min_new_tokens),
        repetition_penalty=float(repetition_penalty),
        do_sample=bool(do_sample),
    )
    if do_sample:  # only add sampling controls if sampling is used
        kw.update(dict(temperature=float(temperature), top_p=float(top_p)))
    return kw


# ---------------------------------------------------------------------
# POST-PROCESSING HELPERS
# ---------------------------------------------------------------------
_RX_TRAIL_PUNCT = re.compile(r"[.?!…]+$")
_RX_MULTI_SPACE = re.compile(r"\s{2,}")

def _clean_spaces(s: str) -> str:
    return _RX_MULTI_SPACE.sub(" ", (s or "").strip())

def _capitalize_first(s: str) -> str:
    return s if not s else (s[0].upper() + s[1:] if s[0].islower() else s)

def _ensure_sentence_end(s: str) -> str:
    return s if not s or _RX_TRAIL_PUNCT.search(s) else (s + ".")

def _soft_postprocess(s: str) -> str:
    s = (s or "").strip()
    # Trim leading bullets/quotes that sometimes appear from models
    s = re.sub(r'^["“”]+|["“”]+$', "", s)
    s = re.sub(r"^[-–•\s]+", "", s)
    s = _clean_spaces(s)
    s = _capitalize_first(s)
    s = _ensure_sentence_end(s)
    return s

# ---------------------------------------------------------------------
# CANDIDATE DUPLICATION (TEXTUAL AND LITE SEMANTIC)
# ---------------------------------------------------------------------
def _normalize_for_dupe(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def deduplicate(cands: List[str]) -> List[str]:
    seen = set()
    uniq = []
    for c in cands:
        key = _normalize_for_dupe(c)
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


# ---------------------------------------------------------------------
# PARAPHRASING API (COMPATIBLE WITH PIPELINE)
# ---------------------------------------------------------------------
def paraphrase(
    text: str,
    num_return_sequences: int = 4,
    num_beams: int = 4,
    max_new_tokens: int = 80,
    tone: str = "polite",
    style: str = "polite",
    strategy_hints: List[str] = None,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    min_new_tokens: int = 8,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> List[str]:
    """
    Generate polite paraphrases for `text`.
    Parameters align with Transformers.generate(); defaults favor safe beam search.
    Set do_sample=True for more diverse outputs (with top_p / temperature).
    """
    text = (text or "").strip()
    if not text:
        return []

    tok, mdl = _load_model()
    prompt = _build_prompt(text, tone=tone, style=style, strategy_hints=strategy_hints)
    enc = tok(prompt, return_tensors="pt", truncation=True).to(_DEVICE)

    gen_kwargs = _generation_kwargs(
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
    )

    with torch.no_grad():
        outs = mdl.generate(**enc, **gen_kwargs)

    decoded = [tok.decode(o, skip_special_tokens=True) for o in outs]

    # Gentle post-processing and de-dup
    cleaned = [_soft_postprocess(s) for s in decoded if s and s.strip()]
    uniq = deduplicate(cleaned)

    # Safety: limit to requested count after de-dup
    return uniq[: max(1, int(num_return_sequences))]


# ---------------------------------------------------------------------
# EXTRA CONVENIENCE (DETERMINISTIC AND SAMPLING MODELS)
# ---------------------------------------------------------------------
def paraphrase_beam(
    text: str,
    num_return_sequences: int = 4,
    num_beams: int = 6,
    max_new_tokens: int = 96,
    tone: str = "polite",
    style: str = "polite",
    strategy_hints: List[str] = None,
) -> List[str]:
    """Deterministic beam search for stable outputs."""
    return paraphrase(
        text=text,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        tone=tone,
        style=style,
        strategy_hints=strategy_hints,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        min_new_tokens=10,
        length_penalty=1.0,
        early_stopping=True,
    )


def paraphrase_sample(
    text: str,
    num_return_sequences: int = 6,
    num_beams: int = 4,
    max_new_tokens: int = 96,
    tone: str = "polite",
    style: str = "polite",
    strategy_hints: List[str] = None,
) -> List[str]:
    """Diverse sampling for exploration (nucleus sampling)."""
    return paraphrase(
        text=text,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,     # Diverse beam sampling if >1
        max_new_tokens=max_new_tokens,
        tone=tone,
        style=style,
        strategy_hints=strategy_hints,
        do_sample=True,
        temperature=0.9,
        top_p=0.92,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
        min_new_tokens=10,
        length_penalty=0.9,
        early_stopping=True,
    )


# ---------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ex = "send me the report asap"
    print("INPUT:", ex)
    outs = paraphrase_beam(ex, num_return_sequences=5, num_beams=6, max_new_tokens=96)
    for i, o in enumerate(outs, 1):
        print(f"{i:02d}. {o}")

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------