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
baseline_rules.py — Expanded rule-based rewriter for Politeness Rewriter
-----------------------------------------------------------------------
Integrates linguistic strategies inspired by Stanford Politeness research:
  - gratitude, apology, hedges, request_softeners, deference, greetings
Adds tone control ("neutral" | "polite" | "extra-polite") and rule explanations.

Usage:
    from src.baseline_rules import rewrite_rule_based
    out = rewrite_rule_based("send me the report asap", level="polite", explain=True)
"""

# -------------------------------------------------------
# SETUP
# -------------------------------------------------------
import re
import random
from typing import Dict, List, Tuple

# ----------------------------------------------------------
# LEXICONS, MARKERS, PATTERNS
# ----------------------------------------------------------
# Add more lexicons
LEXICONS = {
    "gratitude": [
        "thank you", "thanks", "much appreciated", "i appreciate that",
        "thank you so much", "really appreciate it", "i truly appreciate it"
    ],
    "apology": [
        "sorry", "i apologize", "excuse me", "pardon me",
        "my apologies", "i sincerely apologize"
    ],
    "hedges": [
        "maybe", "perhaps", "a bit", "slightly", "i think",
        "i feel", "it seems",
        "possibly", "i guess", "i suppose"
    ],
    "request_softeners": [
        "please", "could you", "would you", "would you mind",
        "can you", "may i",
        "would it be possible to",
        "is there any chance you could",
        "i’d appreciate if you could"
    ],
    "deference": [
        "sir", "madam", "dear", "kindly"
    ],
}

# Add more negative markers
NEGATIVE_MARKERS = {
    "profanity": re.compile(r"\b(damn|hell|stupid|idiot|fuck|shit|wtf|bastard|moron)\b", re.I),
    "demandy": re.compile(r"\b(asap|immediately|right away|now|at once)\b", re.I),
    "blame": re.compile(r"\byou (forgot|didn't|failed to)\b", re.I),

    # Extended negative markers
    "accusation": re.compile(r"\byou (never|always|keep|constantly)\b", re.I),
    "harsh_modal": re.compile(r"\b(don't|stop|you better|you should)\b", re.I),
    "rude_question": re.compile(r"\bwhy (did|didn’t|didn't you|haven't you)\b", re.I),
    "insult": re.compile(r"\b(useless|worthless|terrible|awful|dumb)\b", re.I),
}

# Add more imperative patters
IMPERATIVE_PATTERNS = [
    # Initial imperative declarations
    (re.compile(r"\b(send|give|provide|share)\s+(me|us)\b", re.I), "could you please \\1 \\2"),
    (re.compile(r"\b(send|give|provide|share)\s+(the|a|an)\b", re.I), "could you please \\1 \\2"),
    (re.compile(r"\b(fix|answer|explain|solve|check|do)\s+(this|it|that)\b", re.I),
     "could you please \\1 \\2"),
    (re.compile(r"\b(i need|i want|i expect)\b", re.I), "could i please have"),
    (re.compile(r"\b(get|move|leave)\s+out\b", re.I), "step outside for a moment"),

    # New extended imperative softeners
    (re.compile(r"\btell me\b", re.I), "could you please tell me"),
    (re.compile(r"\bgive me\b", re.I), "could you please give me"),
    (re.compile(r"\bexplain why\b", re.I), "could you please explain why"),
    (re.compile(r"\bstop\b(?!.*for a moment)", re.I), "could you please stop"),
    (re.compile(r"\bdon't\b", re.I), "please avoid"),
    (re.compile(r"\byou should\b", re.I), "it might help if you could"),
    (re.compile(r"\b(reply|respond)\b", re.I), "could you please \\1"),
    (re.compile(r"\bcheck\b", re.I), "could you please check"),
    (re.compile(r"\bdo it again\b", re.I), "could you please try that again"),
    (re.compile(r"\btell (him|her|them)\b", re.I), "could you please tell \\1"),
    (re.compile(r"\bdon't send\b", re.I), "please avoid sending"),
    (re.compile(r"\bdon't send me\b", re.I), "please avoid sending me"),

    # Optional command softener for direct imperatives
    (re.compile(r"^(send|give|fix|check|explain|tell|do|make)\b", re.I),
     "could you please \\1"),
]

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def capitalize(s: str) -> str:
    return s[0].upper() + s[1:] if s else s

def ensure_sentence_end(s: str) -> str:
    s = s.strip()
    if not s.endswith(('.', '?', '!')):
        s += '.'
    return s

def random_choice(words: List[str]) -> str:
    return random.choice(words) if words else ""

# ----------------------------------------------------------
# CORE REWRITER PIPELINE
# ----------------------------------------------------------
def rewrite_rule_based(text: str,
                       level: str = "polite",
                       explain: bool = False) -> Dict[str, str]:
    """
    Main entry for rule-based rewriting.

    Args:
        text (str): input sentence
        level (str): 'neutral', 'polite', or 'extra-polite'
        explain (bool): if True, return dict with diagnostics

    Returns:
        dict with keys: 'output', 'strategies', 'applied_rules'
    """
    original = text.strip()
    if not original:
        return {"output": "", "strategies": [], "applied_rules": []}

    out = original
    strategies, applied = [], []

    # Remove profanity/demandy words
    for name, rx in NEGATIVE_MARKERS.items():
        if rx.search(out):
            out = rx.sub("", out)
            applied.append(f"removed_{name}")

    # Imperative → modal request
    for pat, repl in IMPERATIVE_PATTERNS:
        if pat.search(out):
            out = pat.sub(repl, out)
            applied.append("imperative_to_request")
            strategies.append("request_softeners")

    # Prefix with softener if missing
    lower = out.lower()
    if not any(lower.startswith(s) for s in LEXICONS["request_softeners"]):
        out = f"Could you please {out[0].lower()}{out[1:]}"
        applied.append("added_softener")
        strategies.append("request_softeners")

    # Negation softening
    out = re.sub(r"\byou didn't\b", "it seems the task wasn't done", out, flags=re.I)
    out = re.sub(r"\byou forgot\b", "it seems the step was missed", out, flags=re.I)
    out = re.sub(r"\byou failed\b", "it appears this wasn’t completed", out, flags=re.I)

    # Tone control
    level = level.lower()
    if level == "neutral":
        pass  # Minimal edits only
    elif level == "polite":
        # Maybe add hedges or gratitude randomly
        if random.random() < 0.4:
            hedge = random_choice(LEXICONS["hedges"])
            out = hedge + ", " + out
            strategies.append("hedges")
        if random.random() < 0.5:
            out = ensure_sentence_end(out) + " " + random_choice(LEXICONS["gratitude"]).capitalize() + "."
            strategies.append("gratitude")
    elif level == "extra-polite":
        # Include greetings + apology + gratitude + deference
        greet = random_choice(["Hello,", "Good morning,", "Good afternoon,"])
        out = f"{greet} {out}"
        strategies.append("greetings")

        if random.random() < 0.5:
            out = f"{random_choice(LEXICONS['apology']).capitalize()}, {out}"
            strategies.append("apology")

        if not any(k in out.lower() for k in LEXICONS["gratitude"]):
            out = ensure_sentence_end(out) + " " + random_choice(LEXICONS["gratitude"]).capitalize() + "."
            strategies.append("gratitude")

        if random.random() < 0.4:
            out += " I would greatly appreciate your help."
            strategies.append("deference")

    # Clean spacing
    out = re.sub(r"\s{2,}", " ", out).strip()

    # Capitalize + end punctuation
    out = ensure_sentence_end(capitalize(out))

    if explain:
        return {"output": out, "strategies": sorted(set(strategies)), "applied_rules": applied}
    else:
        return {"output": out}

# ----------------------------------------------------------
# SIMPLE TEST
# ----------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "send me the report asap",
        "you forgot to reply yesterday",
        "fix this issue immediately",
        "i want that file",
        "can you check this",
        "give me your response now",
    ]
    for s in samples:
        print("=" * 60)
        result = rewrite_rule_based(s, level="extra-polite", explain=True)
        print(f"IN:  {s}")
        print(f"OUT: {result['output']}")
        print(f"→ Strategies: {result['strategies']}")
        print(f"→ Rules: {result['applied_rules']}")

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------