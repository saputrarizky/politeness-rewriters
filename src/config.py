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
config.py — Global configuration for the Politeness Rewriter project
--------------------------------------------------------------------
Centralized constants for model paths, thresholds, and tuning weights.
These values are imported across:
  • classifier_train.py / classifier_infer.py
  • rewrite_t5.py
  • rerank.py
  • pipeline.py
"""

# --------------------------------------------------------------
# MODEL SETUP AND BACKBONE
# --------------------------------------------------------------

MODEL_CLS   = "distilroberta-base"                       # Classifier backbone for politeness detection
MODEL_PARA  = "Ateeqq/Text-Rewriter-Paraphraser"         # T5-style paraphraser baseline
PARAPHRASE_PROMPT_STYLE = "concise"                      # Prompt style control (Concise and Verbose for minimal and detailed prompts)
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # SBERT model for semantic similarity

# --------------------------------------------------------------
# CLASSIFIER AND GENERATIVE PARAMETERS
# --------------------------------------------------------------

MAX_LEN = 128   # Max sequence length for politeness classifier inputs

# --------------------------------------------------------------
# THRESHOLD AND SCORING WEIGHTS (ADJUSTABLE FOR TESTING)
# --------------------------------------------------------------

POLITE_THRESHOLD = 0.70   # Skip rewrite if already polite enough (prob ≥ this)
W_POLITE = 0.60           # Rerank weight for politeness probability
W_SIM    = 0.40           # Rerank weight for semantic similarity

# --------------------------------------------------------------
# Notes:
# - You can safely tweak POLITE_THRESHOLD between 0.65–0.75
#   depending on how conservative you want the system to be.
# - W_POLITE + W_SIM should ideally sum ≈ 1.0.
#   Example: W_POLITE=0.6, W_SIM=0.4 balances tone vs. meaning.
# - SBERT_MODEL can be replaced with any sentence-transformers checkpoint
#   (e.g., "all-mpnet-base-v2" for higher accuracy).
# --------------------------------------------------------------

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------