# 🧠 Classifier-Guided Politeness Rewriting via Controlled Text Generation
**Seoul National University — 2025-2 Introduction to NLP Course (001)**  
**Team:** Saputra Rizky Johan, Bat-Orshikh Butemj, Shu Xian Chow  
**Instructors:** Prof. Hwang Seung-Won, TAs Kim Jong-Yoon & Han Sang-Eun  

## 📘 Overview
The **Politeness Rewriter** is a hybrid NLP system that **detects**, **classifies**, and **rewrites** user input into polite, respectful text using a combination of **transformer-based classifiers** and **controlled text generation**. 

It specifically combines:
- A **DistilRoBERTa politeness classifier** (trained on the *Stanford Politeness Corpus* from ConvoKit),
- A **T5-based paraphraser** conditioned for polite rewriting,
- A **modular rewrite pipeline** with scoring, reranking, and explainability,
- And an optional **Gradio demo app** for interactive rewriting.

The project demonstrates *classifier-guided text style transfer* — transforming the *tone* of a sentence without altering its *semantic meaning*.

## 🎯 Project Purpose
Politeness plays a critical role in communication, especially for social networking services within professional settings, AI chatbots, digital assistants, and automated email generation.  
This project aims to:
1. Build a lightweight yet robust **politeness classifier**.
2. Integrate it with a **T5-style paraphraser** to automatically rewrite impolite or neutral sentences into polite versions.
3. Provide a **human-interpretable pipeline** where users can trace:
   - Classification probability,
   - Semantic similarity,
   - Rewrite quality.

The ultimate goal is to construct a language-generation systems that is more **socially intelligent** with our own training set and evaluation results.

## 📚 Dataset: *Stanford Politeness Corpus (ConvoKit)*
We specifcally use the **Stanford Politeness Corpus**, accessed via [`convokit`](https://convokit.cornell.edu). It contains **11,000+ sentences** from online forums and Wikipedia requests, annotated with politeness scores (ranging from impolite to polite).

### Dataset pipeline:
- Downloaded automatically via `src/download_data.py`
- Split into:
  - `train.jsonl`
  - `val.jsonl`
  - `test.jsonl`
- Cleaned and tokenized using `transformers`’ tokenizer
- Used both for classifier training and for evaluation of rewriting quality

## 🧩 Architecture Overview
### 1. **Classifier Module**
**File:** `src/classifier_train.py` / `src/classifier_infer.py`  
- Model: `distilroberta-base`
- Task: Binary classification (polite = 1, impolite = 0)
- Loss: Binary Cross-Entropy
- Optimizer: AdamW with cosine scheduler
- Output: politeness probability `p(politeness|x)`

### 2. **Rewriter Module**
**File:** `src/rewrite_t5.py`  
- Model: `Ateeqq/Text-Rewriter-Paraphraser` (or `humarin/chatgpt_paraphraser_on_T5_base`)
- Conditional prompts:
  - `concise:polite` → “Rewrite this sentence politely:”
  - `concise:extra-polite` → “Rewrite this sentence politely and formally:”
- Uses beam search or nucleus sampling with configurable parameters:
  - `num_beams`, `top_p`, `temperature`, `repetition_penalty`
- Post-processing cleans generated text, removes meta-prompts, and enforces sentence termination.

### 3. **Pipeline**
**File:** `src/pipeline.py`  
- Takes raw input → runs classifier → computes politeness probability  
- If `p ≥ 0.70`: skip rewriting  
- Else:
  - Generate rewrite candidates via `rewrite_t5.py`
  - Score each candidate:
    - Politeness probability (from classifier)
    - Semantic similarity (from SBERT)
  - Weighted reranking:
    ```ini
    FinalScore = 0.6 * P_polite + 0.4 * Similarity
    ```
  - Output top-ranked rewrite with metadata:
    - `before_prob`, `after_prob`, `similarity`, `rule_based` flag

### 4. **Evaluation**
**File:** `src/eval.py`  
Runs automatic evaluation on the test set:
- Computes classifier politeness improvement
- Reports average similarity (SBERT cosine)
- Optionally exports CSV samples

### 5. **Interactive App**
**File:** `app.py`  
Gradio interface for real-time rewriting:
- Input box for text
- Dropdown for tone (`polite`, `extra-polite`)
- Checkbox for rule-based filter
- Displays:
  - Before/after politeness scores
  - Generated rewrite
  - Example outputs history

## ⚙️ File Structure
The project file is structured as show below. However, note that the layout of the project may change in the future with additional changes or implementations to be made in the project itself. 
```php
politeness-rewriters/
├── app.py
├── src/
│   ├── config.py
│   ├── classifier_train.py
│   ├── classifier_infer.py
│   ├── rewrite_t5.py
│   ├── rerank.py
│   ├── pipeline.py
│   ├── eval.py
│   ├── download_data.py
│   └── utils/
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── out/
│   ├── classifier/
│   ├── eval_samples.csv
│   └── model checkpoints
├── run_politeness_colab.ipynb
└── README.md
```
## 🔬 Methodology

### Problem Definition
We aim to perform **style transfer** on short requests/sentences to make them more **polite** while preserving their **semantic meaning**. This is framed as a pipeline with:
1. **Politeness classification** (is the input already polite?),
2. **Controlled rewriting** (make the text more polite using a conditional paraphraser),
3. **Reranking** (choose the best candidate using a composite score).

### Data
- **Corpus:** *Stanford Politeness Corpus* via ConvoKit.  
- **Signals used:**
  - Binary politeness labels (impolite vs. polite),
  - Optional continuous politeness scores (for thresholding / analysis).

### Models
- **Classifier:** DistilRoBERTa fine-tuned for binary politeness (0/1).  
- **Generator:** T5/BART‑style paraphraser (Hugging Face checkpoint), conditioned by a **concise prompt** (e.g., “Rewrite this sentence politely:”).
- **Similarity:** SBERT (`all-MiniLM-L6-v2`) for semantic preservation.

### Pipeline
1. **Pre-check:** If `P(politeness|x) ≥ τ` (default `τ=0.70`), **skip rewriting** (return original).  
2. **Rule pass (optional):** Remove obvious harshness (e.g., profanity, “ASAP”) before feeding to the generator.  
3. **Neural candidates:** Generate `k` rewrites with a friendly prompt (concise, no “Input:” preambles).  
4. **Reranking:** Score each candidate `y`:
   - `P_polite(y)`: classifier politeness probability  
   - `Sim(x, y)`: SBERT cosine similarity  
   - `Score(y) = W_POLITE * P_polite(y) + W_SIM * Sim(x, y)` (defaults: 0.6 / 0.4)  
   - **Strategy bonus (±0.1):** Detect gratitude/softeners/hedges and negative markers; add tiny bonus/penalty for tie‑breaking.  
5. **Output:** Highest‑scoring candidate, plus rich metadata for UI and evaluation.

### Rationale
- **Classifier‑guided generation** improves controllability (politeness actually increases).  
- **SBERT constraint** avoids semantic drift.  
- **Rule‑based stage** handles easy wins (e.g., “now!” → “when you have a moment”).  
- **Strategy detectors** (gratitude/hedges) help with tie‑breaks and qualitative alignment to known politeness markers (inspired by Stanford features).

## 🧩 Module-by-Module Explanation

### `config.py`
- `MODEL_CLS`, `MODEL_PARA`, `SBERT_MODEL`
- Rerank weights `W_POLITE`, `W_SIM`
- Threshold `POLITE_THRESHOLD`
- **Prompt style**: `PARAPHRASE_PROMPT_STYLE = "concise"` (recommended)

### `download_data.py`
- Downloads ConvoKit **Stanford Politeness** corpus.
- Normalizes to JSONL with fields `{"text": ..., "label": 0/1, "score": ...}`.
- Splits into `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

### `classifier_train.py`
- Loads JSON/JSONL (robust key inference: `text`, `label`, `score`).
- Fine‑tunes `distilroberta-base` with:
  - `TrainingArguments(evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True)`
  - Macro F1 and accuracy
- Saves to `out/classifier/model/` (model, tokenizer, metrics).

### `classifier_infer.py`
- CLI + function API to score text or a file.
- Returns dict: `{"polite": p1, "impolite": p0, "label": "polite"/"impolite"}`.

### `rewrite_t5.py`
- Loads HF paraphraser (`MODEL_PARA`).
- **Prompts:** “concise” family avoids generator echoing instruction preambles.
- Generation knobs: `num_beams`, `top_p`, `temperature`, `repetition_penalty`, `no_repeat_ngram_size`.
- **Post-processing:** trims quotes/bullets, normalizes spaces, ensures sentence end, **removes instructiony prefixes** (use `_strip_instruction_prefix()` from `pipeline.py` for final cleaning).

### `rerank.py`
- Encodes original and candidates with SBERT.
- Politeness scores via the classifier.
- Weighted composite + optional strategy-aware bonus.
- Returns `(best_text, [(text, polite, sim, score), ...])`.

### `pipeline.py`
- Orchestrates the full flow:
  - Pre-check → Rules → Paraphrase → Rerank → Strategy bonus
  - **Final clean:** stores `output_raw` and `output` (cleaned via `_strip_instruction_prefix()`).
- Exposes `rewrite()` used by `app.py` and CLI.

### `app.py`
- Gradio UI with:
  - Input box, tone selector, rule toggle
  - Before/after sliders, debug info
  - History table + CSV export
- Launches on a given port; set `server_port` if 7860 is busy.

### `eval.py`
- Samples N items from `data/test.jsonl`.
- Runs `rewrite()` to measure **% improved politeness** and exports CSV rows with before/after.

## 🚀 Executables & How to Run
### Mount Via Drive
Upload `politeness-rewritter.zip` or mount to drive and execute the included notebook:
```bash
#!/bin/bash
%cd /content
!unzip -o politeness-rewriters.zip
!pip install -r requirements.txt
!python src/pipeline.py --text "why are you so slow in replying"
```

## 🛠️ Recommended Adjustments & Troubleshooting

### Generation Prompt Style
- Set in `config.py`: `PARAPHRASE_PROMPT_STYLE = "concise"`  
- Avoid `verbose` prompts; they can leak into outputs (instruction echo).

### Stable Candidate Quality
- Use **beam search** for stability: `num_beams=6`, `num_return_sequences=4`.
- For diversity: `do_sample=True`, `top_p∈[0.9,0.95]`, `temperature∈[0.8,1.0]`.

### Threshold and Beam Search
- Tune `POLITE_THRESHOLD` between 0.65 - 0.75 to the dataset (Better range is also acceptable as long as it is documented)
- Increase `num_beams` for more diverse rewrites at cost of latency

### Classifier Oddities
- If CUDA asserts trigger: set `--batch_size 8` or `CUDA_LAUNCH_BLOCKING=1`.
- Label errors: ensure labels are 0/1; use `score_threshold` if using continuous scores.

### Training Parameterization
- Increase the `--Epoch`, `--lr`, `--batch_size` for better training results
- Implement a balance training classifier betwen `concise` and `verbose` cases.

### Model Choice (`MODEL_PARA`)
- Start with: `Vamsi/T5_Paraphrase_Paws` or `Ateeqq/Text-Rewriter-Paraphraser`.
- If gated/private errors (`401`), switch to a public checkpoint.

## 🔬 Expected Output
### Input
```arduino
"send me the document now."
```
### Output
| Metric | Before | After
|---|---|---:|
| Politeness Probability | ~0.2 | ~ 0.9 |
| Similarity | – | ~ 0.9 |
| Rewritten Text | "Could you please send me the document when you have a moment?" | 

Thus, the system should output a single sentence that is:
- **Polite:**
- **Semantically Faithful:** still a request for the document  
- **Clean:** No instruction preambles like “Rewrite the following…”  
- **Metadata:** (for UI/Eval) before/after politeness, similarity, strategy markers.

## 📈 Future Improvements
1. **Context-Aware Rewriting:**  
   Extend from sentence-level to paragraph-level politeness adjustment.
2. **Multi-Style Transfer:**  
   Add “formal”, “casual”, and “friendly” modes using separate prompt templates.
3. **Explainability:**  
   Highlight specific tokens that influence politeness classifications
4. **Better Reranker:**  
   Train a small MLP on features: `P_polite`, `similarity`, strategy counts, length, punctuation.
5. **Adversarial Evaluation:**  
   Check robustness against tricky inputs (sarcasm, negation, passive aggression).
6. **Multilingual Extension:**  
   Korean/English switching, using mT5 + multilingual SBERT.
7. **User Controls in UI:**  
   Sliders for “extra‑polite”, “neutral”, “brief vs detailed”, optional “add gratitude”.

## 🧪 Example Results
The project is expected to have these results in accordance to a well trained and parameterized module. These changes are expectations for the results during the final simulation, such that:
| Input | Output (clean) | Expected Politeness Change
|---|---|---:|
| Send me the report now! | Could you please send me the report when you have a moment? | +0.5 ~ +0.65 |
| Why didn’t you reply earlier? | Could you please let me know when you had a chance to respond? | +0.45 ~ +0.6 |
| Do it properly next time. | Please make sure it’s done correctly next time. | +0.55 ~ +0.7 |

## 👥 Team Contributions
The work is contributed as follows: (Please update this section as the project evolves.)
- **Saputra Rizky Johan** — Project Architecture, Pipeline Design, Gradio Integration, Classifier Integration, Reranking Integration, Rewriting T5 Integration, Rule Based Modulator, README Authoring, Github Management, Data Training, Notebook Integration.
- **Bat‑Orshikh** — Data Training, Data Evaluation, Error Analysis, UI/UX Management, Simulation and Testing, Documentations, Github Management.
- **Shu Xian Chow** — Data Training, Data Evaluation, Error Analysis, UI/UX Management, Simulation and Testing, Documentations, Github Management, Data pipeline, SBERT integration.

## 🧠 Key Takeaways
This project demonstrates how **hybrid architectures** — combining discriminative and generative models — can perform **controlled style transfer** effectively.  
It balances **semantic preservation** with **tone improvement**, producing rewrites that are both polite and natural, paving the way for socially aware AI communication systems.
