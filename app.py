### -------------------------------------------------------------------------------------------------
## Team: 사푸트라 (Saputra Rizky Johan), 바트오르식 (Butemj Bat-Orshikh), 쉬슈잔 (Shu Xian Chow)
## Institution: Seoul National University, South Korea
## Course: 2025-2 Introduction to Natural Language Processing (001)
## Instructors: 황승원 (Prof), 김종윤(TA), 한상은 (TA)
## Project: Classifier-Guided Politeness Rewriting via Span Detection and Controlled Text Generation
## Corpus: Stanford Politeness Corpus (Convokit)
## Note: If additional implementations are to be made, please document them at the README file
### -------------------------------------------------------------------------------------------------

# ------------------------------------------------------
# SPECIFICATIONS
# ------------------------------------------------------
"""
app.py — Politeness Rewriter (Hybrid Edition)
---------------------------------------------
Full-featured Gradio interface for the Politeness Rewriter system.

Pipeline:
  Classifier → Rule-based edits → T5 Paraphraser → SBERT Reranker

Features:
  • Politeness probability bars (before/after)
  • Rule toggle & tone selector
  • History table + export to CSV
  • Example prompts for quick testing
"""

# ------------------------------------------------------
# SETUP
# ------------------------------------------------------
import sys, os, json
from datetime import datetime
import pandas as pd
import gradio as gr

# Ensure src import works from any cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import rewrite  # Main rewrite pipeline

# ------------------------------------------------------
# SET GLOBAL VARIABLES
# ------------------------------------------------------
APP_TITLE = "Politeness Rewriter (Hybrid Edition)"
APP_SUB = "Classifier → Rule edits → T5 Paraphraser → Reranker"
HISTORY = []  # Session-wide rewrite history

# ------------------------------------------------------
# CORE REWRITE WRAPPER LOGIC
# ------------------------------------------------------
def run_rewriter(text, tone, use_rule, show_probs):
    """
    Wrapper for rewrite() → prepares outputs for Gradio UI.

    Args:
        text (str): user input
        tone (str): target tone ("Polite" or "Impolite")
        use_rule (bool): whether to apply rule-based preprocessing
        show_probs (bool): display before/after probabilities
    """
    if not text.strip():
        return "", "Please enter some text.", None, None, None

    # Run pipeline
    out = rewrite(text, target_tone=tone.lower(), rule_based=use_rule)

    before_prob = out.get("polite_prob_before", 0.0)
    after_prob = out.get("polite_prob_after", 0.0)
    rewritten = out.get("output", "")

    # Update session history
    HISTORY.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "input": text,
        "output": rewritten,
        "before": before_prob,
        "after": after_prob,
        "rule": use_rule
    })

    # Debug feedback text
    debug_txt = (
        f"Polite prob (before): {before_prob:.3f}\n"
        f"Polite prob (after): {after_prob:.3f}\n"
        f"Rule-based edits: {'Enabled' if use_rule else 'Disabled'}\n"
        f"Target tone: {tone}"
    )

    return rewritten, debug_txt, before_prob, after_prob, pd.DataFrame(HISTORY)


# ------------------------------------------------------
# EXPORT LOGIC
# ------------------------------------------------------
def export_csv():
    """Export current rewrite history to a CSV file."""
    if not HISTORY:
        return None
    df = pd.DataFrame(HISTORY)
    path = f"rewrites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------
# GRADIO INTERFACE LAYOUT
# ------------------------------------------------------
with gr.Blocks(css="""
#title {font-size: 2em; font-weight: bold; color: #2c3e50;}
#sub {color:#555; margin-bottom:1em;}
.output-box {background:#f9f9f9; border-radius:8px; padding:0.8em;}
""") as demo:
    # Header
    gr.Markdown(f"<div id='title'>{APP_TITLE}</div>")
    gr.Markdown(f"<div id='sub'>{APP_SUB}</div>")

    # Main layout
    with gr.Row():
        # Input column
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=5,
                placeholder="Type or paste text here...",
                label="Input"
            )
            tone_choice = gr.Radio(
                ["Polite", "Extra-Polite"],
                value="Polite",
                label="Target Tone"
            )
            rule_toggle = gr.Checkbox(
                value=True,
                label="Enable Rule-Based Preprocessing"
            )
            show_probs = gr.Checkbox(
                value=True,
                label="Show Classifier Probabilities"
            )

            with gr.Row():
                run_btn = gr.Button("Rewrite Text", variant="primary")
                clear_btn = gr.Button("Clear History")

        # Ouput column
        with gr.Column(scale=3):
            rewritten_out = gr.Textbox(
                label="🪶 Rewritten Output",
                lines=5,
                elem_classes=["output-box"]
            )
            debug_out = gr.Textbox(label="Debug Info", lines=4)

            with gr.Row():
                before_bar = gr.Slider(
                    label="Politeness Before",
                    minimum=0, maximum=1, value=0,
                    interactive=False
                )
                after_bar = gr.Slider(
                    label="Politeness After",
                    minimum=0, maximum=1, value=0,
                    interactive=False
                )

    # History and export
    gr.Markdown("### Rewrite History")
    hist_table = gr.Dataframe(
        headers=["timestamp", "input", "output", "before", "after", "rule"],
        wrap=True
    )

    with gr.Row():
        export_btn = gr.Button("Export CSV")
        file_out = gr.File(label="Download CSV")

    # Example sentences
    with gr.Accordion("Example Sentences", open=False):
        examples = gr.Dataset(
            components=[text_input],
            samples=[
                ["Send me the report now!"],
                ["Could you please help me with this issue?"],
                ["Why didn’t you reply earlier?"],
                ["Do it properly next time."],
                ["I’d appreciate your feedback when possible."]
            ]
        )

    # --------------------------------------------------
    # BUTTON INTERACTIONS
    # --------------------------------------------------
    run_btn.click(
        fn=run_rewriter,
        inputs=[text_input, tone_choice, rule_toggle, show_probs],
        outputs=[rewritten_out, debug_out, before_bar, after_bar, hist_table]
    )

    clear_btn.click(lambda: HISTORY.clear() or pd.DataFrame(HISTORY), outputs=[hist_table])
    export_btn.click(fn=export_csv, outputs=file_out)

    gr.Markdown(
        "---\n"
        "Built with using **Gradio**, **Transformers**, and **SBERT**  \n"
        "© 2025 Politeness Rewriter Project"
    )

# ------------------------------------------------------
# APP LAUNCHER
# ------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)

### -------------------------------------------------------------------------------------------------
## END: Add implementations if necessary
### -------------------------------------------------------------------------------------------------