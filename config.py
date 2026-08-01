"""
Central configuration for the prompt reliability study.
Edit this file to change models, sample sizes, or thresholds —
every other script reads from here so you only change one place.
"""

# ── Ollama connection ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0          # greedy decoding — removes sampling randomness
MAX_TOKENS = 450   # bumped from 200 — this model reasons at length
                    # ("Step 1... Step 2..." style) before concluding;
                    # 200 was cutting responses off mid-analysis,
                    # before any real answer was ever stated
REQUEST_TIMEOUT = 120     # seconds — small models can be slow on CPU

# ── Models under test ──────────────────────────────────────────────
# Deliberately updated PAST Hariprasad et al. (2026)'s exact model
# set — gemma3:4b replaces gemma2:2b, phi4-mini replaces phi3:mini —
# to use current-generation (mid-2026) small models rather than the
# generation the reference paper tested. This trades some direct
# model-level comparability for relevance: the finding this project
# reports extends the general PHENOMENON Hariprasad identified
# (prompt sensitivity, reliable incorrectness) to new domains AND
# to the current generation of small models, rather than replicating
# their exact model lineup. State this explicitly in the report —
# it's a deliberate choice, not an oversight.
#
# llama3.2:3b and mistral:7b are unchanged from the reference set —
# no smaller/newer direct successor exists for either at this size
# class as of mid-2026, so there was nothing to update.
MODELS = {
    "llama3.2:3b":  {"label": "Llama-3.2 (3B)",     "ram_gb": 2.0},
    "gemma3:4b":    {"label": "Gemma-3 (4B)",       "ram_gb": 3.3},
    "phi4-mini":    {"label": "Phi-4-mini (3.8B)",  "ram_gb": 3.0},
    "mistral:7b":   {"label": "Mistral-7B",         "ram_gb": 5.5},
    # Llama 3.1 8B is not a core model — it's redundant with Mistral-7B
    # for testing whether size alone improves consistency. Keep it as
    # an optional bonus data point if you want a 5th comparison point,
    # not part of the primary reliability matrix.
    # "llama3.1:8b": {"label": "Llama-3.1 (8B)",    "ram_gb": 4.7},
}

# ── Domains and their benchmark datasets ─────────────────────────
# Used only during the empirical research phase (building the
# reliability matrix). The deployed app does NOT classify domain —
# it applies whichever prompt style is globally best for the loaded
# model, decided from this matrix.
DOMAINS = {
    "education": {
        "hf_dataset": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "test",
    },
    "science": {
        "hf_dataset": "allenai/sciq",
        "hf_config": None,
        "split": "test",
    },
    "legal": {
        "hf_dataset": "nguha/legalbench",
        "hf_config": "consumer_contracts_qa",
        # 400 examples, native Yes/No format on real terms-of-service
        # interpretation. Replaces an earlier choice, abercrombie
        # (only ~95 examples, narrow 5-way trademark classification,
        # forced negation into 3-valid-of-4-options ambiguity). See
        # the docstring in src/load_datasets.py:load_legal() for the
        # full reasoning — this task's binary structure also makes
        # the negation prompt style trivially well-defined (NOT Yes =
        # No), a real methodological improvement, not just a
        # data-volume fix.
        "split": "test",
    },
}

SAMPLE_SIZE_PER_DOMAIN = 200   # matches the reference study's 200/dataset

# ── Prompt styles ──────────────────────────────────────────────────
# Four content-preserving reformulations per question.
# "negation" is a deliberate addition beyond Hariprasad et al.'s five
# styles (Original/Formal/Simplified/Roleplay/Direct) — it stresses
# logical inversion rather than just register/tone, which their
# taxonomy does not cover.
PROMPT_STYLES = ["bare", "instructed", "roleplay", "negation"]

# ── Consistency / confidence ───────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85     # min cosine similarity for a valid paraphrase
CONFIDENCE_THRESHOLD = 0.75     # calibrate this from your own results later

# ── Paths ────────────────────────────────────────────────────────
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_PROMPTS_DIR = "data/prompts"
RESULTS_RAW_DIR = "results/raw_responses"
RESULTS_SCORED_DIR = "results/scored"
RESULTS_SUMMARY_DIR = "results/summary"