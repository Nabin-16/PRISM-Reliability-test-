from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_TEMPLATES_DIR = DATA_DIR / "templates"
DATA_PROMPTS_DIR = DATA_DIR / "prompts"
DATA_AUDIT_DIR = DATA_DIR / "audit"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_DIR / "raw_responses"
RESULTS_PARSED_DIR = RESULTS_DIR / "parsed"
RESULTS_SCORED_DIR = RESULTS_DIR / "scored"
RESULTS_SUMMARY_DIR = RESULTS_DIR / "summary"

RANDOM_SEED = 2026
SAMPLE_SIZE_PER_DATASET = 200

DATASETS = {
    "arc_challenge": {
        "huggingface_dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
    },
    "sciq": {
        "huggingface_dataset": "allenai/sciq",
        "config": None,
        "split": "test",
    },
}

PROMPT_CONDITIONS = ("P0", "P1", "P2", "P3", "P4")

OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0.0
NUM_PREDICT = 450
REQUEST_TIMEOUT = 180

MODELS = {
    "llama3.2:3b": {"label": "Llama-3.2 (3B)"},
    "gemma3:4b": {"label": "Gemma-3 (4B)"},
    "phi4-mini:latest": {"label": "Phi-4-mini (3.8B)"},
    "mistral:7b": {"label": "Mistral-7B"},
}

EXPERIMENT_ID = "PRISM-EXP1-v1"
TEMPLATE_VERSION = "1.0"
PROTOCOL_VERSION = "1.0"
