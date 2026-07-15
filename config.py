"""
Central configuration
Edit this file to change models, sample sizes, or thresholds 
every other script reads from here.
"""

# Ollama connection
OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0          # removes sampling randomness
MAX_TOKENS = 200
REQUEST_TIMEOUT = 120     

# Models under test
# Add/remove entries here;
MODELS = {
    "llama3.2:3b":  {"label": "Llama-3.2 (3B)",   "ram_gb": 2.0},
    "gemma2:2b":    {"label": "Gemma-2 (2B)",     "ram_gb": 1.5},
    "phi3:mini":    {"label": "Phi-3 Mini (3.8B)", "ram_gb": 2.5},
    "mistral:7b":   {"label": "Mistral-7B",       "ram_gb": 5.5},
    # "llama3.1:8b": {"label": "Llama-3.1 (8B)",  "ram_gb": 4.7},
}

# Domains and their benchmark datasets
# Used only during the empirical research phase (building the reliability matrix).
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
        "hf_config": "abercrombie",   
        "split": "test",
    },
}

SAMPLE_SIZE_PER_DOMAIN = 200  

# Prompt styles
# Four content-preserving reformulations per question.
PROMPT_STYLES = ["bare", "instructed", "roleplay", "negation"]

# Consistency / confidence 
SIMILARITY_THRESHOLD = 0.85     # min cosine similarity for a valid paraphrase
CONFIDENCE_THRESHOLD = 0.75     # will be calibrated from our own results later

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_PROMPTS_DIR = "data/prompts"
RESULTS_RAW_DIR = "results/raw_responses"
RESULTS_SCORED_DIR = "results/scored"
RESULTS_SUMMARY_DIR = "results/summary"
