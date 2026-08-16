"""
Central configuration for the prompt reliability study.
Edit this file to change models, sample sizes, or thresholds —
every other script reads from here so you only change one place.
"""


OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0          # greedy decoding — removes sampling randomness
MAX_TOKENS = 450   
REQUEST_TIMEOUT = 120     

MODELS = {
    "llama3.2:3b":  {"label": "Llama-3.2 (3B)",     "ram_gb": 2.0},
    "gemma3:4b":    {"label": "Gemma-3 (4B)",       "ram_gb": 3.3},
    "phi4-mini":    {"label": "Phi-4-mini (3.8B)",  "ram_gb": 3.0},
    "mistral:7b":   {"label": "Mistral-7B",         "ram_gb": 5.5},
}

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
  
        "split": "test",
    },
}

SAMPLE_SIZE_PER_DOMAIN = 200 
PROMPT_STYLES = ["bare", "instructed", "roleplay", "negation"]

SIMILARITY_THRESHOLD = 0.85     
CONFIDENCE_THRESHOLD = 0.75    
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_PROMPTS_DIR = "data/prompts"
RESULTS_RAW_DIR = "results/raw_responses"
RESULTS_SCORED_DIR = "results/scored"
RESULTS_SUMMARY_DIR = "results/summary"