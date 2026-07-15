# Prompt Reliability Through Semantic Multiplexing :- Setup & Run Guide (research phase)

Pipeline: sample benchmark questions → generate 4 prompt-style variants → run through local Ollama models → parse, score, and aggregate into a reliability matrix.

## 1. Setup (once per machine)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
conda activate "your env name"
pip install -r requirements.txt
```

Pull the models anyone model:

```bash
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull mistral:7b
```

## 2. Get the benchmark questions (needs internet)

```bash
python src/load_datasets.py
```

Downloads ARC-Challenge, SciQ, and LegalBench via HuggingFace and
samples 200 questions per domain into `data/processed/`.

## 3. Generate prompt variants (run once)

```bash
python src/prompt_variations.py
```

Creates bare / instructed / roleplay / negation variants for every
question, written to `data/prompts/`.

## 4. Run inference (run per model)

```bash
ollama serve
python src/inference.py --model llama3.2:3b --domain all
```

It's resumable safe to Ctrl+C and rerun, it skips questions already logged.
Raw responses land in `results/raw_responses/`.

## 5. Score and build the reliability matrix

Once all models are done, raw response files are collected in one
`results/raw_responses/` folder:

```bash
python src/consistency_scorer.py
```

Produces:
- `results/summary/reliability_matrix.csv` — accuracy, consistency, UNKNOWN rate per model × domain
- `results/summary/style_breakdown.csv` — accuracy per model × domain × style (feeds the Prompt Normalizer's "best style" lookup)