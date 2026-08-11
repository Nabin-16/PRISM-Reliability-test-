# Prompt Reliability Study — Setup & Run Guide

Pipeline: sample benchmark questions → generate 4 prompt-style variants
→ run through local Ollama models → parse, score, and aggregate into
a reliability matrix.

## 1. Setup (run once per machine)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pull the models this teammate is responsible for (see team assignment):

```bash
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull phi4-mini
ollama pull mistral:7b
```

**Note on model choice:** this deliberately updates PAST Hariprasad et
al. (2026)'s exact model set — `gemma3:4b` replaces `gemma2:2b`,
`phi4-mini` replaces `phi3:mini` — to use current-generation (mid-2026)
small models rather than the generation the reference paper tested.
`llama3.2:3b` and `mistral:7b` are unchanged since no smaller/newer
direct successor exists at that size class yet. This trades some
direct model-level comparability for relevance: report this finding
as extending the general phenomenon Hariprasad identified (prompt
sensitivity, reliable incorrectness) to new domains AND to the current
generation of small models — not as a strict replication. State this
explicitly and consistently across the proposal, SRS, and defense
slides, since the config file and any older document text need to
agree with each other.

Llama 3.1 8B is not a core model — it's redundant with Mistral-7B for
testing whether size alone improves consistency. Add it in `config.py`
under `MODELS` only if you want an optional 5th bonus data point.

## 2. Get the benchmark questions (run once, needs internet)

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

## 4. Run inference (the long step — run per model)

**Pilot first, before committing to the full run.** Response parsing
is genuinely hard across different models' output styles — validate
before spending hours of compute:

```bash
ollama serve                                          # in a separate terminal
python src/inference.py --model llama3.2:3b --domain education   # Ctrl+C after ~10-15 questions
python src/audit_unknowns.py results/raw_responses/llama3.2_3b_education.json
python src/spot_check.py results/raw_responses/llama3.2_3b_education.json
```

`audit_unknowns.py` surfaces every response the parser couldn't
extract an answer from — check whether a real answer is hiding in an
unrecognized phrasing. `spot_check.py` samples successfully-parsed
responses so you can verify the extracted letter actually matches
what the raw text concludes — "parser gave up" and "parser guessed
wrong but looked confident" are different failure modes, and only
checking UNKNOWNs misses the second one entirely.

If either turns up a genuinely common pattern the parser is missing
(not a one-off), fix `response_parser.py`, then run:

```bash
python src/reparse.py results/raw_responses/llama3.2_3b_education.json
```

This re-applies the current parser logic to already-saved raw text —
no new Ollama calls needed. Run it on every file whenever
`response_parser.py` changes, so old data doesn't silently keep stale
`parsed` values from a previous parser version.

Once a model's pilot batch looks clean, run the real thing:

```bash
python src/inference.py --model llama3.2:3b --domain all
```

Each teammate runs this for their assigned model. It's resumable —
safe to Ctrl+C and rerun, it skips questions already logged.
Raw responses land in `results/raw_responses/`.

## 5. Score and build the reliability matrix

Once all four teammates' raw response files are collected in one
`results/raw_responses/` folder, do a final `reparse.py` pass on all
of them to make sure everyone's data reflects the same, current
parser version before scoring:

```bash
python src/reparse.py results/raw_responses/
python src/consistency_scorer.py
```

Produces:
- `results/summary/reliability_matrix.csv` — accuracy, consistency,
  UNKNOWN rate per model × domain
- `results/summary/style_breakdown.csv` — accuracy per model × domain
  × style (feeds the Prompt Normalizer's "best style" lookup)

## Notes / design decisions worth knowing before touching this code

- **On response parsing philosophy:** `response_parser.py` is
  deliberately kept close to how real evaluation frameworks do this —
  modeled directly on `lm-evaluation-harness` (the framework behind
  HuggingFace's Open LLM Leaderboard), which uses one broad regex per
  task, not an ever-growing pile of model-specific edge cases. Even
  the best regex-based extraction frameworks in the field top out
  around 74% (xFinder, ICLR 2025) — a parser that can't extract every
  possible response format is normal, not broken. **UNKNOWN is a real
  research finding, not a parsing failure to engineer away.** If a
  model rambles instead of stating a letter, that IS an
  instruction-following problem, and it should show up in the data.
  Before adding a new tier to the parser for a response format you
  encounter, ask whether it's genuinely common across multiple models
  — not just a one-off from a single response.

- Hariprasad et al.'s repo (github.com/shravani-01/clinical-llm-eval)
  adds a `statistical_tests.py` step (Wilcoxon signed-rank + McNemar)
  for significance testing between models — worth doing before
  finalizing results if time allows.
- Their prompt taxonomy is Original/Formal/Simplified/Roleplay/Direct
  (5 styles, all register/tone changes). Ours swaps in **negation**
  instead of a fifth tone variant — a logic-inversion axis they don't
  test. Worth stating explicitly as a deliberate methodological
  difference, not an oversight.
- Model set deliberately updated past the reference study's exact
  lineup — see the note in Step 1 above. State this consistently
  across every document, not just here.