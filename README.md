# PRISM: Prompt Reliability Through Intelligent Semantic Multiplexing (Research Phase)

## Overview

PRISM investigates whether a Small Language Model (SLM) changes its behavior when the **same underlying question** is expressed through different prompt formulations.

The project is designed for a resource-constrained environment: local SLM inference through **Ollama** on a CPU-only machine.

The first research phase focuses on measuring:

- correctness;
- instruction adherence;
- cross-prompt consistency;
- prompt sensitivity;
- potentially unreliable response patterns.

PRISM does **not** assume that a consistent answer is necessarily a correct answer.

---

## Research Scope

### Domain

**Educational multiple-choice question answering**

The initial study intentionally does not claim cross-domain generalization.

### Datasets

- [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)
- [SciQ](https://huggingface.co/datasets/allenai/sciq)

Frozen sample:

- 200 ARC-Challenge questions
- 200 SciQ questions
- 400 questions total

The same frozen questions are used for every model and prompt condition.

### Models

| Model | Ollama tag |
|---|---|
| Llama-3.2 (3B) | `llama3.2:3b` |
| Gemma-3 (4B) | `gemma3:4b` |
| Phi-4-mini (3.8B) | `phi4-mini:latest` |
| Mistral-7B | `mistral:7b` |

---

## Research Questions

### RQ1: Prompt Sensitivity

How does SLM response behavior change when the same task is expressed using different controlled prompt formulations?

### RQ2: Model Differences

Do different SLMs exhibit different levels of sensitivity to prompt formulation?

### RQ3: Consistency and Correctness

Does agreement across prompt formulations correspond to correct model behavior?

### RQ4: Failure Patterns

What failure patterns occur when SLMs answer the same task under different prompt formulations?

### RQ5: Reliability Signal

Can cross-prompt agreement, together with correctness and instruction adherence, provide useful evidence for identifying potentially unreliable responses?

---

## Experiment 1

Experiment 1 uses five controlled prompt formulation conditions.

| ID | Condition | Main manipulation |
|---|---|---|
| **P0** | Minimal / Baseline | Minimal task presentation |
| **P1** | Direct Instruction | Explicit task instruction |
| **P2** | Structured Instruction | Explicit organization |
| **P3** | Role-Based Instruction | Neutral role framing |
| **P4** | Careful-Analysis Instruction | Explicit careful-analysis instruction |

### Experimental invariant

For every question, all five conditions preserve:

- the same question;
- the same answer options;
- the same information available to the model;
- the same task objective;
- the same expected answer space;
- the same evaluation target.

Only the **prompt formulation** changes.

### Output invariant

Every condition uses:

```text
Return only the letter of the correct option.
```

This prevents response-format instructions from becoming an unintended experimental variable.

### Deferred techniques

The following are not part of Experiment 1:

- few-shot prompting;
- chain-of-thought prompting;
- self-consistency;
- critique/revision;
- polarity/negation transformation;
- debate-style prompting;
- other advanced interventions.

These can be investigated later as separate experiments.

---

## Inference Configuration

Experiment 1 uses deterministic inference:

```text
temperature = 0
```

There is one inference for each:

```text
question × model × prompt condition
```

No repeated stochastic runs are used in the core experiment.

### Planned inference volume

```text
400 questions
× 4 models
× 5 prompt conditions
= 8,000 inference runs
```

### Generation limit

```text
num_predict = 450
```

This is a maximum generation limit, not a target response length. It leaves enough room for a model to produce a longer non-compliant answer instead of artificially truncating the response.

---

## Pipeline

```text
Benchmark datasets
        ↓
Fixed seeded sampling
        ↓
Frozen question sets
        ↓
Benchmark audit
        ↓
Frozen prompt templates
        ↓
Rendered prompt artifacts
        ↓
Ollama inference
        ↓
Raw responses
        ↓
Response parsing
        ↓
Prompt-level scoring
        ↓
Question-level consistency
        ↓
Prompt sensitivity analysis
        ↓
Model-level summaries
```

Inference and evaluation are intentionally separated because inference is the most expensive stage on the target hardware.

---

## Repository Structure

```text
reliability_study/
│
├── data/
│   ├── audit/
│   │   ├── benchmark_audit.jsonl
│   │   └── benchmark_audit.csv
│   │
│   ├── processed/
│   │   ├── arc_challenge_sample.json
│   │   └── sciq_sample.json
│   │
│   ├── templates/
│   │   ├── p0_minimal.txt
│   │   ├── p1_direct.txt
│   │   ├── p2_structured.txt
│   │   ├── p3_role_based.txt
│   │   └── p4_careful_analysis.txt
│   │
│   └── prompts/
│       ├── arc_challenge_prompts.json
│       └── sciq_prompts.json
│
├── results/
│   ├── raw_responses/
│   ├── parsed/
│   ├── scored/
│   └── summary/
│
├── src/
│   ├── __init__.py
│   ├── benchmark_audit.py
│   ├── consistency_scorer.py
│   ├── inference.py
│   ├── load_datasets.py
│   ├── prompt_variations.py
│   ├── response_parser.py
│   ├── summary_report.py
│   └── test_response_parser.py
│
├── config.py
├── README.md
└── requirements.txt
```

### Directory roles

**`data/processed/`** — frozen sampled benchmark questions.

**`data/audit/`** — human-reviewable benchmark quality audit.

**`data/templates/`** — exact frozen wording of P0–P4.

**`data/prompts/`** — rendered prompt artifacts.

**`results/raw_responses/`** — immutable model outputs and inference metadata.

**`results/parsed/`** — compact parser results.

**`results/scored/`** — correctness and prompt-level evaluation.

**`results/summary/`** — question-level and model-level analysis.

---

## Setup

Install the project dependencies from `requirements.txt`.

Make sure Ollama is installed and running.

Verify the local models:

```powershell
ollama list
```

Expected tags:

```text
llama3.2:3b
gemma3:4b
phi4-mini:latest
mistral:7b
```

---

## Running the Pipeline

### 1. Freeze the dataset samples

```powershell
python src/load_datasets.py
```

This:

1. loads ARC-Challenge and SciQ;
2. normalizes them into a common A/B/C/D structure;
3. selects 200 questions from each dataset using the fixed seed;
4. saves the frozen samples under `data/processed/`.

### 2. Generate the prompt artifacts

```powershell
python src/prompt_variations.py
```

This renders the five templates from `data/templates/` for every frozen question and writes the results to `data/prompts/`.

### 3. Prepare the benchmark audit

```powershell
python src/benchmark_audit.py
```

This creates:

```text
data/audit/benchmark_audit.jsonl
data/audit/benchmark_audit.csv
```

The audit documents potential ambiguity or answer-key concerns before the full experiment.

It does not automatically change official benchmark labels.

### 4. Test Ollama inference

Before the full run:

```powershell
python src/inference.py --dataset arc_challenge --model llama3.2:3b --max-questions 1
```

This performs:

```text
1 question × 5 prompt conditions = 5 requests
```

Raw outputs are saved under:

```text
results/raw_responses/
```

### 5. Parse responses

```powershell
python src/response_parser.py results/raw_responses/arc_challenge__llama3.2_3b.jsonl
```

The compact parser output is written to:

```text
results/parsed/
```

### 6. Score responses and calculate question metrics

```powershell
python src/consistency_scorer.py results/parsed/arc_challenge__llama3.2_3b.jsonl
```

Outputs are written to:

```text
results/scored/
results/summary/
```

### 7. Roll question-level metrics up into model-level summaries

```powershell
python src/summary_report.py
```

Reads every `*_question_metrics.jsonl` file under `results/summary/` and
writes two report-ready tables:

```text
results/summary/model_dataset_summary.csv     # one row per (model, dataset)
results/summary/model_prompt_summary.csv      # one row per (model, dataset, prompt_condition)
```

`model_dataset_summary.csv` includes `prompt_invariant_incorrect_rate` —
the proportion of questions where all five prompt conditions agreed
unanimously on the same *wrong* answer.

### Parser unit tests

```powershell
python src/test_response_parser.py
```

Checks parser behavior against known text shapes only — no test case
references a dataset's ground-truth answer, consistent with the parser
itself never seeing ground truth.

---

## Response Parsing

The parser separates answer extraction from instruction adherence.

### Clean answers

Examples:

```text
B
B.
(B)
B)
```

These are treated as clean, instruction-compliant answers.

### Recoverable but non-compliant answers

Examples:

```text
Answer: B
The correct answer is B.
The final answer is B.
The correct answer is B because ...
```

A response can therefore produce:

```text
parsed_answer = B
instruction_compliant = false
```

This is deliberate: a format violation should not erase a recoverable answer.

### UNKNOWN

`UNKNOWN` is used only when a unique answer cannot be safely extracted.

Examples:

```text
I don't know.
A or B could be correct.
The answer is unclear.
```

The parser is deliberately conservative and does not use the benchmark answer to infer what the model intended.

---

## Example

Suppose Llama produces:

```text
P0 → B
P1 → B
P2 → B
P3 → B
P4 → B
```

and the benchmark answer is:

```text
D
```

Then:

```text
agreement          = 1.0
prompt_sensitivity = 0.0
unanimous          = true
majority_answer    = B
majority_correct   = false
```

The important observation is:

> **Perfect cross-prompt consistency does not guarantee correctness.**

This is why PRISM reports consistency and correctness separately.

---

## Benchmark Audit

The benchmark audit is intended to identify:

- potentially ambiguous questions;
- possible answer-key concerns;
- questions requiring source verification;
- reviewer observations.

The audit does not automatically delete or relabel questions.

For the primary experiment, the official benchmark label remains the quantitative reference unless a documented methodological decision is made **before the experiment is frozen**.

---

## Data Lineage

PRISM maintains an explicit evidence chain:

```text
Benchmark data
     ↓
Frozen sample
     ↓
Prompt template
     ↓
Rendered prompt
     ↓
Raw model response
     ↓
Parsed result
     ↓
Scored result
     ↓
Question metrics
     ↓
Model-level analysis
```

The raw-response layer is treated as immutable evidence.

Derived files may be regenerated without rerunning model inference.

---

## Reproducibility Rules

Before full inference:

- freeze the question sample;
- freeze P0–P4;
- freeze model tags;
- freeze Ollama parameters;
- freeze parsing rules;
- freeze scoring rules;
- complete the benchmark audit;
- record experiment/protocol/template versions.

After full inference begins:

> **Do not change prompt wording because a model performs poorly.**

Any methodological change becomes a new experiment version.

---

## What Experiment 1 Does Not Claim

Experiment 1 does not claim:

- cross-domain generalization;
- that agreement guarantees truth;
- that PRISM automatically improves every SLM;
- that all benchmark labels are indisputable;
- that five prompt formulations cover the space of prompting strategies;
- that an advanced mitigation technique works before a separate experiment demonstrates it.

The first experiment is specifically a study of:

> **prompt formulation sensitivity and behavioral reliability of locally run SLMs on educational multiple-choice questions under deterministic inference.**

---

## From Research to Application

The research phase above is the empirical foundation for the PRISM desktop application. The `prism_core` engine originated directly as the backend of this research pipeline, and was subsequently adapted and extended into a reusable desktop benchmarking tool rather than remaining a one-off study script.

At a high level, the application implements the same pipeline described above end-to-end:

1. **`prism_core`** — evolved from the research pipeline's `src/` modules — loads a frozen dataset (ARC-Challenge / SciQ), renders the five prompt variants (P0–P4) for each question, and runs inference against a locally installed **Ollama** model.
2. Raw responses are parsed into structured answers, distinguishing clean answers, recoverable-but-non-compliant answers, and `UNKNOWN` responses — exactly as defined in [Response Parsing](#response-parsing).
3. Scoring computes correctness, cross-prompt agreement, prompt sensitivity, and prompt-invariant incorrectness, then aggregates results per model and dataset.
4. Results are stored locally (SQLite) and surfaced through a native **PySide6** desktop dashboard — KPI cards, per-question drill-down, model comparison, and one-click PDF report export — built on top of the backend to make the research metrics above explorable interactively instead of only as CSV output.
5. Benchmark runs can optionally sync to a shared results repository, letting verified baseline results (for `llama3.2:3b`, `gemma3:4b`, `phi4-mini:latest`, `mistral:7b`) be explored without re-running inference.

The application is intended as a **reliability-aware wrapper around SLM inference** — it surfaces the consistency/correctness distinction from this research, rather than claiming to make any model universally reliable.

**The desktop app can be downloaded from:** [PRISM-WEB](https://prism-slm-neon.vercel.app/)

---

## Research Principles

### Change the formulation, not the task

The underlying question stays fixed.

### Separate consistency from correctness

A model can be highly consistent and still be wrong.

### Preserve raw evidence

Every raw response is stored before parsing and scoring.

### Prefer evidence over assumptions

Potential mitigation strategies are tested rather than assumed to improve reliability.
