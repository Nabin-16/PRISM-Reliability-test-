"""
PRISM

Reads every results/summary/*_question_metrics.jsonl file (produced by
consistency_scorer.py) and rolls question-level metrics up into two
report-ready tables.

Outputs
-------
results/summary/model_dataset_summary.csv
    One row per (model, dataset): accuracy, mean prompt sensitivity,
    unanimous rate, usable response rate, and the rate of
    PROMPT-INVARIANT INCORRECTNESS (unanimous=True and majority_correct=
    False - all five prompt conditions agree, deterministically, on the
    same wrong answer; see spec section 14).

results/summary/model_prompt_summary.csv
    One row per (model, dataset, prompt_condition): per-condition
    accuracy and instruction-compliance rate, letting P0-P4 be compared
    directly against each other.

Usage:
    python src/summary_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_question_metrics_files() -> list[Path]:
    return sorted(config.RESULTS_SUMMARY_DIR.glob("*_question_metrics.jsonl"))


def build_model_dataset_summary(all_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_records:
        groups[(r["model"], r["dataset"])].append(r)

    rows = []
    for (model, dataset), records in sorted(groups.items()):
        n = len(records)
        accuracy = sum(r["majority_correct"] for r in records) / n
        mean_sensitivity = sum(r["prompt_sensitivity"] for r in records) / n
        mean_agreement = sum(r["agreement"] for r in records) / n
        unanimous_rate = sum(r["unanimous"] for r in records) / n
        mean_usable_rate = sum(r["usable_response_rate"] for r in records) / n
        prompt_invariant_incorrect = sum(
            1 for r in records if r["unanimous"] and not r["majority_correct"]
        ) / n

        rows.append({
            "model": model,
            "dataset": dataset,
            "n_questions": n,
            "accuracy": round(accuracy, 4),
            "mean_agreement": round(mean_agreement, 4),
            "mean_prompt_sensitivity": round(mean_sensitivity, 4),
            "unanimous_rate": round(unanimous_rate, 4),
            "prompt_invariant_incorrect_rate": round(prompt_invariant_incorrect, 4),
            "mean_usable_response_rate": round(mean_usable_rate, 4),
        })
    return rows


def build_model_prompt_summary(all_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    compliance_groups: dict[tuple[str, str, str], list[bool]] = defaultdict(list)

    for r in all_records:
        for prompt_id, correct in r["prompt_correctness"].items():
            key = (r["model"], r["dataset"], prompt_id)
            groups[key].append(correct)
        for prompt_id, compliant in r["prompt_compliance"].items():
            key = (r["model"], r["dataset"], prompt_id)
            compliance_groups[key].append(compliant)

    rows = []
    for key in sorted(groups.keys()):
        model, dataset, prompt_id = key
        correctness = groups[key]
        compliance = compliance_groups.get(key, [])
        n = len(correctness)
        rows.append({
            "model": model,
            "dataset": dataset,
            "prompt_condition": prompt_id,
            "n": n,
            "accuracy": round(sum(correctness) / n, 4) if n else 0.0,
            "instruction_compliant_rate": round(sum(compliance) / len(compliance), 4) if compliance else 0.0,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"  [skip] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def main() -> None:
    files = find_question_metrics_files()
    if not files:
        print("No *_question_metrics.jsonl files found in "
              f"{config.RESULTS_SUMMARY_DIR} — run consistency_scorer.py first.")
        return

    all_records: list[dict[str, Any]] = []
    for f in files:
        all_records.extend(load_jsonl(f))
    print(f"Loaded {len(all_records)} question-level records from {len(files)} file(s).")

    model_dataset_rows = build_model_dataset_summary(all_records)
    model_prompt_rows = build_model_prompt_summary(all_records)

    write_csv(config.RESULTS_SUMMARY_DIR / "model_dataset_summary.csv", model_dataset_rows)
    write_csv(config.RESULTS_SUMMARY_DIR / "model_prompt_summary.csv", model_prompt_rows)

    if model_dataset_rows:
        print("\nPrompt-invariant incorrectness rate (unanimous but wrong), by model/dataset:")
        for row in model_dataset_rows:
            flag = "  <-- non-zero" if row["prompt_invariant_incorrect_rate"] > 0 else ""
            print(f"  {row['model']:15s} {row['dataset']:15s} "
                  f"{row['prompt_invariant_incorrect_rate']:.4f}{flag}")


if __name__ == "__main__":
    main()
