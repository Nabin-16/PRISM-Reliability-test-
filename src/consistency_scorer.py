"""
PRISM

Input:
    results/parsed/*.jsonl

Outputs:
    results/scored/<same filename>
    results/summary/<same stem>_question_metrics.jsonl

This module is the first stage that uses the dataset ground-truth answer.

Definitions
-----------
For each prompt-level response:
    correct = parsed_answer == expected_answer

UNKNOWN responses are not treated as answer choices.

For each question/model combination:
    valid_response_count = number of parsed A/B/C/D responses
    usable_response_rate = valid_response_count / number of prompt conditions

    majority_answer = most frequent valid answer
    agreement = majority_count / valid_response_count

    prompt_sensitivity = 1 - agreement

    unanimous = True when all five prompt responses are the same valid answer

    majority_correct = majority_answer == expected_answer
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


VALID_ANSWERS = frozenset({"A", "B", "C", "D"})
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all valid JSON objects from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_number} in {path}"
                ) from exc

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Add correctness to one parsed record.

    Ground truth is used here for the first time.
    """
    expected = str(record.get("expected_answer", "")).upper()
    parsed = str(record.get("parsed_answer", "")).upper()
    if expected not in VALID_ANSWERS:
        raise ValueError(
            f"Invalid expected answer {expected!r} for "
            f"{record.get('question_id')!r}"
        )

    usable = parsed in VALID_ANSWERS

    result = dict(record)
    result["usable"] = usable
    result["correct"] = bool(usable and parsed == expected)

    return result

def join_raw_and_parsed(
    raw_records: list[dict[str, Any]],
    parsed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Join each compact parsed record with the matching raw record to recover
    the expected answer for scoring.

    The join key is:
        dataset + question_id + model + prompt_id
    """
    raw_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for raw in raw_records:
        key = (
            str(raw.get("dataset")),
            str(raw.get("question_id")),
            str(raw.get("model")),
            str(raw.get("prompt_id")),
        )

        if key in raw_index:
            raise ValueError(f"Duplicate raw-response key: {key}")

        raw_index[key] = raw

    scored: list[dict[str, Any]] = []

    for parsed in parsed_records:
        key = (
            str(parsed.get("dataset")),
            str(parsed.get("question_id")),
            str(parsed.get("model")),
            str(parsed.get("prompt_id")),
        )

        raw = raw_index.get(key)

        if raw is None:
            raise ValueError(
                f"No raw response found for parsed record: {key}"
            )

        combined = {
            **parsed,
            "expected_answer": str(raw["expected_answer"]).upper(),
        }

        scored.append(score_record(combined))

    return scored

def calculate_question_metrics(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Aggregate five prompt-level records into one question/model record.

    UNKNOWN responses are excluded from the answer-vote denominator, but their
    presence is reflected through usable_response_rate and unknown_count.
    """
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}

    for record in records:
        key = (
            str(record["dataset"]),
            str(record["question_id"]),
            str(record["model"]),
        )
        groups.setdefault(key, []).append(record)

    metrics: list[dict[str, Any]] = []

    for (dataset, question_id, model), group in sorted(groups.items()):
        # Keep a deterministic prompt ordering.
        group.sort(
            key=lambda item: (
                list(config.PROMPT_CONDITIONS).index(
                    item["prompt_id"]
                )
                if item["prompt_id"] in config.PROMPT_CONDITIONS
                else 999
            )
        )

        expected_answers = {
            str(record["expected_answer"]).upper()
            for record in group
        }

        if len(expected_answers) != 1:
            raise ValueError(
                f"Inconsistent ground truth for "
                f"{dataset}/{question_id}/{model}: {expected_answers}"
            )

        expected_answer = next(iter(expected_answers))

        responses: dict[str, str] = {}
        prompt_correctness: dict[str, bool] = {}
        prompt_compliance: dict[str, bool] = {}

        valid_answers: list[str] = []

        for record in group:
            prompt_id = str(record["prompt_id"])
            parsed = str(record["parsed_answer"]).upper()

            responses[prompt_id] = parsed
            prompt_correctness[prompt_id] = bool(record["correct"])
            prompt_compliance[prompt_id] = bool(
                record["instruction_compliant"]
            )

            if parsed in VALID_ANSWERS:
                valid_answers.append(parsed)

        total_prompts = len(config.PROMPT_CONDITIONS)
        valid_count = len(valid_answers)
        unknown_count = total_prompts - valid_count

        if valid_count == 0:
            majority_answer = "UNKNOWN"
            majority_count = 0
            agreement = 0.0
            prompt_sensitivity = 1.0
            unanimous = False
            majority_correct = False
        else:
            counts = Counter(valid_answers)
            # Deterministic tie-break: alphabetical order.
            majority_count = max(
                counts.values()
            )
            tied = sorted(
                answer
                for answer, count in counts.items()
                if count == majority_count
            )
            majority_answer = tied[0]

            agreement = majority_count / valid_count
            prompt_sensitivity = 1.0 - agreement

            unanimous = (
                valid_count == total_prompts
                and len(counts) == 1
            )

            majority_correct = (
                majority_answer == expected_answer
            )

        metrics.append(
            {
                "experiment_id": group[0].get("experiment_id"),
                "protocol_version": group[0].get("protocol_version"),
                "dataset": dataset,
                "question_id": question_id,
                "model": model,
                "expected_answer": expected_answer,
                "responses": responses,
                "prompt_correctness": prompt_correctness,
                "prompt_compliance": prompt_compliance,
                "valid_response_count": valid_count,
                "unknown_count": unknown_count,
                "usable_response_rate": (
                    valid_count / total_prompts
                    if total_prompts
                    else 0.0
                ),
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "agreement": round(agreement, 6),
                "prompt_sensitivity": round(
                    prompt_sensitivity,
                    6,
                ),
                "unanimous": unanimous,
                "majority_correct": majority_correct,
            }
        )

    return metrics

def score_parsed_file(parsed_path: Path) -> tuple[Path, Path]:
    """
    Score one parsed file.

    The corresponding raw file must exist under results/raw_responses/.
    """
    parsed_records = load_jsonl(parsed_path)

    raw_path = config.RESULTS_RAW_DIR / parsed_path.name
    raw_records = load_jsonl(raw_path)

    scored_records = join_raw_and_parsed(
        raw_records,
        parsed_records,
    )

    scored_path = config.RESULTS_SCORED_DIR / parsed_path.name
    write_jsonl(scored_path, scored_records)

    question_metrics = calculate_question_metrics(
        scored_records
    )

    summary_name = (
        f"{parsed_path.stem}_question_metrics.jsonl"
    )
    summary_path = config.RESULTS_SUMMARY_DIR / summary_name

    write_jsonl(
        summary_path,
        question_metrics,
    )

    return scored_path, summary_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score PRISM parsed responses and calculate question metrics."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Parsed JSONL file under results/parsed/.",
    )

    args = parser.parse_args()

    parsed_path = args.input.resolve()

    scored_path, summary_path = score_parsed_file(
        parsed_path
    )

    print(f"Scored output:    {scored_path}")
    print(f"Question metrics: {summary_path}")


if __name__ == "__main__":
    main()
