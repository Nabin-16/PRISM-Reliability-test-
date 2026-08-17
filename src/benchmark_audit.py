"""
PRISM, benchmark item audit preparation......

Outputs
data/processed/benchmark_audit.jsonl
data/processed/benchmark_audit.csv

Review fields:
    audit_status
    ambiguity_flag
    answer_key_issue
    source_checked
    reviewer_notes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

PROCESSED_DIR = config.DATA_PROCESSED_DIR
AUDIT_DIR = config.DATA_AUDIT_DIR

SAMPLE_FILES = {
    "arc_challenge": PROCESSED_DIR / "arc_challenge_sample.json",
    "sciq": PROCESSED_DIR / "sciq_sample.json",
}

JSONL_OUTPUT = AUDIT_DIR / "benchmark_audit.jsonl"
CSV_OUTPUT = AUDIT_DIR / "benchmark_audit.csv"

OPTION_LETTERS = ("A", "B", "C", "D")


def load_sample(path: Path) -> dict[str, Any]:
    """Load one frozen sample artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen sample not found: {path}\n"
            "Run src/load_datasets.py first."
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def validate_question(question: dict[str, Any]) -> None:
    """Validate the minimal structure required for an audit record."""
    required = {"question_id", "question", "options", "correct_answer"}
    missing = required - question.keys()

    if missing:
        raise ValueError(
            f"Question {question.get('question_id', '<unknown>')} is missing: "
            f"{', '.join(sorted(missing))}"
        )

    options = question["options"]

    missing_options = [
        letter for letter in OPTION_LETTERS
        if letter not in options
    ]

    if missing_options:
        raise ValueError(
            f"Question {question['question_id']} is missing options: "
            f"{', '.join(missing_options)}"
        )

    if question["correct_answer"] not in OPTION_LETTERS:
        raise ValueError(
            f"Invalid official answer {question['correct_answer']!r} "
            f"for {question['question_id']}"
        )

def build_audit_record(
    dataset_name: str,
    question: dict[str, Any],
) -> dict[str, Any]:
    """Create one pending human-audit record."""
    validate_question(question)

    options = question["options"]

    return {
        "dataset": dataset_name,
        "question_id": str(question["question_id"]),
        "question": str(question["question"]),
        "option_A": str(options["A"]),
        "option_B": str(options["B"]),
        "option_C": str(options["C"]),
        "option_D": str(options["D"]),
        "official_answer": str(question["correct_answer"]).upper(),
        "audit_status": "pending",
        "ambiguity_flag": "",
        "answer_key_issue": "",
        "source_checked": "",
        "reviewer_notes": "",
    }


def load_all_frozen_questions() -> list[dict[str, Any]]:
    """Load both frozen datasets in deterministic order."""
    records: list[dict[str, Any]] = []

    for dataset_name in ("arc_challenge", "sciq"):
        sample = load_sample(SAMPLE_FILES[dataset_name])

        for question in sample["questions"]:
            records.append(
                build_audit_record(
                    dataset_name,
                    question,
                )
            )

    return records

def write_jsonl(records: list[dict[str, Any]]) -> None:
    """Write the machine-readable audit artifact."""
    JSONL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with JSONL_OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def write_csv(records: list[dict[str, Any]]) -> None:
    """Write a human-reviewable spreadsheet-compatible audit artifact."""
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = list(records[0].keys())

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a human-reviewable benchmark audit artifact "
            "from the frozen PRISM samples."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing audit artifacts. "
            "Without this flag, existing audit files are preserved."
        ),
    )

    args = parser.parse_args()

    if not args.force and (JSONL_OUTPUT.exists() or CSV_OUTPUT.exists()):
        raise FileExistsError(
            "Benchmark audit files already exist.\n"
            "Delete them or use --force if you intentionally want to "
            "regenerate the pending audit."
        )

    records = load_all_frozen_questions()

    write_jsonl(records)
    write_csv(records)

    arc_count = sum(
        1 for record in records
        if record["dataset"] == "arc_challenge"
    )
    sciq_count = sum(
        1 for record in records
        if record["dataset"] == "sciq"
    )

    print("Benchmark audit artifacts created.")
    print(f"ARC-Challenge items: {arc_count}")
    print(f"SciQ items:          {sciq_count}")
    print(f"Total items:         {len(records)}")
    print(f"JSONL:               {JSONL_OUTPUT}")
    print(f"CSV:                 {CSV_OUTPUT}")

    print("\nImportant:")
    print(
        "Review the 400 frozen items before full inference. "
        "Do not change the official benchmark answers automatically."
    )


if __name__ == "__main__":
    main()