"""
PRISM: dataset acquisition and fixed sampling.

Research scope:
    - ARC-Challenge: 200 questions
    - SciQ: 200 questions

The selected question IDs are sampled once using a fixed seed and written to
data/processed/. 

Every later experiment must reuse these frozen files rather
than sampling the datasets again.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

SEED = config.RANDOM_SEED
SAMPLE_SIZE = config.SAMPLE_SIZE_PER_DATASET
PROCESSED_DIR = config.DATA_PROCESSED_DIR

LETTERS = ("A", "B", "C", "D")

def _save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting for reproducible artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def load_arc_challenge() -> list[dict[str, Any]]:
    """Load and normalize ARC-Challenge test questions."""
    dataset = load_dataset(
    "allenai/ai2_arc",
    "ARC-Challenge",
    split="test",
)

    records: list[dict[str, Any]] = []

    for row in dataset:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer_key = str(row["answerKey"]).strip()

        if len(labels) != 4 or len(texts) != 4:
            continue

        try:
            correct_index = labels.index(answer_key)
        except ValueError as exc:
            raise ValueError(
                f"Unknown ARC answer key {answer_key!r} "
                f"for question {row.get('id')!r}"
            ) from exc

        records.append(
            {
                "dataset": "arc_challenge",
                "question_id": str(row["id"]),
                "question": str(row["question"]).strip(),
                "options": {
                    letter: str(text).strip()
                    for letter, text in zip(LETTERS, texts)
                },
                "correct_answer": LETTERS[correct_index],
            }
        )

    return records

def load_sciq() -> list[dict[str, Any]]:
    """Load and normalize SciQ test questions.

    SciQ provides one correct answer and three distractors rather than fixed
    A/B/C/D labels. We therefore shuffle the four options deterministically
    per question so that the correct answer is not always in position A.
    """
    dataset = load_dataset(
    "allenai/sciq",
    split="test",
)

    records: list[dict[str, Any]] = []

    for row in dataset:
        question_id = str(row.get("id", ""))

        # The correct answer is stored separately from the distractors.
        option_items = [
            str(row["correct_answer"]).strip(),
            str(row["distractor1"]).strip(),
            str(row["distractor2"]).strip(),
            str(row["distractor3"]).strip(),
        ]
        rng = random.Random(f"{SEED}:sciq:{question_id}")
        shuffled = option_items.copy()
        rng.shuffle(shuffled)

        correct_text = option_items[0]

        options = dict(zip(LETTERS, shuffled))
        correct_answer = next(
            letter
            for letter, text in options.items()
            if text == correct_text
        )

        records.append(
            {
                "dataset": "sciq",
                "question_id": question_id,
                "question": str(row["question"]).strip(),
                "options": options,
                "correct_answer": correct_answer,
            }
        )

    return records

def fixed_sample(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    sample_size: int = SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Take one reproducible random sample without replacement."""
    if len(records) < sample_size:
        raise ValueError(
            f"{dataset_name} contains only {len(records)} questions; "
            f"cannot sample {sample_size}."
        )

    rng = random.Random(f"{SEED}:{dataset_name}")
    selected = rng.sample(records, sample_size)
    selected.sort(key=lambda item: item["question_id"])

    return selected

def save_frozen_sample(
    dataset_name: str,
    questions: list[dict[str, Any]],
) -> Path:
    """Save a frozen research sample under data/processed/."""
    output = PROCESSED_DIR / f"{dataset_name}_sample.json"

    payload = {
        "dataset": dataset_name,
        "sample_size": len(questions),
        "sampling_seed": SEED,
        "source_split": "test",
        "question_ids": [q["question_id"] for q in questions],
        "questions": questions,
    }

    _save_json(output, payload)
    return output

def main() -> None:
    """Acquire both datasets and freeze the 200-question samples."""
    print("Loading ARC-Challenge...")
    arc_records = load_arc_challenge()

    print("Loading SciQ...")
    sciq_records = load_sciq()

    print(f"ARC-Challenge available: {len(arc_records)}")
    print(f"SciQ available:          {len(sciq_records)}")

    arc_sample = fixed_sample(
        arc_records,
        dataset_name="arc_challenge",
    )
    sciq_sample = fixed_sample(
        sciq_records,
        dataset_name="sciq",
    )

    arc_path = save_frozen_sample("arc_challenge", arc_sample)
    sciq_path = save_frozen_sample("sciq", sciq_sample)

    print("\nFrozen research samples created:")
    print(f"  ARC-Challenge: {arc_path}")
    print(f"  SciQ:          {sciq_path}")
    print(f"  Seed:           {SEED}")
    print(f"  Size/dataset:   {SAMPLE_SIZE}")


if __name__ == "__main__":
    main()
