"""
PRISM

This module performs ONLY model inference and raw-response persistence.

Experiment 1:
    200 ARC-Challenge questions
    200 SciQ questions
    4 SLMs
    5 prompt conditions
    temperature = 0

Expected total:
    8,000 requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def prompt_hash(prompt_text: str) -> str:
    """Return a short stable identifier for the exact prompt sent."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def load_prompt_file(dataset_name: str) -> dict[str, Any]:
    """Load the frozen rendered prompts for one dataset."""
    path = (
        config.DATA_PROMPTS_DIR
        / f"{dataset_name}_prompts.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt artifact not found: {path}\n"
            "Run src/prompt_variations.py first."
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def validate_prompt_artifact(data: dict[str, Any], dataset_name: str) -> None:
    """Validate that a prompt artifact matches the research protocol."""
    expected_conditions = list(config.PROMPT_CONDITIONS)

    actual_conditions = data.get("prompt_conditions", [])
    if actual_conditions != expected_conditions:
        raise ValueError(
            f"{dataset_name}: prompt conditions do not match config.\n"
            f"Expected: {expected_conditions}\n"
            f"Found:    {actual_conditions}"
        )

    if data.get("template_version") != config.TEMPLATE_VERSION:
        raise ValueError(
            f"{dataset_name}: template version mismatch. "
            f"Expected {config.TEMPLATE_VERSION!r}, "
            f"found {data.get('template_version')!r}."
        )


def output_path(dataset_name: str, model_name: str) -> Path:
    """Return the JSONL raw-response file for one dataset/model pair."""
    safe_model = model_name.replace(":", "_").replace("/", "_")
    return (
        config.RESULTS_RAW_DIR
        / f"{dataset_name}__{safe_model}.jsonl"
    )


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    """
    Load completed (question_id, prompt_id) pairs.

    JSONL is used because inference can be interrupted after hundreds of
    requests. Completed lines remain usable and the experiment can resume
    without rerunning successful requests.
    """
    completed: set[tuple[str, str]] = set()

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A truncated final line should not invalidate all previous
                # completed results. The next run will safely skip it.(if anyone's laptop crashes tralalalalalala)
                print(
                    f"Warning: ignoring malformed JSONL line "
                    f"{line_number} in {path}"
                )
                continue

            question_id = record.get("question_id")
            prompt_id = record.get("prompt_id")

            if question_id is not None and prompt_id is not None:
                if record.get("status") == "success":
                    completed.add((str(question_id), str(prompt_id)))

    return completed


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as one durable JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        handle.flush() 

def generate_with_ollama(
    *,
    model_name: str,
    prompt_text: str,
) -> tuple[str, float]:
    """
    Send one non-streaming request to Ollama.

    Returns:
        (raw_response_text, latency_seconds)

    Raises:
        requests.RequestException: for HTTP/network errors.
        RuntimeError: for invalid Ollama responses.
    """
    started = time.perf_counter()

    response = requests.post(
        config.OLLAMA_URL,
        json={
            "model": model_name,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.NUM_PREDICT,
            },
        },
        timeout=config.REQUEST_TIMEOUT,
    )

    latency = time.perf_counter() - started
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned a response that was not valid JSON."
        ) from exc

    raw_response = payload.get("response")

    if raw_response is None:
        raise RuntimeError(
            f"Ollama response does not contain a 'response' field: "
            f"{payload}"
        )

    return str(raw_response), latency

def build_success_record(
    *,
    experiment_id: str,
    dataset_name: str,
    model_name: str,
    question: dict[str, Any],
    prompt: dict[str, Any],
    raw_response: str,
    latency_seconds: float,
) -> dict[str, Any]:
    """Create the immutable raw-response record for a successful request."""
    prompt_text = str(prompt["prompt_text"])

    return {
        "experiment_id": experiment_id,
        "protocol_version": config.PROTOCOL_VERSION,
        "dataset": dataset_name,
        "question_id": str(question["question_id"]),
        "model": model_name,
        "model_label": config.MODELS[model_name]["label"],
        "prompt_id": str(prompt["prompt_id"]),
        "template_version": prompt.get("template_version"),
        "template_sha256": prompt.get("template_sha256"),
        "prompt_sha256": prompt_hash(prompt_text),
        "prompt_text": prompt_text,
        "expected_answer": question["correct_answer"],
        "raw_response": raw_response,
        "latency_seconds": round(latency_seconds, 4),
        "temperature": config.TEMPERATURE,
        "num_predict": config.NUM_PREDICT,
        "timestamp_utc": utc_timestamp(),
        "status": "success",
    }


def build_error_record(
    *,
    experiment_id: str,
    dataset_name: str,
    model_name: str,
    question: dict[str, Any],
    prompt: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Create an auditable record for an inference failure."""
    prompt_text = str(prompt["prompt_text"])

    return {
        "experiment_id": experiment_id,
        "protocol_version": config.PROTOCOL_VERSION,
        "dataset": dataset_name,
        "question_id": str(question["question_id"]),
        "model": model_name,
        "model_label": config.MODELS[model_name]["label"],
        "prompt_id": str(prompt["prompt_id"]),
        "template_version": prompt.get("template_version"),
        "template_sha256": prompt.get("template_sha256"),
        "prompt_sha256": prompt_hash(prompt_text),
        "prompt_text": prompt_text,
        "expected_answer": question["correct_answer"],
        "raw_response": None,
        "latency_seconds": None,
        "temperature": config.TEMPERATURE,
        "num_predict": config.NUM_PREDICT,
        "timestamp_utc": utc_timestamp(),
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

def run_dataset_model(
    *,
    dataset_name: str,
    model_name: str,
    max_questions: int | None = None,
    overwrite: bool = False,
) -> None:
    """Run every prompt condition for one dataset/model pair."""
    prompt_data = load_prompt_file(dataset_name)
    validate_prompt_artifact(prompt_data, dataset_name)

    questions = prompt_data.get("questions", [])

    if max_questions is not None:
        questions = questions[:max_questions]

    raw_path = output_path(dataset_name, model_name)

    completed = set()
    if not overwrite:
        completed = load_completed_keys(raw_path)

    total = len(questions) * len(config.PROMPT_CONDITIONS)
    completed_count = sum(
        1
        for question in questions
        for prompt in question["prompts"]
        if (
            str(question["question_id"]),
            str(prompt["prompt_id"]),
        ) in completed
    )

    print(
        f"\n[{dataset_name}] {config.MODELS[model_name]['label']}"
    )
    print(f"Output: {raw_path}")
    print(f"Requests in selected run: {total}")
    print(f"Already completed: {completed_count}")

    request_number = completed_count

    for question in questions:
        question_id = str(question["question_id"])

        for prompt in question["prompts"]:
            prompt_id = str(prompt["prompt_id"])
            key = (question_id, prompt_id)

            if not overwrite and key in completed:
                continue

            request_number += 1
            print(
                f"[{request_number}/{total}] "
                f"{question_id} | {prompt_id}",
                end=" ... ",
                flush=True,
            )

            try:
                raw_response, latency = generate_with_ollama(
                    model_name=model_name,
                    prompt_text=str(prompt["prompt_text"]),
                )

                record = build_success_record(
                    experiment_id=config.EXPERIMENT_ID,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    question=question,
                    prompt=prompt,
                    raw_response=raw_response,
                    latency_seconds=latency,
                )

                append_record(raw_path, record)

                print(f"OK ({latency:.2f}s)")

            except Exception as exc:
                error_record = build_error_record(
                    experiment_id=config.EXPERIMENT_ID,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    question=question,
                    prompt=prompt,
                    error=exc,
                )

                append_record(raw_path, error_record)

                print(f"ERROR: {exc}")
                continue


def run_experiment(
    *,
    datasets: list[str],
    models: list[str],
    max_questions: int | None = None,
    overwrite: bool = False,
) -> None:
    """Run the requested dataset/model combinations."""
    config.RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        if dataset_name not in config.DATASETS:
            raise ValueError(
                f"Unknown dataset {dataset_name!r}. "
                f"Available: {list(config.DATASETS)}"
            )

    for model_name in models:
        if model_name not in config.MODELS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Available: {list(config.MODELS)}"
            )

    for dataset_name in datasets:
        for model_name in models:
            run_dataset_model(
                dataset_name=dataset_name,
                model_name=model_name,
                max_questions=max_questions,
                overwrite=overwrite,
            )

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PRISM Experiment 1 inference through Ollama."
    )

    parser.add_argument(
        "--dataset",
        choices=["arc_challenge", "sciq", "all"],
        default="all",
        help="Dataset to run. Default: all.",
    )

    parser.add_argument(
        "--model",
        choices=[*config.MODELS.keys(), "all"],
        default="all",
        help="Model to run. Default: all.",
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Optional limit for testing. "
            "Use a small value before the full experiment."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed requests instead of resuming.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    datasets = (
        list(config.DATASETS.keys())
        if args.dataset == "all"
        else [args.dataset]
    )

    models = (
        list(config.MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    print("PRISM Experiment 1")
    print("------------------")
    print(f"Experiment ID: {config.EXPERIMENT_ID}")
    print(f"Temperature:   {config.TEMPERATURE}")
    print(f"Num predict:   {config.NUM_PREDICT}")
    print(f"Datasets:      {', '.join(datasets)}")
    print(f"Models:        {', '.join(models)}")

    if args.max_questions is not None:
        print(f"Question limit: {args.max_questions}")

    run_experiment(
        datasets=datasets,
        models=models,
        max_questions=args.max_questions,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
