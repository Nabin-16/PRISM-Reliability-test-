"""
PRISM 

This module takes the frozen question samples produced by
src/load_datasets.py and renders the five frozen prompt templates stored in
data/templates/.

Input:
    data/processed/arc_challenge_sample.json
    data/processed/sciq_sample.json

Templates:
    data/templates/p0_minimal.txt
    data/templates/p1_direct.txt
    data/templates/p2_structured.txt
    data/templates/p3_role_based.txt
    data/templates/p4_careful_analysis.txt

Output:
    data/prompts/arc_challenge_prompts.json
    data/prompts/sciq_prompts.json

Once the templates and frozen question samples are finalized, these files won't be generated again so please select the
best 5 prompting style brothers...
A changed template or sample should be treated as a new experiment version....
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates"
PROMPT_DIR = PROJECT_ROOT / "data" / "prompts"

TEMPLATE_VERSION = "1.0"

TEMPLATE_FILES: dict[str, str] = {
    "P0": "p0_minimal.txt",
    "P1": "p1_direct.txt",
    "P2": "p2_structured.txt",
    "P3": "p3_role_based.txt",
    "P4": "p4_careful_analysis.txt",
}

REQUIRED_FIELDS = (
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
)

@dataclass(frozen=True)
class PromptTemplate:
    """A frozen prompt template and its identifying metadata."""

    prompt_id: str
    filename: str
    text: str
    sha256: str
def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact and fail with a useful message."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256(text: str) -> str:
    """Return the SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def load_templates() -> dict[str, PromptTemplate]:
    """Load and validate all five frozen prompt templates."""
    templates: dict[str, PromptTemplate] = {}

    for prompt_id, filename in TEMPLATE_FILES.items():
        path = TEMPLATE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template {prompt_id} is missing: {path}"
            )

        text = path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(f"Prompt template {prompt_id} is empty: {path}")

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if "{" + field + "}" not in text
        ]

        if missing_fields:
            raise ValueError(
                f"{path.name} is missing required placeholders: "
                f"{', '.join(missing_fields)}"
            )

        templates[prompt_id] = PromptTemplate(
            prompt_id=prompt_id,
            filename=filename,
            text=text,
            sha256=_sha256(text),
        )

    return templates

def _question_context(question: dict[str, Any]) -> dict[str, str]:
    """Convert a frozen question record into template placeholders."""
    required_keys = {"question", "options", "correct_answer"}

    missing = required_keys - question.keys()
    if missing:
        raise ValueError(
            f"Question {question.get('question_id', '<unknown>')} is missing: "
            f"{', '.join(sorted(missing))}"
        )

    options = question["options"]

    missing_options = [
        letter for letter in ("A", "B", "C", "D")
        if letter not in options
    ]

    if missing_options:
        raise ValueError(
            f"Question {question.get('question_id', '<unknown>')} is missing "
            f"options: {', '.join(missing_options)}"
        )

    return {
        "question": str(question["question"]).strip(),
        "option_a": str(options["A"]).strip(),
        "option_b": str(options["B"]).strip(),
        "option_c": str(options["C"]).strip(),
        "option_d": str(options["D"]).strip(),
    }

def render_prompt(
    question: dict[str, Any],
    template: PromptTemplate,
) -> str:
    """Render one prompt template for one frozen question."""
    context = _question_context(question)

    try:
        rendered = template.text.format(**context)
    except KeyError as exc:
        raise ValueError(
            f"Template {template.filename} contains an unknown placeholder: "
            f"{exc.args[0]!r}"
        ) from exc

    return rendered.strip() + "\n"


def generate_question_prompts(
    question: dict[str, Any],
    templates: dict[str, PromptTemplate],
) -> list[dict[str, Any]]:
    """Generate P0-P4 for one frozen question."""
    question_id = str(question["question_id"])

    generated: list[dict[str, Any]] = []

    for prompt_id, template in templates.items():
        generated.append(
            {
                "question_id": question_id,
                "prompt_id": prompt_id,
                "template_version": TEMPLATE_VERSION,
                "template_file": template.filename,
                "template_sha256": template.sha256,
                "prompt_text": render_prompt(question, template),
            }
        )

    return generated

def generate_dataset_prompts(
    sample_path: Path,
    templates: dict[str, PromptTemplate],
) -> dict[str, Any]:
    """Generate the complete prompt artifact for one frozen dataset sample."""
    sample = _load_json(sample_path)

    dataset_name = str(sample["dataset"])
    questions = sample["questions"]

    all_prompts: list[dict[str, Any]] = []

    for question in questions:
        question_prompts = generate_question_prompts(
            question,
            templates,
        )

        all_prompts.append(
            {
                "question_id": str(question["question_id"]),
                "correct_answer": question["correct_answer"],
                "prompts": question_prompts,
            }
        )

    return {
        "dataset": dataset_name,
        "template_version": TEMPLATE_VERSION,
        "sampling_seed": sample.get("sampling_seed"),
        "sample_size": len(questions),
        "prompt_conditions": list(TEMPLATE_FILES.keys()),
        "templates": [
            {
                "prompt_id": template.prompt_id,
                "filename": template.filename,
                "sha256": template.sha256,
            }
            for template in templates.values()
        ],
        "questions": all_prompts,
    }


def save_dataset_prompts(
    dataset_name: str,
    payload: dict[str, Any],
) -> Path:
    """Save a generated prompt artifact under data/prompts/."""
    output_path = PROMPT_DIR / f"{dataset_name}_prompts.json"
    _write_json(output_path, payload)
    return output_path

def main() -> None:
    """Render frozen templates against frozen ARC and SciQ samples."""
    print("Loading frozen prompt templates...")
    templates = load_templates()

    print("Generating ARC-Challenge prompts...")
    arc_payload = generate_dataset_prompts(
        PROCESSED_DIR / "arc_challenge_sample.json",
        templates,
    )

    print("Generating SciQ prompts...")
    sciq_payload = generate_dataset_prompts(
        PROCESSED_DIR / "sciq_sample.json",
        templates,
    )

    arc_path = save_dataset_prompts(
        "arc_challenge",
        arc_payload,
    )

    sciq_path = save_dataset_prompts(
        "sciq",
        sciq_payload,
    )

    print("\nGenerated prompt artifacts:")
    print(f"  ARC-Challenge: {arc_path}")
    print(f"  SciQ:          {sciq_path}")
    print(f"  Conditions:    {', '.join(TEMPLATE_FILES.keys())}")
    print(f"  Template ver.: {TEMPLATE_VERSION}")


if __name__ == "__main__":
    main()
