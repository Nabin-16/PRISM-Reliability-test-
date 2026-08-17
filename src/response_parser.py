from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

LETTERS = frozenset({"A", "B", "C", "D"})

# one letter
_CLEAN_LETTER_RE = re.compile(
    r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;]?\s*$",
    re.IGNORECASE,
)

# Explicit
_EXPLICIT_PATTERNS = (
    re.compile(
        r"\b(?:the\s+)?(?:correct|selected|chosen)\s+(?:answer|option)\s+"
        r"(?:is|would\s+be|should\s+be)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfinal\s+answer\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\banswer\s*[:\-]\s*[\(\[]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
)

# goomi's trend
_REFUSAL_PATTERNS = (
    re.compile(r"\bi\s+don't\s+know\b", re.IGNORECASE),
    re.compile(r"\bi\s+cannot\s+(?:determine|answer)\b", re.IGNORECASE),
    re.compile(r"\bi\s+can't\s+(?:determine|answer)\b", re.IGNORECASE),
    re.compile(r"\b(?:insufficient|not\s+enough)\s+information\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:determine|answer)\b", re.IGNORECASE),
)

# ...
_AMBIGUITY_PATTERNS = (
    re.compile(r"\b(?:A|B|C|D)\s+or\s+(?:A|B|C|D)\b", re.IGNORECASE),
    re.compile(r"\b(?:A|B|C|D)\s+and\s+(?:A|B|C|D)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:answers?|options?)\s+(?:are|could\s+be)\s*"
        r"[\:\-]?\s*(?:[ABCD](?:\s*[,/&]\s*[ABCD])+)",
        re.IGNORECASE,
    ),
)

# black bird
_FINAL_ANSWER_PATTERNS = (
    re.compile(
        r"\b(?:therefore|thus|hence|so)[,\s]+"
        r"(?:the\s+)?(?:final\s+)?answer\s+"
        r"(?:is|would\s+be)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfinal\s+answer\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
)
# horaa
_COMPLIANCE_RE = re.compile(
    r"^\s*[\(\[]?\s*[ABCD]\s*[\)\].,:;]?\s*$",
    re.IGNORECASE,
)

def _result(
    *,
    parsed_answer: str,
    parse_status: str,
    instruction_compliant: bool,
) -> dict[str, Any]:
    """Build a uniform parser result."""
    return {
        "parsed_answer": parsed_answer,
        "parse_status": parse_status,
        "instruction_compliant": instruction_compliant,
    }

def normalize_response(text: str | None) -> str:
    """Normalize whitespace without changing substantive response content."""
    if text is None:
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _unique_letters(values: Iterable[str]) -> list[str]:
    """Return unique uppercase option letters preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        letter = value.upper()
        if letter in LETTERS and letter not in seen:
            seen.add(letter)
            result.append(letter)
    return result

def parse_response(raw_response: str | None) -> dict[str, Any]:
    text = normalize_response(raw_response)

    if not text:
        return _result(
            parsed_answer="UNKNOWN",
            parse_status="empty",
            instruction_compliant=False,
        )
    
    # Clean output
    clean_match = _CLEAN_LETTER_RE.fullmatch(text)
    if clean_match:
        return _result(
            parsed_answer=clean_match.group(1).upper(),
            parse_status="clean_letter",
            instruction_compliant=True,
        )

    # Explicit declarations
    final_candidates: list[str] = []

    for pattern in _FINAL_ANSWER_PATTERNS:
        final_candidates.extend(match.group(1) for match in pattern.finditer(text))

    final_letters = _unique_letters(final_candidates)

    if len(final_letters) == 1:
        return _result(
            parsed_answer=final_letters[0],
            parse_status="explicit_final_answer",
            instruction_compliant=False,
        )

    if len(final_letters) > 1:
        return _result(
            parsed_answer="UNKNOWN",
            parse_status="ambiguous_final_answer",
            instruction_compliant=False,
        )
    # Generic explicit answer declarations
    explicit_candidates: list[str] = []

    for pattern in _EXPLICIT_PATTERNS:
        explicit_candidates.extend(
            match.group(1) for match in pattern.finditer(text)
        )

    explicit_letters = _unique_letters(explicit_candidates)

    if len(explicit_letters) == 1:
        return _result(
            parsed_answer=explicit_letters[0],
            parse_status="explicit_answer",
            instruction_compliant=False,
        )

    if len(explicit_letters) > 1:
        return _result(
            parsed_answer="UNKNOWN",
            parse_status="ambiguous_answer",
            instruction_compliant=False,
        )

    # Refusal
    if any(pattern.search(text) for pattern in _REFUSAL_PATTERNS):
        return _result(
            parsed_answer="UNKNOWN",
            parse_status="refusal_or_uncertainty",
            instruction_compliant=False,
        )

    # Multiple candidate / ambiguous responses
    if any(pattern.search(text) for pattern in _AMBIGUITY_PATTERNS):
        return _result(
            parsed_answer="UNKNOWN",
            parse_status="ambiguous_response",
            instruction_compliant=False,
        )

    # No safe extraction
    return _result(
        parsed_answer="UNKNOWN",
        parse_status="unparseable",
        instruction_compliant=False,
    )

def parse_record(record: dict[str, Any]) -> dict[str, Any]:
    """Parse a raw-response record while preserving its original fields."""
    result = dict(record)

    parsed = parse_response(record.get("raw_response"))

    result.update(parsed)
    return result


def parse_jsonl_file(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    Parse a JSONL raw-response file.

    Returns:
        (records_processed, records_failed_to_decode)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    malformed = 0

    with (
        input_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                print(
                    f"Warning: skipping malformed JSONL line "
                    f"{line_number} in {input_path}"
                )
                continue

            parsed_record = parse_record(record)

            target.write(
                json.dumps(parsed_record, ensure_ascii=False) + "\n"
            )

            processed += 1

    return processed, malformed


def main() -> None:
    """CLI entry point for parsing one raw-response JSONL file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse PRISM raw Ollama responses."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Raw-response JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Scored/parsed JSONL output path. "
            "If omitted, writes beside the input with '_parsed' suffix."
        ),
    )

    args = parser.parse_args()

    output = (
            args.output.resolve()
            if args.output is not None
            else (config.RESULTS_PARSED_DIR / args.input.name).resolve()
        )
    processed, malformed = parse_jsonl_file(
        args.input,
        output,
    )

    print(f"Processed records: {processed}")
    print(f"Malformed lines:    {malformed}")
    print(f"Output:             {output}")

if __name__ == "__main__":
    main()
