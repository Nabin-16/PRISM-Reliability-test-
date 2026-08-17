from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# don't change anything it took about 6hrs to build this plsssss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

LETTERS = frozenset({"A", "B", "C", "D"})

CLEAN = re.compile(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;]?\s*$", re.I)

# black bird
FINAL_PATTERNS = (
    re.compile(r"\b(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:final\s+)?answer\s+(?:is|would\s+be)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b", re.I),
    re.compile(r"\bfinal\s+answer\s*(?:is|would\s+be)?\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b", re.I),
)

STANDALONE_LINE_PATTERN = re.compile(r"(?m)^\s*([ABCD])\s*$")
BOXED_PATTERN = re.compile(r"\\boxed\{\s*([ABCD])\s*\}")

EXPLICIT_PATTERNS = (
    re.compile(r"\b(?:the\s+)?(?:correct|selected|chosen)\s+(?:answer|option)\s+(?:is|would\s+be|should\s+be)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b", re.I),
    re.compile(r"\banswer\s*[:\-]\s*[\(\[]?\s*([ABCD])\b", re.I),
)

# goomi's trend
REFUSAL_PATTERNS = (
    re.compile(r"\bi\s+don't\s+know\b", re.I),
    re.compile(r"\bi\s+(?:cannot|can't)\s+(?:determine|answer)\b", re.I),
    re.compile(r"\b(?:insufficient|not\s+enough)\s+information\b", re.I),
    re.compile(r"\bunable\s+to\s+(?:determine|answer)\b", re.I),
)
#...
AMBIGUITY_PATTERNS = (
    re.compile(r"\b(?:A|B|C|D)\s+or\s+(?:A|B|C|D)\b", re.I),
    re.compile(r"\b(?:A|B|C|D)\s+and\s+(?:A|B|C|D)\b", re.I),
)

def normalize_response(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

def unique_letters(values):
    seen = set()
    result = []
    for value in values:
        letter = value.upper()
        if letter in LETTERS and letter not in seen:
            seen.add(letter)
            result.append(letter)
    return result

def result(parsed_answer, parse_status, instruction_compliant):
    return {
        "parsed_answer": parsed_answer,
        "parse_status": parse_status,
        "instruction_compliant": instruction_compliant,
    }

def parse_response(raw_response: str | None):
    text = normalize_response(raw_response)

    if not text:
        return result("UNKNOWN", "empty", False)

    match = CLEAN.fullmatch(text)
    if match:
        return result(match.group(1).upper(), "clean_letter", True)

    final_letters = []
    for pattern in FINAL_PATTERNS:
        final_letters.extend(m.group(1) for m in pattern.finditer(text))
    final_letters.extend(m.group(1) for m in BOXED_PATTERN.finditer(text))
    final_letters.extend(m.group(1) for m in STANDALONE_LINE_PATTERN.finditer(text))
    final_letters = unique_letters(final_letters)

    if len(final_letters) == 1:
        return result(final_letters[0], "explicit_final_answer", False)
    if len(final_letters) > 1:
        return result("UNKNOWN", "ambiguous_final_answer", False)

    explicit_letters = []
    for pattern in EXPLICIT_PATTERNS:
        explicit_letters.extend(m.group(1) for m in pattern.finditer(text))
    explicit_letters = unique_letters(explicit_letters)

    if len(explicit_letters) == 1:
        return result(explicit_letters[0], "explicit_answer", False)
    if len(explicit_letters) > 1:
        return result("UNKNOWN", "ambiguous_answer", False)

    if any(pattern.search(text) for pattern in REFUSAL_PATTERNS):
        return result("UNKNOWN", "refusal_or_uncertainty", False)

    if any(pattern.search(text) for pattern in AMBIGUITY_PATTERNS):
        return result("UNKNOWN", "ambiguous_response", False)

    return result("UNKNOWN", "unparseable", False)

def build_parsed_record(record):
    parsed = parse_response(record.get("raw_response"))
    return {
        "experiment_id": record.get("experiment_id"),
        "protocol_version": record.get("protocol_version"),
        "dataset": record.get("dataset"),
        "question_id": record.get("question_id"),
        "model": record.get("model"),
        "prompt_id": record.get("prompt_id"),
        **parsed,
    }

def parse_jsonl_file(input_path: Path, output_path: Path):
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("Parsed output cannot overwrite raw responses.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    malformed = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                print(f"Warning: skipping malformed line {line_number}")
                continue

            target.write(json.dumps(build_parsed_record(record), ensure_ascii=False) + "\n")
            processed += 1

    return processed, malformed

def main():
    parser = argparse.ArgumentParser(description="Parse PRISM raw responses.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = (
        args.output.resolve()
        if args.output is not None
        else (config.RESULTS_PARSED_DIR / args.input.name).resolve()
    )

    processed, malformed = parse_jsonl_file(args.input, output)
    print(f"Processed records: {processed}")
    print(f"Malformed lines:    {malformed}")
    print(f"Input:              {args.input.resolve()}")
    print(f"Output:             {output}")

if __name__ == "__main__":
    main()
