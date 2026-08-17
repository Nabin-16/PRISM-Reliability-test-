"""
PRISM

These tests check parser BEHAVIOR against known input text shapes only. No test case references or
compares against a dataset's ground-truth answer, that would test the
scorer, not the parser, and the parser is explicitly required to never see
ground truth in the first place.

Run:
    python src/test_response_parser.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.response_parser1 import parse_response

CASES = [
    ("B", "B", True),
    ("B.", "B", True),
    ("(B)", "B", True),
    ("B)", "B", True),
    ("Answer: B", "B", False),
    ("The correct answer is B.", "B", False),
    ("The final answer is B.", "B", False),
    ("The correct answer is B because it directly explains the "
     "observation described in the question.", "B", False),
    ("A or B", "UNKNOWN", False),
    ("A and B", "UNKNOWN", False),
    ("I don't know", "UNKNOWN", False),
    ("There is insufficient information to determine the answer.",
     "UNKNOWN", False),
    ("Let's work through each option carefully. Option A relates to "
     "one interpretation, option C to another, and after considering "
     "the evidence the final answer is B.", "B", False),
    ("Let's think through this step by step.\nOption A seems plausible "
     "at first.\nOption C also has some support.\n\nD", "D", False),
    ("Working through the physics: power equals work over time.\n\n"
     "The final answer is: $\\boxed{C}$", "C", False),
    ("C", "C", True),
    ("Therefore the final answer is C. However, on reflection:\n\nD",
     "UNKNOWN", False),
    ("", "UNKNOWN", False),
    ("   \n  ", "UNKNOWN", False),
    (None, "UNKNOWN", False),
    ("This response contains no identifiable option at all.",
     "UNKNOWN", False),
]


def run():
    passed = 0
    failed = 0
    for raw, expected_answer, expected_compliant in CASES:
        result = parse_response(raw)
        ok_answer = result["parsed_answer"] == expected_answer
        ok_compliant = (
            expected_answer == "UNKNOWN"
            or result["instruction_compliant"] == expected_compliant
        )
        ok = ok_answer and ok_compliant
        passed += ok
        failed += not ok
        status = "PASS" if ok else "FAIL"
        preview = repr(raw)[:60] if raw is not None else "None"
        print(f"  [{status}] expected={expected_answer:8s} "
              f"got={result['parsed_answer']:8s} "
              f"status={result['parse_status']:22s} "
              f"| {preview}")

    print(f"\n{passed}/{passed + failed} test cases passed")
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
