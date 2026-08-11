"""
Run this after a small pilot batch (10-15 questions) with a new model,
before committing to the full run.

Usage:
    python src/audit_unknowns.py results/raw_responses/gemma2_2b_education.json
"""

import json
import sys


def audit(path: str):
    with open(path) as f:
        results = json.load(f)

    unknown_count = 0
    total_calls = 0

    for q in results:
        for style, r in q["responses"].items():
            total_calls += 1
            if r["parsed"] != "UNKNOWN":
                continue
            unknown_count += 1
            print(f"\n{'─' * 70}")
            print(f"question: {q['question_id']}  |  style: {style}")
            print(f"{'─' * 70}")
            raw = r["raw"]
            if len(raw) <= 600:
                print(raw)
            else:
                # show head + tail, not just head — the conclusion (or
                # lack of one) that actually determines whether this
                # UNKNOWN is legitimate is almost always at the END,
                # and truncating from the start alone was silently
                # hiding it
                omitted = len(raw) - 600
                print(raw[:300])
                print(f"\n... [{omitted} characters omitted] ...\n")
                print(raw[-300:])

    print(f"\n{'=' * 70}")
    print(f"{unknown_count}/{total_calls} calls parsed as UNKNOWN "
          f"({100 * unknown_count / total_calls:.1f}%)" if total_calls else "no calls found")
    print("\nFor each one above, ask: is there a real answer in the raw "
          "text that the parser missed? If yes, add that phrasing to "
          "conclusion_pattern in response_parser.py. If the model "
          "genuinely didn't produce a usable answer, UNKNOWN is correct "
          "as-is — that's a real instruction-following failure worth "
          "keeping in the data.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/audit_unknowns.py <path_to_raw_responses.json>")
        sys.exit(1)
    audit(sys.argv[1])