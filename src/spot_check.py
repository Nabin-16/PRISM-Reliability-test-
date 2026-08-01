"""
Complements audit_unknowns.py — that script only surfaces calls the
parser couldn't extract ANY answer from. This one randomly samples
calls the parser DID extract an answer from, so you can verify the
extracted letter actually matches what the raw text concludes.

This matters because "confidently wrong" parses (the D-instead-of-C
and D-instead-of-B bugs found earlier) never show up as UNKNOWN —
they look like clean, successful parses right up until a human reads
the raw text and notices it doesn't match.

Usage:
    python src/spot_check.py results/raw_responses/llama3.1_8b_education.json
    python src/spot_check.py results/raw_responses/llama3.1_8b_education.json --n 25
"""

import argparse
import json
import random


def spot_check(path: str, n: int, seed: int):
    with open(path) as f:
        results = json.load(f)

    calls = []
    for q in results:
        for style, r in q["responses"].items():
            if r["parsed"] == "UNKNOWN":
                continue  # already covered by audit_unknowns.py
            calls.append((q["question_id"], style, r))

    random.seed(seed)
    sample = random.sample(calls, min(n, len(calls)))

    for question_id, style, r in sample:
        print(f"\n{'─' * 70}")
        print(f"question: {question_id}  |  style: {style}  |  "
              f"PARSED AS: {r['parsed']}  |  valid_answers: {r['valid_answers']}")
        print(f"{'─' * 70}")
        raw = r["raw"]
        if len(raw) <= 600:
            print(raw)
        else:
            # show head + tail, not just head — the conclusion that
            # actually determines correctness is almost always at the
            # END, and truncating from the start alone was silently
            # hiding it, making some responses unverifiable
            omitted = len(raw) - 600
            print(raw[:300])
            print(f"\n... [{omitted} characters omitted] ...\n")
            print(raw[-300:])

    print(f"\n{'=' * 70}")
    print(f"Sampled {len(sample)} of {len(calls)} successfully-parsed calls.")
    print("\nFor each one: does the raw text actually conclude with the "
          "letter shown as PARSED AS? If any don't match, that's a real "
          "parser bug — send the raw text back for a fix, the same way "
          "we did for the D-instead-of-C and D-instead-of-B cases.")
    print("\nRun with a different --seed to sample a fresh batch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to a raw_responses json file")
    parser.add_argument("--n", type=int, default=15,
                         help="how many calls to sample (default 15)")
    parser.add_argument("--seed", type=int, default=42,
                         help="random seed, change for a different sample")
    args = parser.parse_args()
    spot_check(args.path, args.n, args.seed)