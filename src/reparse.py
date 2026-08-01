"""
Re-runs the CURRENT response_parser.py logic against already-saved
raw response text — no Ollama calls, no new inference, just fresh
parsing. Essential after any parser fix: raw_responses/*.json files
otherwise keep whatever "parsed" value was computed by whichever
parser version was active back when inference.py originally ran.
Every fix made since then silently does nothing until this is run.

Pulls the real options dict from data/prompts/{domain}_prompts.json
(matched by question_id) so the fallback text-matching layer works
correctly too, not just the regex layer — the raw_responses files
don't store options themselves.

Usage:
    python src/reparse.py results/raw_responses/llama3.1_8b_education.json
    python src/reparse.py results/raw_responses/  (reparses every file in the folder)
"""

import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.response_parser import extract_answer


def load_options_by_question_id(domain: str) -> dict:
    prompts_path = os.path.join(config.DATA_PROMPTS_DIR, f"{domain}_prompts.json")
    if not os.path.exists(prompts_path):
        print(f"  [warn] {prompts_path} not found — reparsing with empty "
              f"options (fallback text-matching will be skipped)")
        return {}
    with open(prompts_path) as f:
        prompts = json.load(f)
    return {p["id"]: p.get("options", {}) for p in prompts}


def reparse_file(path: str):
    with open(path) as f:
        results = json.load(f)
    if not results:
        print(f"  [skip] {path} is empty")
        return

    domain = results[0].get("domain", "")
    options_by_id = load_options_by_question_id(domain)

    changed = 0
    total = 0
    for q in results:
        options = options_by_id.get(q["question_id"], {})
        for style, r in q["responses"].items():
            total += 1
            old_parsed = r["parsed"]
            new_parsed = extract_answer(r["raw"], options) if r["raw"] else "UNKNOWN"
            if new_parsed != old_parsed:
                changed += 1
                print(f"    [CHANGED] {q['question_id']}/{style}: "
                      f"{old_parsed} -> {new_parsed}")
            r["parsed"] = new_parsed

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  {changed}/{total} responses changed. Updated in place: {path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/reparse.py <path_to_json_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.json")))
        if not files:
            print(f"No .json files found in {target}")
            return
        for f in files:
            print(f"\n{f}")
            reparse_file(f)
    else:
        reparse_file(target)


if __name__ == "__main__":
    main()