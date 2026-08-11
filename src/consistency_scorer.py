"""
Reads raw inference results and computes, per question:
  - majority-vote answer (with bare-prompt tiebreak)
  - agreement score (proportion of styles matching the majority)
  - correctness (does the majority answer satisfy valid_answers?)

Then aggregates to the model x domain x style level to build the
reliability matrix that the Prompt Normalizer will read from later.

Usage:
    python src/consistency_scorer.py
"""

import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def majority_vote_with_tiebreak(responses: dict) -> tuple[str, float]:
    valid = {style: r["parsed"] for style, r in responses.items()
             if r["parsed"] != "UNKNOWN"}

    if not valid:
        return "UNKNOWN", 0.0

    counts = Counter(valid.values())
    max_count = max(counts.values())
    top_answers = [a for a, c in counts.items() if c == max_count]

    if len(top_answers) == 1:
        majority = top_answers[0]
    else:
        majority = valid.get("bare", top_answers[0])

    agreement = max_count / len(valid)
    return majority, agreement


def is_correct(majority_answer: str, valid_answers_by_style: dict) -> bool:
    all_valid = set()
    for style_data in valid_answers_by_style.values():
        all_valid.update(style_data)
    return majority_answer in all_valid


def score_file(raw_path: str) -> list[dict]:
    with open(raw_path) as f:
        questions = json.load(f)

    scored = []
    for q in questions:
        responses = q["responses"]
        majority, agreement = majority_vote_with_tiebreak(responses)

        valid_answers_by_style = {
            style: r["valid_answers"] for style, r in responses.items()
        }
        correct = is_correct(majority, valid_answers_by_style)

        unknown_count = sum(1 for r in responses.values() if r["parsed"] == "UNKNOWN")

        scored.append({
            "question_id": q["question_id"],
            "domain": q["domain"],
            "majority_answer": majority,
            "agreement_score": round(agreement, 3),
            "correct": correct,
            "unknown_count": unknown_count,
            "total_styles": len(responses),
            "per_style_answers": {s: r["parsed"] for s, r in responses.items()},
        })

    return scored


def build_reliability_matrix():
    raw_files = glob.glob(os.path.join(config.RESULTS_RAW_DIR, "*.json"))
    if not raw_files:
        print("No raw response files found yet. Run inference.py first.")
        return

    os.makedirs(config.RESULTS_SCORED_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_SUMMARY_DIR, exist_ok=True)

    model_domain_rows = []
    model_domain_style_rows = []

    for raw_path in raw_files:
        fname = os.path.basename(raw_path).replace(".json", "")
        # filenames are {model_safe}_{domain} — model_safe has underscores
        # in place of colons, and domain is one of education/science/legal
        domain = fname.split("_")[-1]
        model_safe = fname[: -(len(domain) + 1)]
        model = model_safe.replace("_", ":", 1) 

        scored = score_file(raw_path)

        scored_out = os.path.join(config.RESULTS_SCORED_DIR, f"{fname}_scored.json")
        with open(scored_out, "w") as f:
            json.dump(scored, f, indent=2)

        n = len(scored)
        if n == 0:
            continue

        accuracy = sum(s["correct"] for s in scored) / n
        mean_consistency = sum(s["agreement_score"] for s in scored) / n
        total_calls = sum(s["total_styles"] for s in scored)
        total_unknown = sum(s["unknown_count"] for s in scored)
        unknown_rate = total_unknown / total_calls if total_calls else 0.0

        model_domain_rows.append({
            "model": model,
            "domain": domain,
            "n_questions": n,
            "accuracy": round(accuracy, 4),
            "mean_consistency": round(mean_consistency, 4),
            "unknown_rate": round(unknown_rate, 4),
        })

        with open(raw_path) as f:
            raw_questions = json.load(f)

        style_correct = {s: 0 for s in config.PROMPT_STYLES}
        style_total = {s: 0 for s in config.PROMPT_STYLES}
        for q in raw_questions:
            for style, r in q["responses"].items():
                style_total[style] += 1
                if r["parsed"] in r["valid_answers"]:
                    style_correct[style] += 1

        for style in config.PROMPT_STYLES:
            if style_total[style] == 0:
                continue
            model_domain_style_rows.append({
                "model": model,
                "domain": domain,
                "style": style,
                "accuracy": round(style_correct[style] / style_total[style], 4),
                "n": style_total[style],
            })

    matrix_path = os.path.join(config.RESULTS_SUMMARY_DIR, "reliability_matrix.csv")
    with open(matrix_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "domain", "n_questions", "accuracy",
                           "mean_consistency", "unknown_rate"]
        )
        writer.writeheader()
        writer.writerows(sorted(model_domain_rows, key=lambda r: (r["model"], r["domain"])))

    style_path = os.path.join(config.RESULTS_SUMMARY_DIR, "style_breakdown.csv")
    with open(style_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "domain", "style", "accuracy", "n"]
        )
        writer.writeheader()
        writer.writerows(sorted(model_domain_style_rows,
                                 key=lambda r: (r["model"], r["domain"], r["style"])))

    print(f"Reliability matrix -> {matrix_path}")
    print(f"Style breakdown     -> {style_path}")

    print("\nBest overall prompt style per model (for the Normalizer):")
    by_model_style = {}
    for row in model_domain_style_rows:
        key = (row["model"], row["style"])
        by_model_style.setdefault(key, []).append(row["accuracy"])

    best_per_model = {}
    for (model, style), accs in by_model_style.items():
        mean_acc = sum(accs) / len(accs)
        if model not in best_per_model or mean_acc > best_per_model[model][1]:
            best_per_model[model] = (style, mean_acc)

    for model, (style, acc) in sorted(best_per_model.items()):
        print(f"  {model:20s} -> {style:12s} (mean accuracy across domains: {acc:.3f})")


if __name__ == "__main__":
    build_reliability_matrix()
