"""
Downloads and samples the three benchmark datasets used in the
empirical study phase. Requires internet access to HuggingFace —
run this on your own machine, not in a sandboxed environment.

    pip install datasets
    python src/load_datasets.py

Produces data/processed/{education,science,legal}.json, each a list of:
    {
        "id": "edu_0007",
        "domain": "education",
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "correct_answer": "B"
    }
"""

import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from datasets import load_dataset
except ImportError as e:
    print("Failed to import 'datasets'. This could mean it's not")
    print("installed in the Python environment actually running this")
    print("script (check with: where python / where pip — they should")
    print("point to the same environment), or that it's installed but")
    print("a sub-dependency is broken.")
    print(f"\nActual error: {e}")
    sys.exit(1)

random.seed(42)  # reproducibility — same 200 questions every run


def load_education():
    """ARC-Challenge — 4-option science/reasoning MCQs."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        if len(labels) != 4:
            continue  # keep only clean 4-option questions
        # normalize labels to A/B/C/D regardless of source labeling (1/2/3/4 etc.)
        letter_map = {orig: chr(65 + i) for i, orig in enumerate(labels)}
        options = {letter_map[l]: t for l, t in zip(labels, texts)}
        correct_orig = row["answerKey"]
        if correct_orig not in letter_map:
            continue
        items.append({
            "question": row["question"],
            "options": options,
            "correct_answer": letter_map[correct_orig],
        })
    return items


def load_science():
    """SciQ — 4-option science questions with a distractor set."""
    ds = load_dataset("allenai/sciq", split="test")
    items = []
    for row in ds:
        distractors = [row["distractor1"], row["distractor2"], row["distractor3"]]
        correct = row["correct_answer"]
        all_options = distractors + [correct]
        random.shuffle(all_options)
        letters = ["A", "B", "C", "D"]
        options = dict(zip(letters, all_options))
        correct_letter = letters[all_options.index(correct)]
        items.append({
            "question": row["question"],
            "options": options,
            "correct_answer": correct_letter,
        })
    return items


def load_legal():
    """
    LegalBench consumer_contracts_qa — 400 Yes/No questions on rights
    and obligations in consumer terms-of-service contracts.

    Replaces an earlier choice, abercrombie (5-way trademark
    distinctiveness classification), for two concrete reasons:
      1. Abercrombie's test set is only ~95 items — never enough for
         a 200-question sample. Consumer Contracts QA has 400.
      2. Abercrombie is a narrow IP-law taxonomy task, not
         representative "legal reasoning" to a general audience, and
         its 5-way structure forced into a 4-option MCQ made negation
         ("which is NOT the answer") inherently fuzzy — 3 valid "not X"
         answers out of 4. A binary Yes/No task makes negation
         trivially well-defined: NOT Yes = No, exactly one right
         answer, no ambiguity. This is a real methodological
         improvement to the negation prompt style, not just a
         data-volume workaround.

    Kept as a 2-option "MCQ" (A=Yes, B=No) rather than padded to 4
    options — the task is natively binary, and forcing in two
    meaningless extra options would be worse than just using two.
    """
    ds = load_dataset("nguha/legalbench", "consumer_contracts_qa", split="test")
    items = []
    for row in ds:
        answer = row["answer"].strip().capitalize()  # normalize "yes"/"YES" -> "Yes"
        if answer not in ("Yes", "No"):
            continue  # skip any malformed rows
        options = {"A": "Yes", "B": "No"}
        correct_letter = "A" if answer == "Yes" else "B"
        items.append({
            "question": f"{row['question']}\n\nContract clause:\n{row['contract']}",
            "options": options,
            "correct_answer": correct_letter,
        })
    return items


LOADERS = {
    "education": load_education,
    "science": load_science,
    "legal": load_legal,
}


def sample_and_save(domain: str, n: int = None):
    n = n or config.SAMPLE_SIZE_PER_DOMAIN
    print(f"Loading {domain}...")
    items = LOADERS[domain]()
    print(f"  {len(items)} candidate questions available")

    sampled = random.sample(items, min(n, len(items)))
    for i, item in enumerate(sampled):
        item["id"] = f"{domain[:3]}_{i:04d}"
        item["domain"] = domain

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.DATA_PROCESSED_DIR, f"{domain}.json")
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"  sampled {len(sampled)} -> {out_path}")


if __name__ == "__main__":
    for domain in config.DOMAINS.keys():
        sample_and_save(domain)
    print("\nAll datasets loaded and sampled.")