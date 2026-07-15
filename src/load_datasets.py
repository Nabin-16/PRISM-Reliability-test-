"""
Downloads and samples the three benchmark datasets used in the
empirical study phase. Requires internet access to HuggingFace datasets. Please,
run this on your own machine, not in a sandboxed environment......

    pip install datasets
    python src/load_datasets.py
"""

import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from datasets import load_dataset
except ImportError:
    print("Missing dependency. Run: pip install datasets")
    sys.exit(1)

random.seed(42)  # same 200 questions every run


def load_education():
    """ARC-Challenge"""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        if len(labels) != 4:
            continue  # keep only clean 4-option questions
        # normalize labels to A/B/C/D
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
    """SciQ"""
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
    LegalBench abercrombie subtask, only 95 pulled......
    """
    ds = load_dataset("nguha/legalbench", "abercrombie", split="test")
    all_labels = sorted(set(row["answer"] for row in ds))

    items = []
    for row in ds:
        correct = row["answer"]
        wrong_pool = [l for l in all_labels if l != correct]
        wrong_choices = random.sample(wrong_pool, min(3, len(wrong_pool)))
        all_options = wrong_choices + [correct]
        random.shuffle(all_options)
        letters = ["A", "B", "C", "D"][: len(all_options)]
        options = dict(zip(letters, all_options))
        correct_letter = letters[all_options.index(correct)]
        items.append({
            "question": f"Classify the mark described below.\n\n{row['text']}",
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
