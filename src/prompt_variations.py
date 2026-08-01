"""
Generates four semantically equivalent prompt variants per question:
bare, instructed, roleplay, negation.

Run script to process every file in data/processed/ and write
variant sets to data/prompts/.

    python src/prompt_variations.py
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Templates 
TEMPLATES = {
    "bare": "{question}\n{options}",

    "instructed": (
        "Answer the following question. Respond with only the letter "
        "of the correct option.\n\n{question}\n{options}"
    ),

    "roleplay": (
        "You are an expert in this subject. A student asks you the "
        "following question. Respond with only the letter of the "
        "correct option.\n\n{question}\n{options}"
    ),

    # Negation is kept separately, see build_negation_variant() because it must also flag which option becomes "correct" under the inverted framing.
}


def format_options(options: dict) -> str:
    return "\n".join(f"{letter}) {text}" for letter, text in options.items())


def build_standard_variants(question: str, options: dict) -> dict:
    """Builds bare, instructed, roleplay variants."""
    options_text = format_options(options)
    variants = {}
    for style, template in TEMPLATES.items():
        variants[style] = template.format(question=question, options=options_text)
    return variants


def build_negation_variant(question: str, options: dict, correct_answer: str) -> dict:
    options_text = format_options(options)

    prompt = (
        "Read the question and options below.\n\n"
        f"{question}\n{options_text}\n\n"
        "Now answer a different question: which ONE of the options "
        "above is INCORRECT, that is, which option does NOT correctly "
        "answer the question above? Respond with only the letter of "
        "that incorrect option."
    )

    # any option other than the original correct answer is valid here
    valid_letters = [k for k in options.keys() if k != correct_answer]

    return {
        "prompt": prompt,
        "valid_answers": valid_letters,   # scorer checks membership, not equality
    }


def generate_all_variants(item: dict) -> dict:
    standard = build_standard_variants(item["question"], item["options"])
    negation = build_negation_variant(
        item["question"], item["options"], item["correct_answer"]
    )

    return {
        "id": item["id"],
        "domain": item.get("domain"),
        "correct_answer": item["correct_answer"],
        "options": item["options"],   # kept for the fallback text-match parser
        "variants": {
            "bare": {
                "prompt": standard["bare"],
                "valid_answers": [item["correct_answer"]],
            },
            "instructed": {
                "prompt": standard["instructed"],
                "valid_answers": [item["correct_answer"]],
            },
            "roleplay": {
                "prompt": standard["roleplay"],
                "valid_answers": [item["correct_answer"]],
            },
            "negation": negation,
        },
    }


def process_domain_file(domain: str):
    in_path = os.path.join(config.DATA_PROCESSED_DIR, f"{domain}.json")
    out_path = os.path.join(config.DATA_PROMPTS_DIR, f"{domain}_prompts.json")

    if not os.path.exists(in_path):
        print(f"  [skip] {in_path} not found — run load_datasets.py first")
        return

    with open(in_path) as f:
        items = json.load(f)

    all_variants = [generate_all_variants(item) for item in items]

    os.makedirs(config.DATA_PROMPTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_variants, f, indent=2)

    print(f"  [ok] {domain}: {len(all_variants)} questions -> {out_path}")


if __name__ == "__main__":
    print("Generating prompt variants for all domains...")
    for domain in config.DOMAINS.keys():
        process_domain_file(domain)
    print("Done.")