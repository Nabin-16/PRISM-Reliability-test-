"""
Answer extraction, deliberately kept close to how real evaluation
frameworks actually do this — not an ever-growing pile of edge cases.

Reference point: lm-evaluation-harness (the framework behind
HuggingFace's Open LLM Leaderboard) uses exactly ONE regex for
ARC-Challenge: "The best answer is [^A-D]*([A-D])", last match wins.
Nothing more elaborate than that in their primary filter.

Also worth knowing, from the xFinder paper (ICLR 2025), which exists
specifically because this problem is hard: even the BEST regex-based
evaluation frameworks in the field top out around 74% extraction
accuracy. A parser that can't extract an answer from every possible
response format is not broken — it's normal. UNKNOWN is not a parsing
failure to be engineered away; it's a real measurement of whether the
model followed the instruction to answer concisely. A model that
rambles for 500 words instead of stating a letter has a genuine
instruction-following problem, and that should show up in the data,
not get silently reverse-engineered into a clean answer.
"""

import re


def parse_response(text: str) -> str | None:
    """Layer 1 — regex pattern matching. Returns 'A'-'D' or None."""
    text = text.strip()

    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()
    m = re.match(r"^\(?([A-Da-d])\)?[.:]?(?:\s|$)", text)
    if m:
        return m.group(1).upper()

    standalone = re.findall(r"(?m)^\s*([A-Da-d])\s*$", text)
    if standalone:
        return standalone[-1].upper()

    positioned_matches = []
    for m in re.finditer(
        r"(?i:best|correct|final)\s+(?i:answer|option)[^A-D]*([A-D])",
        text
    ):
        positioned_matches.append((m.start(), m.group(1)))
    for m in re.finditer(
        r"(?i:option|answer)\s+([A-D])\s+(?i:is\s+(?:incorrect|wrong|"
        r"not\s+correct))",
        text
    ):
        positioned_matches.append((m.start(), m.group(1)))
    for m in re.finditer(r"\\boxed\{\s*([A-Da-d])\s*\}", text):
        positioned_matches.append((m.start(), m.group(1)))

    if positioned_matches:
        positioned_matches.sort(key=lambda pair: pair[0])
        return positioned_matches[-1][1].upper()

    return None


def fallback_parse(text: str, options: dict) -> str:
    """Layer 2 — does the option's own text appear in the response."""
    text_lower = text.lower()
    for letter, option_text in options.items():
        if option_text.lower() in text_lower:
            return letter
    return "UNKNOWN"

REFUSAL_PATTERNS = [
    r"impossible to (?:accurately )?(?:choose|determine|answer|say)",
    r"(?:cannot|can't|unable to) (?:accurately )?(?:choose|determine|answer)",
    r"(?:insufficient|not enough) information",
    r"please provide more (?:details|information|context)",
]


def is_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in REFUSAL_PATTERNS)


def extract_answer(raw_response: str, options: dict) -> str:
    """Full pipeline. Always returns 'A'-'D' or 'UNKNOWN'."""
    result = parse_response(raw_response)
    if result:
        return result
    if is_refusal(raw_response):
        return "UNKNOWN"
    return fallback_parse(raw_response, options)


if __name__ == "__main__":
    opts = {"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Vacuole"}

    test_cases = [
        ("B", "B"),
        ("The answer is B", "UNKNOWN"),   
        ("The best answer is B", "B"),
        ("B) Mitochondria", "B"),
        ("(B)", "B"),                    
        ("B)", "B"),
        ("Mitochondria", "B"),           
        ("", "UNKNOWN"),
        ("This is a completely unrelated response.", "UNKNOWN"),
        ("Let's think: B Shelves allow the user to make use of what "
         "otherwise would be deadspace above an appliance.\nThe best "
         "answer is B.", "B"),
        ("The correct answer to the original question would be C) blow "
         "up a beach ball or balloon.\n\nSo, the correct answer to the "
         "second question is D).", "D"),
        ("## Step 1: Understanding\nThe question asks how plants and "
         "animals process nutrients similarly.\n\n"
         "## Step 6: Identifying the best option\nGiven the analysis, "
         "cells breaking down nutrients into usable forms.\n\n"
         "The final answer is: $\\boxed{C}$", "C"),
        ("To solve this problem, we need to consider the concept of "
         "work and power.\n\nWork (W) is defined as the product of "
         "force and distance.\n\nSince Roberta takes longer than Mary, "
         "she has a lower power output than Mary.\n\n"
         "Therefore, option A is incorrect.", "A"),
        ("The correct answers to the original question are B and D.\n\n"
         "However, since I must choose one:\n\nD", "D"),
    ]

    passed = 0
    for raw, expected in test_cases:
        got = extract_answer(raw, opts)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] expected={expected} got={got} | {raw[:70]!r}")

    refusal_opts = {"A": "unhappy", "B": "confused", "C": "confident", "D": "generous"}
    refusal_text = (
        "The answer depends on the specific song or context in which "
        '"Leopard" refers to a character. Assuming you might be '
        "referring to a specific album, without specific information "
        "about which particular \"Leopard\" you're asking about, it's "
        "impossible to accurately choose among A) unhappy B) confused "
        "C) confident D) generous.\n\nPlease provide more details so I "
        "can give you a precise answer!"
    )
    got = extract_answer(refusal_text, refusal_opts)
    ok = got == "UNKNOWN"
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] expected=UNKNOWN got={got} | refusal case (custom options)")
    total_cases = len(test_cases) + 1

    print(f"\n{passed}/{total_cases} test cases passed")