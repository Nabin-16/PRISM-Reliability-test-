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

    # Tier 1 — bare letter, or letter with punctuation at the very
    # start. Universal across essentially every eval framework.
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()
    m = re.match(r"^\(?([A-Da-d])\)?[.:]?(?:\s|$)", text)
    if m:
        return m.group(1).upper()

    # Tier 2 — a bare letter alone on its own line, anywhere in the
    # text. Checked before the declaration-phrase tier below because
    # it's the more deliberate, less ambiguous signal: a declaration
    # phrase like "the correct answer is X" can appear while a model
    # is still discussing earlier context (e.g. restating what the
    # original, non-negated question's answer was) before reaching its
    # real conclusion — a standalone letter on its own line essentially
    # never does that; it's reserved for the actual final answer.
    standalone = re.findall(r"(?m)^\s*([A-Da-d])\s*$", text)
    if standalone:
        return standalone[-1].upper()

    # Tier 3 — one broad declaration pattern, modeled directly on
    # lm-evaluation-harness's actual ARC-Challenge filter, plus LaTeX
    # \boxed{} which is the standard convention for math-reasoning
    # benchmarks (GSM8K/MATH). Note: [^A-D] excludes ONLY uppercase
    # A-D, matching the real framework's pattern exactly — this
    # matters, because excluding lowercase a-d too (an earlier version
    # of this parser did, via a global re.IGNORECASE flag that silently
    # affected the character class too) causes the scan to stop early
    # on ordinary words like "original" or "are", which contain those
    # letters incidentally and are not answer references.
    #
    # A second alternative pattern covers the negation-mirror phrasing:
    # "option A is incorrect" / "answer A is wrong". This isn't a
    # one-off — our negation prompt style literally asks "which option
    # is incorrect", so models naturally answer in that shape rather
    # than "the correct answer is X" (which barely fits a negation
    # question at all). Structural consequence of our own prompt
    # wording, expected to recur across every model on this style, not
    # a single-model quirk.
    #
    # Both patterns' matches are collected WITH their character
    # position and merged in true text order before taking the last
    # one — simply concatenating findall() results from pattern 1 then
    # pattern 2 would let an earlier dismissal from pattern 2 override
    # a later, real declaration from pattern 1 just because of code
    # ordering, not because it actually came last in the text.
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

    # Anything else: no match. UNKNOWN is the honest, correct outcome
    # here — not a gap to keep patching.
    return None


def fallback_parse(text: str, options: dict) -> str:
    """Layer 2 — does the option's own text appear in the response."""
    text_lower = text.lower()
    for letter, option_text in options.items():
        if option_text.lower() in text_lower:
            return letter
    return "UNKNOWN"


# Explicit hedge/refusal phrasing — a model stating it cannot commit
# to an answer, often while listing all four options as part of
# explaining its own confusion. This is a genuinely different problem
# from "which letter did the model pick": there is no real answer to
# extract, and letting fallback_parse's naive substring matching run
# anyway just returns whichever option happens first in dict
# iteration order — an artifact of dict ordering, not a real answer.
# This only gates fallback_parse, not parse_response's tiers above:
# if the model explicitly DOES declare a letter (even amid some
# hedging elsewhere in the response), that declaration is a genuine
# signal and should still be trusted.
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


# ── self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    opts = {"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Vacuole"}

    test_cases = [
        ("B", "B"),
        ("The answer is B", "UNKNOWN"),   # "answer is" alone, no best/correct/final -> honestly unrecognized now, not force-matched
        ("The best answer is B", "B"),
        ("B) Mitochondria", "B"),
        ("(B)", "B"),                     # bare, unambiguous as the whole response
        ("B)", "B"),
        ("Mitochondria", "B"),            # caught by simple fallback text-match
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
        # real case: negation-style prompt, model naturally answers in
        # the mirrored "option X is incorrect" shape rather than
        # "correct answer is X" — the latter barely fits a negation
        # question at all
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

    # real case: explicit refusal — model lists all four options while
    # explaining it cannot choose. Needs its own options dict since the
    # refusal specifically depends on option text ("unhappy", etc.)
    # appearing in the response for fallback_parse to be tempted by.
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