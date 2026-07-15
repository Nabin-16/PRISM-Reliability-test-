"""
Two-layer answer extraction: strict regex patterns first, then
fallback heuristics that match against the actual option text.
Anything that survives both layers unparsed is logged as UNKNOWN.
"""

import re


def parse_response(text: str) -> str | None:
    """Layer 1, regex pattern matching. Returns 'A'-'D' or None."""
    text = text.strip()

    # single letter alone: "B" or "b"
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()

    # letter with punctuation at the start: "B.", "B)", "B:"
    m = re.match(r"^\(?([A-Da-d])\)?[.:]?\s", text)
    if m:
        return m.group(1).upper()

    # "answer is X" / "answer: X"
    m = re.search(r"answer\s*(?:is|:)\s*[:\s]*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # "option X" / "option (X)"
    m = re.search(r"option\s*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # letter in parentheses anywhere: "(B)"
    m = re.search(r"\(([A-Da-d])\)", text)
    if m:
        return m.group(1).upper()

    # "B) <word>" pattern anywhere in the text
    m = re.search(r"\b([A-Da-d])\)\s*\w", text)
    if m:
        return m.group(1).upper()

    return None


def fallback_parse(text: str, options: dict) -> str:
    """
    Layer 2, string matching against actual option text.
    Used when the model answers with the option's content
    instead of its letter.
    """
    text_lower = text.lower()

    # exact option text appears in the response
    for letter, option_text in options.items():
        if option_text.lower() in text_lower:
            return letter

    # first significant word of the option appears
    for letter, option_text in options.items():
        first_word = option_text.split()[0].lower()
        if len(first_word) > 3 and first_word in text_lower:
            return letter

    # most frequently mentioned letter, if any, search the ORIGINAL
    # (non-uppercased) text and require the capital form specifically.
    # Uppercasing first would let the English article "a" masquerade
    # as answer choice A; real letter-references are capitalized in
    # natural model output, the article almost never is.
    counts = {L: len(re.findall(rf"\b{L}\b", text)) for L in "ABCD"}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best

    return "UNKNOWN"


def extract_answer(raw_response: str, options: dict) -> str:
    """Full two-layer pipeline. Always returns 'A'-'D' or 'UNKNOWN'."""
    result = parse_response(raw_response)
    if result:
        return result
    return fallback_parse(raw_response, options)


# a self-test
if __name__ == "__main__":
    opts = {"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome", "D": "Vacuole"}

    test_cases = [
        ("B", "B"),
        ("The answer is B", "B"),
        ("B) Mitochondria", "B"),
        ("(B)", "B"),
        ("Option B is correct", "B"),
        ("I believe the correct answer would be B) Mitochondria, "
         "since it produces ATP.", "B"),
        ("Mitochondria", "B"),                    
        ("It's definitely the mitochondria", "B"), 
        ("I'm not sure, could be A or C", "A"),    
        ("", "UNKNOWN"),
        ("This is a completely unrelated response.", "UNKNOWN"),
    ]

    passed = 0
    for raw, expected in test_cases:
        got = extract_answer(raw, opts)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] input={raw!r:55s} expected={expected} got={got}")

    print(f"\n{passed}/{len(test_cases)} test cases passed")
