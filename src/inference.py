"""
Runs every prompt variant through a locally-loaded Ollama model and
logs raw, parsed responses.
Usage:
    python src/inference.py --model <model_name> --domain <domain>
    python src/inference.py --model <model_name> --domain all
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.response_parser import extract_answer


def call_ollama(prompt: str, model: str) -> tuple[str, float]:
    """Returns (raw_response_text, latency_seconds)."""
    start = time.time()
    response = requests.post(
        config.OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "temperature": config.TEMPERATURE,
            "stream": False,
            "options": {"num_predict": config.MAX_TOKENS},
        },
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    latency = time.time() - start
    return response.json()["response"], latency


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "unknown"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_domain(model: str, domain: str):
    prompts_path = os.path.join(config.DATA_PROMPTS_DIR, f"{domain}_prompts.json")
    if not os.path.exists(prompts_path):
        print(f"  [skip] {prompts_path} not found — "
              f"run load_datasets.py and prompt_variations.py first")
        return

    with open(prompts_path) as f:
        questions = json.load(f)

    model_safe = model.replace(":", "_")
    out_path = os.path.join(
        config.RESULTS_RAW_DIR, f"{model_safe}_{domain}.json"
    )
    os.makedirs(config.RESULTS_RAW_DIR, exist_ok=True)

    # skip questions already logged
    already_done = set()
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        already_done = {r["question_id"] for r in results}
        print(f"  resuming: {len(already_done)} questions already done")

    total = len(questions)
    remaining_ids = [q["id"] for q in questions if q["id"] not in already_done]
    n_remaining = len(remaining_ids)
    n_styles = len(config.PROMPT_STYLES)
    total_calls_remaining = n_remaining * n_styles

    calls_done = 0
    run_start = time.time()
    latency_window = []   # rolling average over the last 20 calls, adapts if a bigger/slower model kicks in mid-run

    for i, q in enumerate(questions):
        if q["id"] in already_done:
            continue

        question_result = {
            "question_id": q["id"],
            "domain": q["domain"],
            "correct_answer": q["correct_answer"],
            "responses": {},
        }

        for style, variant in q["variants"].items():
            try:
                raw, latency = call_ollama(variant["prompt"], model)
            except requests.exceptions.RequestException as e:
                print(f"    [error] {model} on {q['id']}/{style}: {e}")
                raw, latency = "", -1

            options = q.get("options", {})
            parsed = extract_answer(raw, options) if raw else "UNKNOWN"

            question_result["responses"][style] = {
                "raw": raw,
                "parsed": parsed,
                "latency_sec": round(latency, 2),
                "valid_answers": variant["valid_answers"],
            }

            calls_done += 1
            if latency > 0:
                latency_window.append(latency)
                latency_window = latency_window[-20:]  # keep last 20 only

        results.append(question_result)

        # progress line after every question
        pct = 100 * calls_done / total_calls_remaining if total_calls_remaining else 100
        avg_latency = sum(latency_window) / len(latency_window) if latency_window else 0
        calls_left = total_calls_remaining - calls_done
        eta_sec = avg_latency * calls_left
        elapsed = time.time() - run_start
        print(f"  [{model}/{domain}] {calls_done}/{total_calls_remaining} calls "
              f"({pct:.1f}%) | elapsed {format_eta(elapsed)} "
              f"| avg {avg_latency:.1f}s/call")

        if (i + 1) % 10 == 0 or (i + 1) == total:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    total_time = time.time() - run_start
    print(f"  [ok] {model} / {domain}: {len(results)} questions "
          f"in {format_eta(total_time)} -> {out_path}")


def check_ollama_running():
    try:
        requests.get("http://localhost:11434", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. llama3.2:3b")
    parser.add_argument("--domain", required=True,
                         help="education | science | legal | all")
    args = parser.parse_args()

    if not check_ollama_running():
        print("ERROR: Ollama does not appear to be running.")
        print("Start it with: ollama serve")
        sys.exit(1)

    domains = list(config.DOMAINS.keys()) if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\nRunning {args.model} on {domain}...")
        run_domain(args.model, domain)
