from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.evals.benchmark_eval import RESULTS_DIR


SUMMARY_KEYS = [
    "intent_accuracy_pct",
    "ticker_accuracy_pct",
    "stop_reason_accuracy_pct",
    "right_tool_call_pct",
    "evidence_coverage_pct",
    "end_to_end_success_pct",
]


def _load_payload(path_text: str) -> dict:
    path = Path(path_text)
    return json.loads(path.read_text(encoding="utf-8"))


def _default_candidate_path(base_path: str) -> str:
    base_payload = _load_payload(base_path)
    suite = base_payload.get("suite")
    mode = base_payload.get("mode")
    mode_dir = RESULTS_DIR / (str(mode).strip().lower() if mode else "deterministic")
    if isinstance(suite, str) and suite.strip():
        suite_path = mode_dir / f"latest_{suite.strip().lower()}.json"
        if suite_path.exists():
            return str(suite_path)
    return str(mode_dir / "latest.json")


def compare_result_files(base_path: str, candidate_path: str) -> dict:
    base_payload = _load_payload(base_path)
    candidate_payload = _load_payload(candidate_path)
    base_summary = base_payload["report"]["summary"]
    candidate_summary = candidate_payload["report"]["summary"]

    deltas = {}
    for key in SUMMARY_KEYS:
        deltas[key] = round(candidate_summary[key] - base_summary[key], 2)

    return {
        "base_path": str(Path(base_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "base_label": base_payload.get("label"),
        "candidate_label": candidate_payload.get("label"),
        "base_suite": base_payload.get("suite"),
        "candidate_suite": candidate_payload.get("suite"),
        "base_mode": base_payload.get("mode"),
        "candidate_mode": candidate_payload.get("mode"),
        "base_generated_at_utc": base_payload.get("generated_at_utc"),
        "candidate_generated_at_utc": candidate_payload.get("generated_at_utc"),
        "base_summary": base_summary,
        "candidate_summary": candidate_summary,
        "deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved benchmark result files.")
    parser.add_argument("base", help="Baseline JSON file.")
    parser.add_argument(
        "candidate",
        nargs="?",
        default=None,
        help="Candidate JSON file. Defaults to the latest snapshot for the base suite.",
    )
    args = parser.parse_args()

    candidate_path = args.candidate or _default_candidate_path(args.base)
    comparison = compare_result_files(args.base, candidate_path)

    print(f"Base: {comparison['base_path']}")
    print(f"Candidate: {comparison['candidate_path']}")
    print(f"Base mode: {comparison['base_mode']}")
    print(f"Candidate mode: {comparison['candidate_mode']}")
    print(f"Base suite: {comparison['base_suite']}")
    print(f"Candidate suite: {comparison['candidate_suite']}")
    print("Metric deltas (candidate - base):")
    for key in SUMMARY_KEYS:
        delta = comparison["deltas"][key]
        sign = "+" if delta >= 0 else ""
        print(f"  {key}: {sign}{delta}")


if __name__ == "__main__":
    main()
