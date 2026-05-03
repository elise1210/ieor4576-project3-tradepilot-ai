from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tests.evals.benchmark_eval import RESULTS_DIR, run_benchmark_suite


def _safe_label(label: str | None) -> str | None:
    if label is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or None


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return sha or None


def _build_payload(
    label: str | None = None,
    note: str | None = None,
    suite: str = "v1",
    mode: str = "deterministic",
) -> dict:
    report = run_benchmark_suite(suite=suite, mode=mode)
    timestamp = datetime.now(timezone.utc)
    return {
        "generated_at_utc": timestamp.isoformat(),
        "label": label,
        "note": note,
        "suite": suite,
        "mode": mode,
        "git_commit_sha": _git_commit_sha(),
        "benchmark_version": 1,
        "report": report,
    }


def _timestamp_stem(timestamp_text: str) -> str:
    return timestamp_text.replace(":", "").replace("+00:00", "Z")


def save_benchmark_results(
    label: str | None = None,
    note: str | None = None,
    suite: str = "v1",
    mode: str = "deterministic",
) -> tuple[Path, Path, Path, dict]:
    payload = _build_payload(label=label, note=note, suite=suite, mode=mode)
    timestamp_text = payload["generated_at_utc"]
    stem = _timestamp_stem(timestamp_text)
    label_suffix = _safe_label(label)
    if label_suffix:
        stem = f"{stem}_{label_suffix}"

    mode_dir = RESULTS_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    timestamped_path = mode_dir / f"{stem}.json"
    latest_path = mode_dir / "latest.json"
    latest_suite_path = mode_dir / f"latest_{suite}.json"

    serialized = json.dumps(payload, indent=2, ensure_ascii=True)
    timestamped_path.write_text(serialized + "\n", encoding="utf-8")
    latest_path.write_text(serialized + "\n", encoding="utf-8")
    latest_suite_path.write_text(serialized + "\n", encoding="utf-8")
    return timestamped_path, latest_path, latest_suite_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark suite and save the scored results.")
    parser.add_argument(
        "--mode",
        default="deterministic",
        help="Benchmark mode to run: deterministic or llm.",
    )
    parser.add_argument("--suite", default="v1", help="Benchmark suite to run, such as v1 or v2.")
    parser.add_argument("--label", help="Optional short label such as prompt_v1 or planner_tweak.")
    parser.add_argument("--note", help="Optional free-text note describing what changed.")
    args = parser.parse_args()

    timestamped_path, latest_path, latest_suite_path, payload = save_benchmark_results(
        label=args.label,
        note=args.note,
        suite=args.suite,
        mode=args.mode,
    )
    summary = payload["report"]["summary"]

    print(f"Saved benchmark results to: {timestamped_path}")
    print(f"Updated latest snapshot: {latest_path}")
    print(f"Updated suite snapshot: {latest_suite_path}")
    print(f"Mode: {payload['mode']}")
    print(f"Suite: {payload['suite']}")
    print("Summary:")
    print(f"  Intent Accuracy %: {summary['intent_accuracy_pct']}")
    print(f"  Ticker Accuracy %: {summary['ticker_accuracy_pct']}")
    print(f"  Stop Reason Accuracy %: {summary['stop_reason_accuracy_pct']}")
    print(f"  Right Tool Call %: {summary['right_tool_call_pct']}")
    print(f"  Evidence Coverage %: {summary['evidence_coverage_pct']}")
    print(f"  End-to-End Success %: {summary['end_to_end_success_pct']}")


if __name__ == "__main__":
    main()
