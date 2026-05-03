import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.evals.compare_benchmark_results import compare_result_files
from tests.evals.save_benchmark_results import _safe_label, save_benchmark_results


class BenchmarkResultsToolsTests(unittest.TestCase):
    def test_safe_label_normalizes_text(self):
        self.assertEqual(_safe_label("prompt tweak #1"), "prompt_tweak_1")
        self.assertEqual(_safe_label("  planner.v2  "), "planner.v2")
        self.assertIsNone(_safe_label("   "))

    def test_save_benchmark_results_writes_timestamped_and_latest_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_results_dir = Path(tmpdir)
            with patch("tests.evals.save_benchmark_results.RESULTS_DIR", temp_results_dir):
                timestamped_path, latest_path, latest_suite_path, payload = save_benchmark_results(
                    label="baseline",
                    note="current project",
                    suite="v2",
                    mode="llm",
                )

            self.assertTrue(timestamped_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertTrue(latest_suite_path.exists())
            self.assertEqual(payload["label"], "baseline")
            self.assertEqual(payload["note"], "current project")
            self.assertEqual(payload["suite"], "v2")
            self.assertEqual(payload["mode"], "llm")
            self.assertEqual(timestamped_path.parent.name, "llm")

            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_suite_payload = json.loads(latest_suite_path.read_text(encoding="utf-8"))
            self.assertIn("report", latest_payload)
            self.assertIn("summary", latest_payload["report"])
            self.assertEqual(latest_suite_payload["suite"], "v2")
            self.assertEqual(latest_suite_payload["mode"], "llm")
            self.assertEqual(
                latest_payload["report"]["summary"]["case_count"],
                payload["report"]["summary"]["case_count"],
            )

    def test_compare_result_files_reports_summary_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_path = tmpdir_path / "base.json"
            candidate_path = tmpdir_path / "candidate.json"

            base_payload = {
                "generated_at_utc": "2026-05-03T00:00:00+00:00",
                "label": "base",
                "mode": "deterministic",
                "suite": "v1",
                "report": {
                    "summary": {
                        "intent_accuracy_pct": 100.0,
                        "ticker_accuracy_pct": 100.0,
                        "stop_reason_accuracy_pct": 100.0,
                        "right_tool_call_pct": 60.0,
                        "evidence_coverage_pct": 90.0,
                        "end_to_end_success_pct": 50.0,
                    }
                },
            }
            candidate_payload = {
                "generated_at_utc": "2026-05-03T01:00:00+00:00",
                "label": "candidate",
                "mode": "deterministic",
                "suite": "v1",
                "report": {
                    "summary": {
                        "intent_accuracy_pct": 100.0,
                        "ticker_accuracy_pct": 100.0,
                        "stop_reason_accuracy_pct": 87.5,
                        "right_tool_call_pct": 70.0,
                        "evidence_coverage_pct": 95.0,
                        "end_to_end_success_pct": 62.5,
                    }
                },
            }
            base_path.write_text(json.dumps(base_payload), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

            comparison = compare_result_files(str(base_path), str(candidate_path))

            self.assertEqual(comparison["deltas"]["right_tool_call_pct"], 10.0)
            self.assertEqual(comparison["deltas"]["evidence_coverage_pct"], 5.0)
            self.assertEqual(comparison["deltas"]["stop_reason_accuracy_pct"], -12.5)


if __name__ == "__main__":
    unittest.main()
