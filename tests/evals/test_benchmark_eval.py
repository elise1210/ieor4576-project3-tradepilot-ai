import unittest

from app.state import build_initial_state

from tests.evals.benchmark_cases import BENCHMARK_CASES
from tests.evals.benchmark_eval import aggregate_case_scores, run_benchmark_suite, score_case


class BenchmarkEvalLogicTests(unittest.TestCase):
    def test_score_case_grants_full_marks_when_state_matches_case(self):
        case = {
            "id": "synthetic_full_match",
            "query": "Should I buy Apple this week?",
            "expected_intent": "buy_sell_decision",
            "expected_tickers": ["AAPL"],
            "required_tools": ["news", "market"],
            "forbidden_tools": ["chart"],
            "required_evidence": ["news", "market"],
            "expected_stop_reason": "decision_completed",
        }
        state = build_initial_state(case["query"])
        state["intent"] = "buy_sell_decision"
        state["tickers"] = ["AAPL"]
        state["metadata"]["stopped_reason"] = "decision_completed"
        state["metadata"]["executed_research_steps"] = [
            {"skill": "news", "ticker": "AAPL", "params": {}},
            {"skill": "market", "ticker": "AAPL", "params": {}},
        ]
        state["evidence"]["news"]["AAPL"] = {"summary": "Positive coverage."}
        state["evidence"]["market"]["AAPL"] = {"trend_label": "upward"}

        result = score_case(state, case)

        self.assertTrue(result["intent_correct"])
        self.assertTrue(result["tickers_correct"])
        self.assertTrue(result["stop_reason_correct"])
        self.assertEqual(result["right_tool_call_pct"], 100.0)
        self.assertEqual(result["wrong_tool_call_pct"], 0.0)
        self.assertEqual(result["evidence_coverage_pct"], 100.0)
        self.assertEqual(result["missing_required_tools"], [])
        self.assertEqual(result["wrong_tools"], [])
        self.assertTrue(result["case_pass"])

    def test_score_case_penalizes_missing_and_forbidden_tools(self):
        case = {
            "id": "synthetic_mismatch",
            "query": "What is AAPL stock price today?",
            "expected_intent": "general_research",
            "expected_tickers": ["AAPL"],
            "required_tools": ["market", "chart"],
            "forbidden_tools": ["news", "fundamentals", "sentiment"],
            "required_evidence": ["market", "chart"],
            "expected_stop_reason": "research_completed",
        }
        state = build_initial_state(case["query"])
        state["intent"] = "general_research"
        state["tickers"] = ["AAPL"]
        state["metadata"]["stopped_reason"] = "research_completed"
        state["metadata"]["executed_research_steps"] = [
            {"skill": "fundamentals", "ticker": "AAPL", "params": {}},
            {"skill": "news", "ticker": "AAPL", "params": {}},
        ]
        state["evidence"]["market"]["AAPL"] = {"current_price": 123.45}

        result = score_case(state, case)

        self.assertEqual(result["right_tool_call_pct"], 0.0)
        self.assertEqual(result["wrong_tool_call_pct"], 100.0)
        self.assertEqual(result["evidence_coverage_pct"], 50.0)
        self.assertEqual(result["missing_required_tools"], ["market", "chart"])
        self.assertEqual(result["wrong_tools"], ["fundamentals", "news"])
        self.assertFalse(result["case_pass"])

    def test_aggregate_case_scores_rolls_up_logic_metrics(self):
        scorecards = [
            {
                "intent_correct": True,
                "tickers_correct": True,
                "stop_reason_correct": True,
                "required_tool_hits": 2,
                "required_tool_total": 2,
                "wrong_tool_hits": 0,
                "called_tool_total": 2,
                "evidence_hits": 2,
                "evidence_total": 2,
                "case_pass": True,
            },
            {
                "intent_correct": True,
                "tickers_correct": False,
                "stop_reason_correct": True,
                "required_tool_hits": 1,
                "required_tool_total": 2,
                "wrong_tool_hits": 1,
                "called_tool_total": 2,
                "evidence_hits": 1,
                "evidence_total": 2,
                "case_pass": False,
            },
        ]

        summary = aggregate_case_scores(scorecards)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["intent_accuracy_pct"], 100.0)
        self.assertEqual(summary["ticker_accuracy_pct"], 50.0)
        self.assertEqual(summary["stop_reason_accuracy_pct"], 100.0)
        self.assertEqual(summary["right_tool_call_pct"], 75.0)
        self.assertEqual(summary["wrong_tool_call_pct"], 25.0)
        self.assertEqual(summary["evidence_coverage_pct"], 75.0)
        self.assertEqual(summary["end_to_end_success_pct"], 50.0)


class BenchmarkEvalIntegrationTests(unittest.TestCase):
    def test_run_benchmark_suite_returns_scored_cases_and_summary(self):
        report = run_benchmark_suite()

        self.assertEqual(len(report["cases"]), len(BENCHMARK_CASES))
        self.assertEqual(report["summary"]["case_count"], len(BENCHMARK_CASES))
        self.assertIn("right_tool_call_pct", report["summary"])
        self.assertIn("wrong_tool_call_pct", report["summary"])
        self.assertIn("evidence_coverage_pct", report["summary"])
        self.assertIn("end_to_end_success_pct", report["summary"])
        self.assertTrue(any(not case["case_pass"] for case in report["cases"]))
        self.assertTrue(any(case["case_pass"] for case in report["cases"]))


if __name__ == "__main__":
    unittest.main()
