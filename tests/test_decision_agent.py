import unittest

from app.agents.decision_agent import run_decision_agent
from app.state import build_initial_state


def build_single_decision_state() -> dict:
    state = build_initial_state("Should I buy Apple this week?")
    state["tickers"] = ["AAPL"]
    state["evidence"]["news"]["AAPL"] = {"summary": "Positive product and earnings coverage."}
    state["evidence"]["market"]["AAPL"] = {
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
    }
    state["evidence"]["fundamentals"]["AAPL"] = {"summary": "Strong business profile."}
    state["evidence"]["sentiment"]["AAPL"] = {
        "sentiment": "positive",
        "score": 0.42,
        "dispersion": 0.10,
    }
    state["confidence"] = "High"
    return state


class DecisionAgentTests(unittest.TestCase):
    def test_decision_agent_creates_single_ticker_decision(self):
        state = build_single_decision_state()

        result = run_decision_agent(state)

        self.assertEqual(result["decision"]["ticker"], "AAPL")
        self.assertIn(result["decision"]["recommendation"], {"BUY", "HOLD", "SELL"})
        self.assertEqual(result["decision"]["recommendation"], "BUY")
        self.assertIn("disclaimer", result["decision"])
        self.assertTrue(result["decision"]["reasoning"])

    def test_decision_agent_creates_comparison_decision(self):
        state = build_initial_state("Compare Nvidia vs AMD")
        state["tickers"] = ["NVDA", "AMD"]
        state["evidence"]["news"]["NVDA"] = {"summary": "Strong AI demand coverage."}
        state["evidence"]["market"]["NVDA"] = {"trend_7d": 0.05, "trend_label": "upward", "volatility": 0.02}
        state["evidence"]["fundamentals"]["NVDA"] = {"summary": "Strong positioning."}
        state["evidence"]["sentiment"]["NVDA"] = {"sentiment": "positive", "score": 0.45, "dispersion": 0.10}
        state["evidence"]["news"]["AMD"] = {"summary": "Mixed near-term coverage."}
        state["evidence"]["market"]["AMD"] = {"trend_7d": 0.01, "trend_label": "sideways", "volatility": 0.02}
        state["evidence"]["fundamentals"]["AMD"] = {"summary": "Solid but more mixed profile."}
        state["evidence"]["sentiment"]["AMD"] = {"sentiment": "neutral", "score": 0.05, "dispersion": 0.12}
        state["confidence"] = "Medium"

        result = run_decision_agent(state)

        self.assertEqual(result["decision"]["type"], "comparison")
        self.assertEqual(set(result["decision"]["per_ticker"].keys()), {"NVDA", "AMD"})
        self.assertIn("comparison_summary", result["decision"])
        self.assertEqual(result["decision"]["confidence"], "Medium")

    def test_decision_agent_adds_caution_when_supporting_evidence_missing(self):
        state = build_single_decision_state()
        state["critic_result"] = {
            "supporting_missing": ["fundamentals:AAPL"],
            "conflicts": [],
        }

        result = run_decision_agent(state)

        self.assertEqual(result["decision"]["confidence"], "Low")
        self.assertIn(
            "Some supporting evidence was unavailable: fundamentals.",
            result["decision"]["reasoning"],
        )
        self.assertEqual(
            result["decision"]["evidence_status"]["supporting_missing"],
            ["fundamentals"],
        )


if __name__ == "__main__":
    unittest.main()
