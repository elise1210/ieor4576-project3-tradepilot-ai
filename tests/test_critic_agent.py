import unittest

from app.agents.critic_agent import run_critic_agent
from app.state import build_initial_state


def build_complete_state_single() -> dict:
    state = build_initial_state("Should I buy Apple this week?")
    state["tickers"] = ["AAPL"]
    state["plan"]["required_evidence"] = ["news", "market", "fundamentals", "sentiment"]
    state["evidence"]["news"]["AAPL"] = {"summary": "Positive coverage."}
    state["evidence"]["market"]["AAPL"] = {"trend_label": "upward", "trend_7d": 0.03}
    state["evidence"]["fundamentals"]["AAPL"] = {"summary": "Stable fundamentals."}
    state["evidence"]["sentiment"]["AAPL"] = {"sentiment": "positive", "score": 0.42}
    return state


class CriticAgentTests(unittest.TestCase):
    def test_critic_marks_complete_single_ticker_state_as_enough(self):
        state = build_complete_state_single()

        result = run_critic_agent(state)

        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertEqual(result["critic_result"]["missing"], [])
        self.assertEqual(result["critic_result"]["fairness_issues"], [])
        self.assertEqual(result["critic_result"]["conflicts"], [])
        self.assertEqual(result["critic_result"]["confidence"], "High")

    def test_critic_flags_missing_fundamentals(self):
        state = build_complete_state_single()
        state["evidence"]["fundamentals"]["AAPL"] = {}

        result = run_critic_agent(state)

        self.assertFalse(result["critic_result"]["enough_evidence"])
        self.assertIn("fundamentals:AAPL", result["critic_result"]["missing"])
        self.assertIn("collect_fundamentals:AAPL", result["critic_result"]["follow_up_tasks"])
        self.assertEqual(result["critic_result"]["confidence"], "Low")

    def test_critic_flags_comparison_fairness_issue(self):
        state = build_initial_state("Compare Nvidia vs AMD")
        state["tickers"] = ["NVDA", "AMD"]
        state["plan"]["required_evidence"] = ["news", "market", "sentiment"]
        state["evidence"]["news"]["NVDA"] = {"summary": "NVDA news"}
        state["evidence"]["news"]["AMD"] = {"summary": "AMD news"}
        state["evidence"]["market"]["NVDA"] = {"trend_label": "upward", "trend_7d": 0.04}
        state["evidence"]["sentiment"]["NVDA"] = {"sentiment": "positive", "score": 0.4}
        state["evidence"]["sentiment"]["AMD"] = {"sentiment": "neutral", "score": 0.0}

        result = run_critic_agent(state)

        self.assertFalse(result["critic_result"]["enough_evidence"])
        self.assertIn("asymmetric_market", result["critic_result"]["fairness_issues"])
        self.assertIn("resolve_asymmetric_market", result["critic_result"]["follow_up_tasks"])
        self.assertEqual(result["critic_result"]["confidence"], "Low")

    def test_critic_flags_conflicting_market_and_sentiment(self):
        state = build_complete_state_single()
        state["evidence"]["market"]["AAPL"] = {"trend_label": "downward", "trend_7d": -0.03}
        state["evidence"]["sentiment"]["AAPL"] = {"sentiment": "positive", "score": 0.42}

        result = run_critic_agent(state)

        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertIn("trend_vs_sentiment:AAPL", result["critic_result"]["conflicts"])
        self.assertEqual(result["critic_result"]["confidence"], "Medium")


if __name__ == "__main__":
    unittest.main()
