import unittest

from app.agents.planner_agent import run_planner_agent
from app.state import build_initial_state


class PlannerAgentTests(unittest.TestCase):
    def test_buy_query_with_company_name_infers_ticker(self):
        state = build_initial_state("Should I buy Apple this week?")

        result = run_planner_agent(state)

        self.assertEqual(result["intent"], "buy_sell_decision")
        self.assertEqual(result["tickers"], ["AAPL"])
        self.assertEqual(result["time_horizon"], "short_term")
        self.assertEqual(
            result["plan"]["required_evidence"],
            ["news", "market", "fundamentals", "sentiment"],
        )
        self.assertFalse(result["needs_human"])
        self.assertEqual(result["metadata"]["ticker_source"], "company_name")

    def test_compare_query_infers_both_tickers(self):
        state = build_initial_state("Compare Nvidia vs AMD")

        result = run_planner_agent(state)

        self.assertEqual(result["intent"], "comparison")
        self.assertEqual(result["tickers"], ["NVDA", "AMD"])
        self.assertFalse(result["needs_human"])

    def test_buy_query_without_horizon_requests_clarification(self):
        state = build_initial_state("Should I buy Apple?")

        result = run_planner_agent(state)

        self.assertEqual(result["tickers"], ["AAPL"])
        self.assertTrue(result["needs_human"])
        self.assertIn("short-term trading view", result["clarification_question"])

    def test_missing_company_requests_clarification(self):
        state = build_initial_state("Should I buy this stock right now?")

        result = run_planner_agent(state)

        self.assertEqual(result["tickers"], [])
        self.assertTrue(result["needs_human"])
        self.assertEqual(
            result["clarification_question"],
            "Which company or ticker do you want me to analyze?",
        )

    def test_explicit_ticker_wins_over_company_inference(self):
        state = build_initial_state("Should I buy MSFT this month?")

        result = run_planner_agent(state)

        self.assertEqual(result["tickers"], ["MSFT"])
        self.assertEqual(result["metadata"]["ticker_source"], "explicit_query")
        self.assertFalse(result["needs_human"])


if __name__ == "__main__":
    unittest.main()
