import unittest

from app.graph.nodes import apply_clarification_to_state
from app.state import build_initial_state


class GraphClarificationTests(unittest.TestCase):
    def test_time_horizon_clarification_resolves_short_term(self):
        state = build_initial_state("Should I buy Apple?")
        state["intent"] = "buy_sell_decision"
        state["tickers"] = ["AAPL"]
        state["needs_human"] = True
        state["clarification_question"] = "Short-term or long-term?"
        state["clarification_type"] = "time_horizon"
        state["clarification_options"] = [
            {"label": "Short-term", "value": "short_term"},
            {"label": "Long-term", "value": "long_term"},
        ]

        result = apply_clarification_to_state(state, "short_term")

        self.assertEqual(result["time_horizon"], "short_term")
        self.assertFalse(result["needs_human"])
        self.assertIsNone(result["clarification_question"])
        self.assertEqual(result["clarification_options"], [])

    def test_ticker_clarification_resolves_company_name(self):
        state = build_initial_state("Should I buy this stock right now?")
        state["intent"] = "buy_sell_decision"
        state["needs_human"] = True
        state["clarification_question"] = "Which company or ticker do you want me to analyze?"
        state["clarification_type"] = "ticker"

        result = apply_clarification_to_state(state, "Apple")

        self.assertEqual(result["tickers"], ["AAPL"])
        self.assertFalse(result["needs_human"])
        self.assertEqual(result["metadata"]["ticker_source"], "clarification")

    def test_invalid_ticker_reply_requests_clarification_again(self):
        state = build_initial_state("Should I buy this stock right now?")
        state["intent"] = "buy_sell_decision"
        state["needs_human"] = True
        state["clarification_question"] = "Which company or ticker do you want me to analyze?"
        state["clarification_type"] = "ticker"

        result = apply_clarification_to_state(state, "not sure")

        self.assertTrue(result["needs_human"])
        self.assertEqual(result["clarification_type"], "ticker")
        self.assertIn("Please reply with a stock ticker", result["clarification_question"])


if __name__ == "__main__":
    unittest.main()
