import unittest
from unittest.mock import patch

from app.agents.planner_agent import run_planner_agent
from app.state import build_initial_state


class PlannerAgentTests(unittest.TestCase):
    @patch.dict("os.environ", {"USE_LLM_PLANNER": "false"}, clear=False)
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
        self.assertEqual(result["metadata"]["planner_mode"], "deterministic_only")

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

    def test_irrelevant_query_is_marked_out_of_scope(self):
        state = build_initial_state("What is the weather in New York tomorrow?")

        result = run_planner_agent(state)

        self.assertTrue(result["guardrails"]["out_of_scope"])
        self.assertIn("informational stock analysis", result["guardrails"]["message"])

    def test_weekly_forecast_request_gets_scope_note(self):
        state = build_initial_state("Should I sell Tesla this week?")

        result = run_planner_agent(state)

        self.assertEqual(result["tickers"], ["TSLA"])
        self.assertIn("cannot forecast the full week", result["guardrails"]["scope_note"])

    def test_today_price_query_requests_chart_evidence(self):
        state = build_initial_state("What is AAPL stock price today?")

        result = run_planner_agent(state)

        self.assertIn("chart", result["plan"]["required_evidence"])
        self.assertFalse(result["needs_human"])

    def test_future_period_buy_query_does_not_request_chart_evidence(self):
        state = build_initial_state("Should I buy Apple this week?")

        result = run_planner_agent(state)

        self.assertNotIn("chart", result["plan"]["required_evidence"])

    @patch.dict("os.environ", {"USE_LLM_PLANNER": "false"}, clear=False)
    def test_news_summary_query_is_classified_as_explanation(self):
        state = build_initial_state("Summarize Nvidia news from yesterday")

        result = run_planner_agent(state)

        self.assertEqual(result["intent"], "explanation")
        self.assertEqual(result["tickers"], ["NVDA"])
        self.assertFalse(result["needs_human"])

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_PLANNER": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_PLANNER_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_planner.urllib.request.urlopen")
    def test_llm_planner_can_infer_fuzzy_company_reference(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"intent\\":\\"buy_sell_decision\\",\\"tickers\\":[\\"AAPL\\"],\\"time_horizon\\":\\"short_term\\",\\"needs_human\\":false,\\"clarification_question\\":null,\\"ticker_source\\":\\"llm_inference\\",\\"ticker_inference_confidence\\":\\"medium\\",\\"reasoning_brief\\":\\"The iPhone company refers to Apple.\\"}"}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

        state = build_initial_state("Should I buy the iPhone company this week?")
        result = run_planner_agent(state)

        self.assertEqual(result["intent"], "buy_sell_decision")
        self.assertEqual(result["tickers"], ["AAPL"])
        self.assertEqual(result["time_horizon"], "short_term")
        self.assertEqual(result["metadata"]["ticker_source"], "llm_inference")
        self.assertEqual(result["metadata"]["ticker_inference_confidence"], "medium")
        self.assertEqual(result["metadata"]["planner_mode"], "llm")
        self.assertEqual(
            result["metadata"]["planner_reasoning_brief"],
            "The iPhone company refers to Apple.",
        )
        self.assertFalse(result["needs_human"])

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_PLANNER": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_PLANNER_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_planner.urllib.request.urlopen")
    def test_llm_planner_falls_back_to_deterministic_on_invalid_json(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"not-json"}}]}'

        mock_urlopen.return_value = FakeResponse()

        state = build_initial_state("Should I buy Apple this week?")
        result = run_planner_agent(state)

        self.assertEqual(result["intent"], "buy_sell_decision")
        self.assertEqual(result["tickers"], ["AAPL"])
        self.assertEqual(result["time_horizon"], "short_term")
        self.assertEqual(result["metadata"]["ticker_source"], "company_name")
        self.assertEqual(result["metadata"]["planner_mode"], "deterministic_fallback")
        self.assertFalse(result["needs_human"])


if __name__ == "__main__":
    unittest.main()
