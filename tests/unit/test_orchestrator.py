import unittest

from app.orchestrator import run_tradepilot_pipeline


def fake_news_skill(ticker: str, query: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} received mostly positive recent coverage.",
        "items": [{"headline": f"{ticker} positive headline"}],
        "article_count": 1,
        "query_echo": query,
    }


def fake_market_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
    }


def fake_sentiment_skill(news_result: dict) -> dict:
    return {
        "sentiment": "positive",
        "score": 0.42,
        "dispersion": 0.10,
        "source_ticker": news_result.get("ticker"),
    }


def fake_fundamentals_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} has a stable large-cap business profile.",
        "market_cap_bucket": "large_cap",
    }


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_stops_for_human_clarification(self):
        result = run_tradepilot_pipeline("Should I buy Apple?")

        self.assertTrue(result["needs_human"])
        self.assertEqual(result["metadata"]["stopped_reason"], "human_clarification_required")
        self.assertEqual(result["metadata"]["iterations_used"], 0)
        self.assertIsNone(result["decision"])

    def test_orchestrator_stops_for_out_of_scope_question(self):
        result = run_tradepilot_pipeline("What is the weather in New York tomorrow?")

        self.assertTrue(result["guardrails"]["out_of_scope"])
        self.assertEqual(result["metadata"]["stopped_reason"], "out_of_scope")
        self.assertEqual(result["metadata"]["iterations_used"], 0)
        self.assertIsNone(result["decision"])

    def test_orchestrator_completes_full_loop_with_realistic_fake_skills(self):
        result = run_tradepilot_pipeline(
            query="Should I buy Apple this week?",
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "fundamentals": fake_fundamentals_skill,
                "sentiment": fake_sentiment_skill,
            },
        )

        self.assertFalse(result["needs_human"])
        self.assertEqual(result["metadata"]["stopped_reason"], "decision_completed")
        self.assertEqual(result["metadata"]["iterations_used"], 1)
        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertEqual(result["decision"]["ticker"], "AAPL")
        self.assertIn(result["decision"]["recommendation"], {"BUY", "HOLD", "SELL"})

    def test_orchestrator_stops_when_iteration_budget_exhausted(self):
        result = run_tradepilot_pipeline(
            query="Should I buy Apple this week?",
            skills={
                "news": fake_news_skill,
                "sentiment": fake_sentiment_skill,
            },
            max_iterations=2,
        )

        self.assertEqual(result["metadata"]["stopped_reason"], "iteration_budget_exhausted")
        self.assertEqual(result["metadata"]["iterations_used"], 2)
        self.assertFalse(result["critic_result"]["enough_evidence"])
        self.assertIsNone(result["decision"])

    def test_orchestrator_completes_research_without_decision_for_explanation_intent(self):
        result = run_tradepilot_pipeline(
            query="Summarize Nvidia news from yesterday",
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "fundamentals": fake_fundamentals_skill,
            },
        )

        self.assertEqual(result["intent"], "explanation")
        self.assertEqual(result["metadata"]["stopped_reason"], "research_completed")
        self.assertEqual(result["metadata"]["iterations_used"], 1)
        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertIsNone(result["decision"])


if __name__ == "__main__":
    unittest.main()
