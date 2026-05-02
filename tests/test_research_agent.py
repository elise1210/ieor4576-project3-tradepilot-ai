import unittest
from unittest.mock import patch

from app.agents.research_agent import run_research_agent
from app.state import build_initial_state


def fake_news_skill(ticker: str, query: str, days: int = 7, max_items: int = 8, target_date=None) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} received mostly positive recent coverage.",
        "items": [
            {"headline": f"{ticker} positive headline 1"},
            {"headline": f"{ticker} positive headline 2"},
        ],
        "article_count": 2,
        "query_echo": query,
        "days_used": days,
        "max_items_used": max_items,
        "target_date_used": target_date,
    }


def fake_market_skill(ticker: str, lookback_days: int = 7, requested_date=None) -> dict:
    return {
        "ticker": ticker,
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
        "lookback_days_used": lookback_days,
        "requested_date_used": requested_date,
    }


def fake_fundamentals_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} has a stable business profile.",
        "market_cap_bucket": "large_cap",
    }


def fake_sentiment_skill(news_result: dict) -> dict:
    return {
        "sentiment": "positive",
        "score": 0.42,
        "dispersion": 0.10,
        "source_ticker": news_result.get("ticker"),
    }


def fake_chart_skill(ticker: str, evidence: dict, query: str) -> dict:
    return {
        "ticker": ticker,
        "kind": "seven_day_price_trend",
        "source_query": query,
        "charts": [
            {
                "id": f"{ticker.lower()}-price",
                "type": "line",
                "data": evidence.get("market", {}).get("history", []),
            }
        ],
    }


class ResearchAgentTests(unittest.TestCase):
    @patch.dict("os.environ", {"USE_LLM_RESEARCH": "false"}, clear=False)
    def test_research_agent_populates_evidence_with_fake_skills(self):
        state = build_initial_state("Should I buy Apple this week?")
        state["tickers"] = ["AAPL"]
        state["plan"]["required_evidence"] = ["news", "market", "fundamentals", "sentiment"]

        result = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "fundamentals": fake_fundamentals_skill,
                "sentiment": fake_sentiment_skill,
            },
        )

        self.assertEqual(result["gaps"], [])
        self.assertEqual(result["evidence"]["news"]["AAPL"]["ticker"], "AAPL")
        self.assertEqual(result["evidence"]["market"]["AAPL"]["trend_label"], "upward")
        self.assertEqual(
            result["evidence"]["fundamentals"]["AAPL"]["market_cap_bucket"],
            "large_cap",
        )
        self.assertEqual(result["evidence"]["sentiment"]["AAPL"]["sentiment"], "positive")
        self.assertEqual(result["metadata"]["research_mode"], "deterministic_only")
        self.assertEqual(len(result["metadata"]["research_plan_steps"]), 4)
        self.assertEqual(len(result["metadata"]["executed_research_steps"]), 4)

    def test_research_agent_records_missing_skill_gap(self):
        state = build_initial_state("Should I buy Apple this week?")
        state["tickers"] = ["AAPL"]
        state["plan"]["required_evidence"] = ["news", "market", "fundamentals", "sentiment"]

        result = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "sentiment": fake_sentiment_skill,
            },
        )

        self.assertIn("missing_skill:fundamentals:AAPL", result["gaps"])
        self.assertNotIn("AAPL", result["evidence"]["fundamentals"])

    def test_research_agent_handles_human_gate_before_running_skills(self):
        state = build_initial_state("Should I buy Apple?")
        state["tickers"] = ["AAPL"]
        state["needs_human"] = True
        state["clarification_question"] = "Short term or long term?"

        calls = {"news": 0}

        def tracking_news_skill(ticker: str, query: str) -> dict:
            calls["news"] += 1
            return fake_news_skill(ticker, query)

        result = run_research_agent(
            state,
            skills={"news": tracking_news_skill},
        )

        self.assertEqual(calls["news"], 0)
        self.assertIn("human_clarification_required", result["gaps"])

    def test_research_agent_collects_evidence_for_multiple_tickers(self):
        state = build_initial_state("Compare Nvidia vs AMD")
        state["tickers"] = ["NVDA", "AMD"]
        state["plan"]["required_evidence"] = ["news", "market", "sentiment"]

        result = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "sentiment": fake_sentiment_skill,
            },
        )

        self.assertEqual(set(result["evidence"]["news"].keys()), {"NVDA", "AMD"})
        self.assertEqual(set(result["evidence"]["market"].keys()), {"NVDA", "AMD"})
        self.assertEqual(set(result["evidence"]["sentiment"].keys()), {"NVDA", "AMD"})
        self.assertEqual(result["evidence"]["sentiment"]["NVDA"]["source_ticker"], "NVDA")
        self.assertEqual(result["evidence"]["sentiment"]["AMD"]["source_ticker"], "AMD")

    def test_research_agent_stores_chart_when_required(self):
        state = build_initial_state("What is AAPL stock price today?")
        state["tickers"] = ["AAPL"]
        state["plan"]["required_evidence"] = ["market", "chart"]

        def market_with_history(ticker: str) -> dict:
            result = fake_market_skill(ticker)
            result["history"] = [
                {"date": "2026-04-30", "close": 218.70},
                {"date": "2026-05-01", "close": 220.00},
            ]
            return result

        result = run_research_agent(
            state,
            skills={
                "market": market_with_history,
                "chart": fake_chart_skill,
            },
        )

        self.assertEqual(result["gaps"], [])
        self.assertEqual(len(result["evidence"]["charts"]), 1)
        self.assertEqual(
            result["evidence"]["charts"][0]["source_query"],
            "What is AAPL stock price today?",
        )

    def test_research_agent_builds_chart_from_existing_state_evidence(self):
        state = build_initial_state("What is AAPL stock price today?")
        state["tickers"] = ["AAPL"]
        state["plan"]["required_evidence"] = ["chart"]
        state["evidence"]["market"]["AAPL"] = {
            "ticker": "AAPL",
            "history": [
                {"date": "2026-04-30", "close": 218.70},
                {"date": "2026-05-01", "close": 220.00},
            ],
        }

        result = run_research_agent(
            state,
            skills={
                "chart": fake_chart_skill,
            },
        )

        self.assertEqual(result["gaps"], [])
        self.assertEqual(len(result["evidence"]["charts"]), 1)
        self.assertEqual(
            result["evidence"]["charts"][0]["charts"][0]["data"],
            [
                {"date": "2026-04-30", "close": 218.70},
                {"date": "2026-05-01", "close": 220.00},
            ],
        )

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_RESEARCH": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_RESEARCH_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_research.urllib.request.urlopen")
    def test_llm_research_plan_can_drive_fetch_parameters(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"steps\\":[{\\"skill\\":\\"news\\",\\"ticker\\":\\"AAPL\\",\\"params\\":{\\"days\\":14,\\"max_items\\":5}},{\\"skill\\":\\"market\\",\\"ticker\\":\\"AAPL\\",\\"params\\":{\\"lookback_days\\":10}},{\\"skill\\":\\"fundamentals\\",\\"ticker\\":\\"AAPL\\",\\"params\\":{}},{\\"skill\\":\\"sentiment\\",\\"ticker\\":\\"AAPL\\",\\"params\\":{},\\"depends_on\\":\\"news\\"}],\\"reasoning_brief\\":\\"Widen the news window and recent market context.\\"}"}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

        state = build_initial_state("Should I buy Apple this week?")
        state["tickers"] = ["AAPL"]
        state["plan"]["required_evidence"] = ["news", "market", "fundamentals", "sentiment"]

        result = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "fundamentals": fake_fundamentals_skill,
                "sentiment": fake_sentiment_skill,
            },
        )

        self.assertEqual(result["gaps"], [])
        self.assertEqual(result["evidence"]["news"]["AAPL"]["days_used"], 14)
        self.assertEqual(result["evidence"]["news"]["AAPL"]["max_items_used"], 5)
        self.assertEqual(result["evidence"]["market"]["AAPL"]["lookback_days_used"], 10)
        self.assertEqual(result["evidence"]["sentiment"]["AAPL"]["source_ticker"], "AAPL")
        self.assertEqual(result["metadata"]["research_mode"], "llm")
        self.assertEqual(
            result["metadata"]["research_reasoning_brief"],
            "Widen the news window and recent market context.",
        )
        self.assertEqual(result["metadata"]["research_plan_steps"][0]["params"]["days"], 14)

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_RESEARCH": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_RESEARCH_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_research.urllib.request.urlopen")
    def test_llm_research_falls_back_to_default_steps_on_invalid_plan(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"steps\\":[{\\"skill\\":\\"market\\",\\"ticker\\":\\"AAPL\\",\\"params\\":{}}]}"}}]}'

        mock_urlopen.return_value = FakeResponse()

        state = build_initial_state("Should I buy Apple this week?")
        state["tickers"] = ["AAPL"]
        state["plan"]["required_evidence"] = ["news", "market", "fundamentals", "sentiment"]

        result = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "fundamentals": fake_fundamentals_skill,
                "sentiment": fake_sentiment_skill,
            },
        )

        self.assertEqual(result["gaps"], [])
        self.assertIn("AAPL", result["evidence"]["news"])
        self.assertIn("AAPL", result["evidence"]["market"])
        self.assertIn("AAPL", result["evidence"]["fundamentals"])
        self.assertIn("AAPL", result["evidence"]["sentiment"])
        self.assertEqual(result["metadata"]["research_mode"], "deterministic_fallback")


if __name__ == "__main__":
    unittest.main()
