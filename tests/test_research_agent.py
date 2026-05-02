import unittest

from app.agents.research_agent import run_research_agent
from app.state import build_initial_state


def fake_news_skill(ticker: str, query: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} received mostly positive recent coverage.",
        "items": [
            {"headline": f"{ticker} positive headline 1"},
            {"headline": f"{ticker} positive headline 2"},
        ],
        "article_count": 2,
        "query_echo": query,
    }


def fake_market_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
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


if __name__ == "__main__":
    unittest.main()
