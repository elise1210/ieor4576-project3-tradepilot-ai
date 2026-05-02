import unittest

from app.agents.critic_agent import run_critic_agent
from app.agents.decision_agent import run_decision_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.research_agent import run_research_agent
from app.state import build_initial_state


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


class PipelineFlowTests(unittest.TestCase):
    def test_critic_can_send_flow_back_to_research_before_decision(self):
        state = build_initial_state("Should I buy Apple this week?")

        # Step 1: Planner creates the initial plan.
        state = run_planner_agent(state)
        self.assertEqual(state["tickers"], ["AAPL"])
        self.assertFalse(state["needs_human"])

        # Step 2: First research pass runs without market/fundamentals.
        state = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "sentiment": fake_sentiment_skill,
            },
        )
        self.assertIn("missing_skill:market:AAPL", state["gaps"])
        self.assertIn("missing_skill:fundamentals:AAPL", state["gaps"])

        # Step 3: Critic catches the blocking missing evidence and requests follow-up work.
        state = run_critic_agent(state)
        self.assertFalse(state["critic_result"]["enough_evidence"])
        self.assertIn("market:AAPL", state["critic_result"]["blocking_missing"])
        self.assertIn("fundamentals:AAPL", state["critic_result"]["missing"])
        self.assertIn("collect_market:AAPL", state["critic_result"]["follow_up_tasks"])
        self.assertIn(
            "collect_fundamentals:AAPL",
            state["critic_result"]["follow_up_tasks"],
        )

        # Step 4: Research runs again with the missing skill added.
        state = run_research_agent(
            state,
            skills={
                "news": fake_news_skill,
                "market": fake_market_skill,
                "fundamentals": fake_fundamentals_skill,
                "sentiment": fake_sentiment_skill,
            },
        )
        self.assertEqual(state["gaps"], [])
        self.assertEqual(
            state["evidence"]["fundamentals"]["AAPL"]["market_cap_bucket"],
            "large_cap",
        )

        # Step 5: Critic approves the completed evidence set.
        state = run_critic_agent(state)
        self.assertTrue(state["critic_result"]["enough_evidence"])
        self.assertEqual(state["critic_result"]["missing"], [])

        # Step 6: Decision runs only after the critic approves.
        state = run_decision_agent(state)
        self.assertEqual(state["decision"]["ticker"], "AAPL")
        self.assertEqual(state["decision"]["recommendation"], "BUY")
        self.assertIn("disclaimer", state["decision"])


if __name__ == "__main__":
    unittest.main()
