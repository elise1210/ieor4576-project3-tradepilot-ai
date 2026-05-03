import unittest
from unittest.mock import patch

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
    @patch.dict("os.environ", {"USE_LLM_CRITIC": "false"}, clear=False)
    def test_critic_marks_complete_single_ticker_state_as_enough(self):
        state = build_complete_state_single()

        result = run_critic_agent(state)

        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertEqual(result["critic_result"]["missing"], [])
        self.assertEqual(result["critic_result"]["fairness_issues"], [])
        self.assertEqual(result["critic_result"]["conflicts"], [])
        self.assertEqual(result["critic_result"]["confidence"], "High")
        self.assertEqual(result["metadata"]["critic_mode"], "deterministic_only")

    @patch.dict("os.environ", {"USE_LLM_CRITIC": "false"}, clear=False)
    def test_critic_flags_missing_fundamentals(self):
        state = build_complete_state_single()
        state["evidence"]["fundamentals"]["AAPL"] = {}

        result = run_critic_agent(state)

        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertIn("fundamentals:AAPL", result["critic_result"]["missing"])
        self.assertEqual(result["critic_result"]["blocking_missing"], [])
        self.assertIn("fundamentals:AAPL", result["critic_result"]["supporting_missing"])
        self.assertIn("collect_fundamentals:AAPL", result["critic_result"]["follow_up_tasks"])
        self.assertEqual(result["critic_result"]["confidence"], "Medium")

    @patch.dict("os.environ", {"USE_LLM_CRITIC": "false"}, clear=False)
    def test_critic_blocks_single_ticker_when_market_missing(self):
        state = build_complete_state_single()
        state["evidence"]["market"]["AAPL"] = {}

        result = run_critic_agent(state)

        self.assertFalse(result["critic_result"]["enough_evidence"])
        self.assertIn("market:AAPL", result["critic_result"]["blocking_missing"])
        self.assertEqual(result["critic_result"]["confidence"], "Low")

    @patch.dict("os.environ", {"USE_LLM_CRITIC": "false"}, clear=False)
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

    @patch.dict("os.environ", {"USE_LLM_CRITIC": "false"}, clear=False)
    def test_critic_flags_conflicting_market_and_sentiment(self):
        state = build_complete_state_single()
        state["evidence"]["market"]["AAPL"] = {"trend_label": "downward", "trend_7d": -0.03}
        state["evidence"]["sentiment"]["AAPL"] = {"sentiment": "positive", "score": 0.42}

        result = run_critic_agent(state)

        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertIn("trend_vs_sentiment:AAPL", result["critic_result"]["conflicts"])
        self.assertEqual(result["critic_result"]["confidence"], "Medium")

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_CRITIC": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_CRITIC_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_critic.urllib.request.urlopen")
    def test_llm_critic_can_mark_structurally_complete_state_as_semantically_thin(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"semantic_enough\\":false,\\"quality_issues\\":[\\"news_thin\\"],\\"follow_up_steps\\":[{\\"skill\\":\\"news\\",\\"ticker\\":\\"AAPL\\",\\"params\\":{\\"days\\":14,\\"max_items\\":8}}],\\"reasoning_brief\\":\\"Current evidence is too thin for a confident recommendation.\\"}"}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

        state = build_complete_state_single()
        result = run_critic_agent(state)

        self.assertFalse(result["critic_result"]["enough_evidence"])
        self.assertFalse(result["critic_result"]["semantic_enough"])
        self.assertEqual(result["critic_result"]["quality_issues"], ["news_thin"])
        self.assertEqual(
            result["critic_result"]["llm_follow_up_steps"],
            [
                {
                    "skill": "news",
                    "ticker": "AAPL",
                    "params": {"days": 14, "max_items": 8},
                }
            ],
        )
        self.assertEqual(result["critic_result"]["confidence"], "Low")
        self.assertEqual(result["metadata"]["critic_mode"], "llm")
        self.assertEqual(
            result["metadata"]["critic_reasoning_brief"],
            "Current evidence is too thin for a confident recommendation.",
        )

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_CRITIC": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_CRITIC_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_critic.urllib.request.urlopen")
    def test_llm_critic_falls_back_on_invalid_json(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"not-json"}}]}'

        mock_urlopen.return_value = FakeResponse()

        state = build_complete_state_single()
        result = run_critic_agent(state)

        self.assertTrue(result["critic_result"]["enough_evidence"])
        self.assertIsNone(result["critic_result"]["semantic_enough"])
        self.assertEqual(result["critic_result"]["quality_issues"], [])
        self.assertEqual(result["critic_result"]["llm_follow_up_steps"], [])
        self.assertEqual(result["metadata"]["critic_mode"], "deterministic_fallback")
        self.assertIsNone(result["metadata"]["critic_reasoning_brief"])


if __name__ == "__main__":
    unittest.main()
