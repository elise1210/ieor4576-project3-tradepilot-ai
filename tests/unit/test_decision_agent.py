import unittest
from unittest.mock import patch

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
    @patch.dict("os.environ", {"USE_LLM_DECISION": "false"}, clear=False)
    def test_decision_agent_creates_single_ticker_decision(self):
        state = build_single_decision_state()

        result = run_decision_agent(state)

        self.assertEqual(result["decision"]["ticker"], "AAPL")
        self.assertIn(result["decision"]["recommendation"], {"BUY", "HOLD", "SELL"})
        self.assertEqual(result["decision"]["recommendation"], "BUY")
        self.assertIn("disclaimer", result["decision"])
        self.assertTrue(result["decision"]["reasoning"])
        self.assertEqual(result["metadata"]["decision_mode"], "deterministic_only")

    @patch.dict("os.environ", {"USE_LLM_DECISION": "false"}, clear=False)
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
        self.assertEqual(result["metadata"]["decision_mode"], "deterministic_only")

    @patch.dict("os.environ", {"USE_LLM_DECISION": "false"}, clear=False)
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

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_DECISION": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_DECISION_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_decision.urllib.request.urlopen")
    def test_llm_decision_can_rewrite_single_stock_reasoning(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"reasoning\\":[\\"The recent evidence leans bullish because earnings and sentiment are both strong.\\",\\"Short-term price action has also been supportive.\\",\\"Risk remains contained relative to the current signal.\\"],\\"key_driver\\":\\"Apple\\u2019s strong earnings and buyback announcement are the main drivers of the current signal.\\",\\"reasoning_brief\\":\\"LLM synthesized a cleaner single-stock explanation.\\"}"}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

        state = build_single_decision_state()
        result = run_decision_agent(state)

        self.assertEqual(result["decision"]["recommendation"], "BUY")
        self.assertEqual(result["metadata"]["decision_mode"], "llm")
        self.assertEqual(
            result["metadata"]["decision_reasoning_brief"],
            "LLM synthesized a cleaner single-stock explanation.",
        )
        self.assertEqual(
            result["decision"]["drivers"]["key_driver"],
            "Apple’s strong earnings and buyback announcement are the main drivers of the current signal.",
        )
        self.assertEqual(
            result["decision"]["reasoning"][0],
            "The recent evidence leans bullish because earnings and sentiment are both strong.",
        )

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_DECISION": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_DECISION_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_decision.urllib.request.urlopen")
    def test_llm_decision_can_rewrite_comparison_summary(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"comparison_summary\\":\\"NVDA currently looks stronger overall, while AMD shows more mixed evidence.\\",\\"per_ticker\\":{\\"NVDA\\":{\\"reasoning\\":[\\"NVDA benefits from stronger sentiment and a better near-term score.\\"],\\"key_driver\\":\\"AI demand remains the main support for NVDA.\\"},\\"AMD\\":{\\"reasoning\\":[\\"AMD has upside momentum, but the supporting evidence is more mixed.\\"],\\"key_driver\\":\\"AMD\\u2019s momentum is strong, but the signal is less clean.\\"}},\\"reasoning_brief\\":\\"LLM clarified the comparison wording.\\"}"}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

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
        self.assertEqual(result["metadata"]["decision_mode"], "llm")
        self.assertEqual(
            result["decision"]["comparison_summary"],
            "NVDA currently looks stronger overall, while AMD shows more mixed evidence.",
        )
        self.assertEqual(
            result["decision"]["per_ticker"]["NVDA"]["drivers"]["key_driver"],
            "AI demand remains the main support for NVDA.",
        )
        self.assertEqual(
            result["decision"]["per_ticker"]["AMD"]["reasoning"][0],
            "AMD has upside momentum, but the supporting evidence is more mixed.",
        )

    @patch.dict(
        "os.environ",
        {
            "USE_LLM_DECISION": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_DECISION_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    @patch("app.agents.llm_decision.urllib.request.urlopen")
    def test_llm_decision_falls_back_on_invalid_json(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"not-json"}}]}'

        mock_urlopen.return_value = FakeResponse()

        state = build_single_decision_state()
        result = run_decision_agent(state)

        self.assertEqual(result["decision"]["recommendation"], "BUY")
        self.assertEqual(result["metadata"]["decision_mode"], "deterministic_fallback")


if __name__ == "__main__":
    unittest.main()
