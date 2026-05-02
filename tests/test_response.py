import unittest

from app.response import apply_compliance_to_state, format_pipeline_answer
from app.state import build_initial_state


class ResponseLayerTests(unittest.TestCase):
    def test_apply_compliance_to_single_stock_decision(self):
        state = build_initial_state("Should I buy Apple this week?")
        state["tickers"] = ["AAPL"]
        state["critic_result"] = {"conflicts": []}
        state["decision"] = {
            "ticker": "AAPL",
            "recommendation": "BUY",
            "confidence": "Low",
            "risk_level": "High",
            "reasoning": ["Signal is mixed."],
            "drivers": {"key_driver": "Short-term momentum is positive."},
        }

        result = apply_compliance_to_state(state)
        decision = result["decision"]

        self.assertIn("disclaimer", decision)
        self.assertIn("uncertainty_notes", decision)
        self.assertFalse(decision["safety"]["is_financial_advice"])

    def test_apply_compliance_to_comparison_decision(self):
        state = build_initial_state("Compare Nvidia vs AMD")
        state["tickers"] = ["NVDA", "AMD"]
        state["critic_result"] = {"conflicts": ["trend_vs_sentiment:AMD"]}
        state["decision"] = {
            "type": "comparison",
            "tickers": ["NVDA", "AMD"],
            "comparison_summary": "AMD has the stronger score right now.",
            "per_ticker": {
                "NVDA": {
                    "ticker": "NVDA",
                    "recommendation": "HOLD",
                    "confidence": "Low",
                    "risk_level": "High",
                    "reasoning": ["Mixed signal."],
                    "drivers": {"key_driver": "Mixed data."},
                },
                "AMD": {
                    "ticker": "AMD",
                    "recommendation": "BUY",
                    "confidence": "Low",
                    "risk_level": "High",
                    "reasoning": ["Upward price momentum."],
                    "drivers": {"key_driver": "Strong price move."},
                },
            },
            "confidence": "Medium",
        }

        result = apply_compliance_to_state(state)
        decision = result["decision"]

        self.assertIn("disclaimer", decision)
        self.assertIn("uncertainty_notes", decision)
        self.assertIn("AMD", decision["per_ticker"])
        self.assertIn("uncertainty_notes", decision["per_ticker"]["AMD"])

    def test_format_pipeline_answer_includes_caution_notes(self):
        state = build_initial_state("Should I buy Apple this week?")
        state["intent"] = "buy_sell_decision"
        state["tickers"] = ["AAPL"]
        state["decision"] = {
            "ticker": "AAPL",
            "recommendation": "BUY",
            "confidence": "Low",
            "risk_level": "High",
            "reasoning": ["Signal is mixed."],
            "drivers": {"key_driver": "Short-term momentum is positive."},
            "disclaimer": "Informational only.",
            "uncertainty_notes": ["Evidence confidence is limited, so the result should be interpreted cautiously."],
        }

        answer = format_pipeline_answer(state)

        self.assertIn("Caution:", answer)
        self.assertIn("Informational only.", answer)

    def test_format_pipeline_answer_returns_research_summary_for_explanation_intent(self):
        state = build_initial_state("Summarize Nvidia news from yesterday")
        state["intent"] = "explanation"
        state["tickers"] = ["NVDA"]
        state["evidence"]["news"]["NVDA"] = {
            "summary": "NVDA news focused on China-related AI chip restrictions and data-center demand."
        }
        state["evidence"]["market"]["NVDA"] = {
            "trend_label": "sideways",
            "trend_7d": -0.006,
        }
        state["evidence"]["fundamentals"]["NVDA"] = {
            "summary": "NVIDIA remains a mega-cap semiconductor company with elevated beta."
        }
        state["critic_result"] = {
            "supporting_missing": ["sentiment:NVDA"],
        }

        answer = format_pipeline_answer(state)

        self.assertIn("NVDA news focused on China-related AI chip restrictions", answer)
        self.assertIn("Recent market context:", answer)
        self.assertIn("Company context:", answer)
        self.assertIn("Some supporting evidence was unavailable: sentiment.", answer)

    def test_format_pipeline_answer_includes_thin_evidence_note_for_explanation(self):
        state = build_initial_state("Why did NVDA move today?")
        state["intent"] = "explanation"
        state["tickers"] = ["NVDA"]
        state["metadata"]["stopped_reason"] = "iteration_budget_exhausted"
        state["metadata"]["critic_reasoning_brief"] = "The evidence is thin with only 2 articles and limited causal detail."
        state["evidence"]["news"]["NVDA"] = {
            "summary": "Two articles discussed bullish commentary, but neither identified a clear immediate catalyst."
        }
        state["critic_result"] = {
            "semantic_enough": False,
            "supporting_missing": ["sentiment:NVDA"],
        }

        answer = format_pipeline_answer(state)

        self.assertIn("Evidence note:", answer)
        self.assertIn("The evidence is thin with only 2 articles", answer)


if __name__ == "__main__":
    unittest.main()
