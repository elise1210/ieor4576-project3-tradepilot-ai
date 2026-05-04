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

    def test_format_pipeline_answer_prioritizes_structured_sentiment(self):
        state = build_initial_state("What was the sentiment of Nvidia on 2026-04-02")
        state["intent"] = "general_research"
        state["tickers"] = ["NVDA"]
        state["plan"]["required_evidence"] = ["news", "sentiment"]
        state["evidence"]["news"]["NVDA"] = {
            "summary": "Nvidia news discussed AI chip demand and partnerships."
        }
        state["evidence"]["sentiment"]["NVDA"] = {
            "sentiment": "positive",
            "score": 0.42,
            "article_count": 5,
            "positive_count": 3,
            "neutral_count": 1,
            "negative_count": 1,
            "requested_date": "2026-04-02",
        }

        answer = format_pipeline_answer(state)

        self.assertIn("Sentiment for NVDA on 2026-04-02: positive.", answer)
        self.assertIn("Score: +0.420", answer)
        self.assertIn("Breakdown: 3 positive, 1 neutral, 1 negative.", answer)
        self.assertIn("Nvidia news discussed AI chip demand", answer)

    def test_format_pipeline_answer_prioritizes_requested_price(self):
        state = build_initial_state("What was the price of Nvidia on 2026-04-02")
        state["intent"] = "general_research"
        state["tickers"] = ["NVDA"]
        state["plan"]["required_evidence"] = ["market", "chart"]
        state["evidence"]["market"]["NVDA"] = {
            "ticker": "NVDA",
            "current_price": 188.25,
            "requested_date": "2026-04-02",
            "used_end_date": "2026-04-02",
            "trend_label": "sideways",
            "trend_7d": -0.0072,
        }

        answer = format_pipeline_answer(state)

        self.assertIn("NVDA's closing price on 2026-04-02 was $188.25.", answer)
        self.assertIn("Recent market context:", answer)

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

    def test_format_pipeline_answer_falls_back_to_summary_when_decision_missing(self):
        state = build_initial_state("Should I buy the iPhone company this week?")
        state["intent"] = "buy_sell_decision"
        state["tickers"] = ["AAPL"]
        state["metadata"]["stopped_reason"] = "iteration_budget_exhausted"
        state["metadata"]["critic_reasoning_brief"] = (
            "The sentiment is mixed and the news coverage is thin, which may affect the confidence in the buy/sell decision."
        )
        state["guardrails"]["scope_note"] = (
            "I cannot forecast the full week. I can provide a daily informational buy/hold/sell signal based on the latest available data."
        )
        state["evidence"]["news"]["AAPL"] = {
            "summary": "Apple reported strong results and a large buyback, but supply concerns remain."
        }
        state["evidence"]["market"]["AAPL"] = {
            "current_price": 280.14,
            "trend_label": "upward",
            "trend_7d": 0.0245,
        }
        state["evidence"]["sentiment"]["AAPL"] = {
            "sentiment": "positive",
            "score": 0.438,
            "article_count": 6,
        }
        state["evidence"]["fundamentals"]["AAPL"] = {
            "summary": "Apple is a mega-cap technology company with elevated valuation but strong scale."
        }
        state["critic_result"] = {
            "semantic_enough": False,
            "supporting_missing": [],
        }

        answer = format_pipeline_answer(state)

        self.assertIn("I could not produce a reliable final recommendation within the iteration budget.", answer)
        self.assertIn("Why no final recommendation:", answer)
        self.assertIn("Current evidence summary:", answer)
        self.assertIn("Sentiment is positive with score +0.438 across 6 article(s).", answer)
        self.assertIn("Recent price trend is upward over 7 trading days (+2.45%).", answer)
        self.assertIn("News snapshot: Apple reported strong results and a large buyback, but supply concerns remain.", answer)
        self.assertIn("Company context:", answer)
        self.assertIn("I cannot forecast the full week.", answer)
        self.assertNotIn("Evidence note:", answer)

    def test_apply_compliance_adds_provisional_preview_for_exhausted_decision(self):
        state = build_initial_state("Should I buy Apple this week?")
        state["intent"] = "buy_sell_decision"
        state["tickers"] = ["AAPL"]
        state["metadata"]["stopped_reason"] = "iteration_budget_exhausted"
        state["evidence"]["market"]["AAPL"] = {
            "trend_7d": 0.0245,
            "trend_label": "upward",
            "volatility": 0.041,
        }
        state["evidence"]["sentiment"]["AAPL"] = {
            "sentiment": "positive",
            "score": 0.438,
            "dispersion": 0.12,
        }

        safe_state = apply_compliance_to_state(state)
        preview = safe_state.get("decision_preview", {})
        answer = format_pipeline_answer(safe_state)

        self.assertEqual(preview.get("type"), "provisional_single")
        self.assertEqual(preview.get("risk_level"), "High")
        self.assertEqual(preview.get("confidence"), "Low")
        self.assertIn("Provisional risk level from the available evidence is High.", answer)


if __name__ == "__main__":
    unittest.main()
