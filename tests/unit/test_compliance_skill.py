import unittest

from app.skills.compliance import apply_compliance, run_compliance_skill


class ComplianceSkillTests(unittest.TestCase):
    def test_compliance_adds_daily_signal_disclaimer(self):
        decision = {
            "ticker": "AAPL",
            "recommendation": "BUY",
            "confidence": "High",
            "reasoning": ["Positive sentiment and upward price trend."],
        }

        result = apply_compliance(decision)

        self.assertIn("disclaimer", result)
        self.assertIn("informational daily signal", result["disclaimer"])
        self.assertFalse(result["safety"]["is_financial_advice"])
        self.assertFalse(result["safety"]["predicts_future_prices"])

    def test_compliance_adds_low_confidence_and_conflict_notes(self):
        decision = {
            "ticker": "AAPL",
            "recommendation": "HOLD",
            "confidence": "Low",
        }
        state = {
            "critic_result": {
                "conflicts": ["trend_vs_sentiment:AAPL"],
            }
        }

        result = apply_compliance(decision, state=state)

        joined_notes = " ".join(result["uncertainty_notes"])
        self.assertIn("confidence is limited", joined_notes)
        self.assertIn("mixed or conflicting", joined_notes)

    def test_compliance_keeps_existing_disclaimer(self):
        decision = {
            "recommendation": "SELL",
            "disclaimer": "Existing disclaimer.",
        }

        result = apply_compliance(decision)

        self.assertEqual(result["disclaimer"], "Existing disclaimer.")

    def test_compliance_uses_comparison_disclaimer(self):
        decision = {
            "type": "comparison",
            "tickers": ["NVDA", "AMD"],
            "confidence": "Medium",
        }

        result = run_compliance_skill(decision)

        self.assertIn("comparison is informational", result["disclaimer"])

    def test_compliance_handles_plain_text_response(self):
        result = apply_compliance("AAPL will definitely rise tomorrow.")

        self.assertEqual(result["text"], "AAPL will definitely rise tomorrow.")
        self.assertTrue(result["safety"]["unsafe_language_detected"])
        self.assertTrue(result["uncertainty_notes"])


if __name__ == "__main__":
    unittest.main()
