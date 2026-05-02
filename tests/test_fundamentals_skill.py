import unittest
from unittest.mock import patch

from app.skills.fundamentals import run_fundamentals_skill


class FundamentalsSkillTests(unittest.TestCase):
    @patch("app.skills.fundamentals.finnhub_fundamentals_basic")
    @patch("app.skills.fundamentals.finnhub_company_profile")
    def test_fundamentals_merges_profile_and_metrics(
        self,
        mock_profile,
        mock_metrics,
    ):
        mock_profile.return_value = {
            "ticker": "AAPL",
            "name": "Apple Inc",
            "country": "US",
            "currency": "USD",
            "exchange": "NASDAQ",
            "finnhubIndustry": "Technology",
            "ipo": "1980-12-12",
            "marketCapitalization": 275000,
            "weburl": "https://www.apple.com",
        }
        mock_metrics.return_value = {
            "ticker": "AAPL",
            "marketCapitalization": 275000,
            "peTTM": 28.7,
            "pb": 35.2,
            "epsTTM": 6.43,
            "dividendYieldIndicatedAnnual": 0.47,
            "52WeekHigh": 230.0,
            "52WeekLow": 165.0,
            "52WeekPriceReturnDaily": 0.18,
            "beta": 1.12,
        }

        result = run_fundamentals_skill("aapl")

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["company_name"], "Apple Inc")
        self.assertEqual(result["industry"], "Technology")
        self.assertEqual(result["market_cap_bucket"], "mega_cap")
        self.assertAlmostEqual(result["pe_ttm"], 28.7)
        self.assertAlmostEqual(result["beta"], 1.12)
        self.assertIn("Apple Inc (AAPL) is a mega-cap company", result["summary"])
        self.assertIn("Technology", result["summary"])

    @patch("app.skills.fundamentals.finnhub_fundamentals_basic")
    @patch("app.skills.fundamentals.finnhub_company_profile")
    def test_fundamentals_returns_error_when_both_sources_fail(
        self,
        mock_profile,
        mock_metrics,
    ):
        mock_profile.return_value = {"ticker": "AAPL", "error": "profile failed"}
        mock_metrics.return_value = {"ticker": "AAPL", "error": "metrics failed"}

        result = run_fundamentals_skill("AAPL")

        self.assertEqual(result["ticker"], "AAPL")
        self.assertIn("profile_error=profile failed", result["error"])
        self.assertIn("fundamentals_error=metrics failed", result["error"])


if __name__ == "__main__":
    unittest.main()
