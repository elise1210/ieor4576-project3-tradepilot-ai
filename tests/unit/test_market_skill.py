import unittest
from unittest.mock import patch

import pandas as pd

from app.skills.market import run_market_skill


def build_market_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 105, 106],
            "High": [101, 103, 104, 105, 106, 107, 109],
            "Low": [99, 100, 101, 102, 103, 104, 105],
            "Close": [100, 102, 101, 104, 105, 107, 108],
            "Volume": [1000, 1200, 1100, 1300, 1400, 1500, 1600],
        },
        index=pd.to_datetime([
            "2026-04-23",
            "2026-04-24",
            "2026-04-27",
            "2026-04-28",
            "2026-04-29",
            "2026-04-30",
            "2026-05-01",
        ]),
    )


class MarketSkillTests(unittest.TestCase):
    @patch("app.skills.yfinance_tool.yf.download")
    def test_market_skill_returns_history_for_chart(self, mock_download):
        mock_download.return_value = build_market_df()

        result = run_market_skill("AAPL", days=7)

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(len(result["history"]), 7)
        self.assertEqual(result["history"][-1]["date"], "2026-05-01")
        self.assertEqual(result["current_price"], 108)
        self.assertGreater(result["trend_7d"], 0)
        self.assertEqual(result["start_date"], "2026-04-23")
        self.assertEqual(result["end_date"], "2026-05-01")
        self.assertEqual(result["used_end_date"], "2026-05-01")

    @patch("app.skills.yfinance_tool.yf.download")
    def test_market_skill_accepts_end_date(self, mock_download):
        mock_download.return_value = build_market_df()

        result = run_market_skill("AAPL", days=3, end_date="2026-05-01")

        self.assertEqual(result["requested_date"], "2026-05-01")
        self.assertEqual(result["used_end_date"], "2026-05-01")
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["history"][0]["date"], "2026-04-29")

    @patch("app.skills.yfinance_tool.yf.download")
    def test_market_skill_accepts_llm_friendly_parameter_names(self, mock_download):
        mock_download.return_value = build_market_df()

        result = run_market_skill(
            "AAPL",
            lookback_days=3,
            requested_date="2026-05-01",
        )

        self.assertEqual(result["requested_date"], "2026-05-01")
        self.assertEqual(result["used_end_date"], "2026-05-01")
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["history"][0]["date"], "2026-04-29")


if __name__ == "__main__":
    unittest.main()
