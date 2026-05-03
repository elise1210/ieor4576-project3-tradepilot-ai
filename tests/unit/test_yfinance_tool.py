import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from app.skills.yfinance_tool import (
    fetch_daily_prices_until,
    fetch_ohlcv_on_date,
    normalize_reference_date,
)


def build_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 102.5, 103.5],
            "Volume": [1000, 1200, 1300],
        },
        index=pd.to_datetime(["2026-04-29", "2026-04-30", "2026-05-01"]),
    )


class YfinanceToolTests(unittest.TestCase):
    def test_normalize_reference_date_accepts_today_words(self):
        self.assertEqual(normalize_reference_date("today"), date.today())
        self.assertEqual(normalize_reference_date("current"), date.today())
        self.assertEqual(normalize_reference_date("2026-05-01"), date(2026, 5, 1))
        self.assertEqual(normalize_reference_date("2026.12.01"), date(2026, 12, 1))
        self.assertEqual(normalize_reference_date("2026/12/01"), date(2026, 12, 1))

    @patch("app.skills.yfinance_tool.yf.download")
    def test_fetch_daily_prices_until_returns_trading_window(self, mock_download):
        mock_download.return_value = build_price_df()

        result = fetch_daily_prices_until(
            ticker="AAPL",
            end_date="2026-05-01",
            days=2,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["date"], "2026-04-30")
        self.assertEqual(result[1]["close"], 103.5)

    @patch("app.skills.yfinance_tool.yf.download")
    def test_fetch_ohlcv_on_date_falls_back_to_previous_trading_day(self, mock_download):
        mock_download.return_value = build_price_df()

        result = fetch_ohlcv_on_date(
            ticker="AAPL",
            target_date="2026-05-02",
            fields=["Price", "Volume"],
        )

        self.assertEqual(result["used_date"], "2026-05-01")
        self.assertEqual(result["Price"], 103.5)
        self.assertEqual(result["Volume"], 1300)


if __name__ == "__main__":
    unittest.main()
