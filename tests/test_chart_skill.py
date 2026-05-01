import unittest
from datetime import date

from app.skills.chart import build_chart_spec, run_chart_skill, should_show_chart_for_query


class ChartSkillTests(unittest.TestCase):
    def test_chart_skill_builds_frontend_ready_price_trend(self):
        evidence = {
            "market": {
                "ticker": "AAPL",
                "current_price": 220.0,
                "start_price_7d": 210.0,
                "trend_7d": 0.0476,
                "trend_label": "upward",
                "volatility": 0.018,
                "ma20": 215.0,
            },
        }

        result = run_chart_skill(ticker="aapl", evidence=evidence)

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["kind"], "seven_day_price_trend")
        self.assertTrue(result["charts"])
        self.assertIn("highlights", result)
        self.assertEqual(len(result["charts"]), 1)

        chart_ids = {chart["id"] for chart in result["charts"]}
        self.assertIn("aapl-price", chart_ids)

    def test_chart_skill_accepts_market_history_points(self):
        result = build_chart_spec(
            ticker="MSFT",
            evidence={
                "market": {
                    "history": [
                        {"date": "2026-04-29", "close": 410.25},
                        {"date": "2026-04-30", "close": 415.50},
                    ],
                }
            },
        )

        price_chart = result["charts"][0]

        self.assertEqual(price_chart["type"], "line")
        self.assertEqual(price_chart["data"][1]["close"], 415.5)

    def test_chart_skill_returns_error_for_empty_evidence(self):
        result = build_chart_spec(ticker="AAPL", evidence={})

        self.assertEqual(result["ticker"], "AAPL")
        self.assertIn("error", result)

    def test_chart_only_shows_for_today_price_query(self):
        evidence = {
            "market": {
                "history": [
                    {"date": "2026-04-30", "close": 218.70},
                    {"date": "2026-05-01", "close": 220.00},
                ],
            },
        }

        result = run_chart_skill(
            ticker="AAPL",
            evidence=evidence,
            query="What is AAPL price today?",
            reference_date=date(2026, 5, 1),
        )

        self.assertTrue(result["charts"])
        self.assertTrue(should_show_chart_for_query(
            "Should I buy AAPL on 2026-05-01?",
            reference_date=date(2026, 5, 1),
        ))

    def test_chart_does_not_show_for_historical_news_query(self):
        result = run_chart_skill(
            ticker="AAPL",
            evidence={"market": {"trend_7d": 0.02}},
            query="Summarize AAPL news from two months ago",
            reference_date=date(2026, 5, 1),
        )

        self.assertFalse(result["chart_available"])
        self.assertIn("reason", result)

    def test_chart_does_not_show_for_explicit_date_price_query(self):
        evidence = {
            "market": {
                "history": [
                    {"date": "2026-04-30", "close": 218.70},
                    {"date": "2026-05-01", "close": 220.00},
                ],
            },
        }

        result = run_chart_skill(
            ticker="AAPL",
            evidence=evidence,
            query="What was AAPL price on 2026.05.01?",
        )

        self.assertFalse(result["chart_available"])
        self.assertFalse(should_show_chart_for_query("AAPL price on Dec 1 2026"))


if __name__ == "__main__":
    unittest.main()
