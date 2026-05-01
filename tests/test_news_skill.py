import unittest
from datetime import date, timedelta
from unittest.mock import patch

from app.skills.news import (
    extract_news_date,
    run_news_agent,
    summarize_news_with_openai,
    summarize_news_with_openai_result,
)


class NewsSkillDateTests(unittest.TestCase):
    def test_extract_news_date_accepts_explicit_date(self):
        self.assertEqual(extract_news_date("AAPL news on 2026-05-01"), date(2026, 5, 1))
        self.assertEqual(extract_news_date("AAPL news on 2026.12.01"), date(2026, 12, 1))
        self.assertEqual(extract_news_date("AAPL news on 2026/12/01"), date(2026, 12, 1))
        self.assertEqual(extract_news_date("AAPL news on December 1, 2026"), date(2026, 12, 1))

    def test_extract_news_date_accepts_today_terms(self):
        self.assertEqual(extract_news_date("AAPL news today"), date.today())
        self.assertEqual(extract_news_date("current AAPL news"), date.today())
        self.assertEqual(extract_news_date("latest AAPL news"), date.today())

    def test_extract_news_date_accepts_yesterday(self):
        self.assertEqual(
            extract_news_date("AAPL news yesterday"),
            date.today() - timedelta(days=1),
        )

    @patch("app.skills.news.finnhub_company_news_range")
    def test_run_news_agent_accepts_explicit_target_date(self, mock_news_range):
        mock_news_range.return_value = []

        result = run_news_agent(
            ticker="AAPL",
            user_query="AAPL news",
            target_date="2026.12.01",
        )

        self.assertEqual(result["requested_date"], "2026-12-01")
        self.assertEqual(result["start_date"], "2026-12-01")
        self.assertEqual(result["end_date"], "2026-12-01")
        mock_news_range.assert_called_once()

    @patch("app.skills.news.finnhub_company_news_range")
    def test_run_news_agent_preserves_summary_for_sentiment(self, mock_news_range):
        mock_news_range.return_value = [
            {
                "datetime": 1777636800,
                "headline": "AAPL beats expectations",
                "summary": "Revenue beats estimates.",
                "related": "AAPL",
                "source": "Demo",
                "url": "https://example.com",
            }
        ]

        result = run_news_agent(
            ticker="AAPL",
            target_date="2026-05-01",
        )

        self.assertEqual(result["items"][0]["summary"], "Revenue beats estimates.")

    @patch.dict("os.environ", {}, clear=True)
    def test_openai_summary_returns_none_without_api_key(self):
        result = summarize_news_with_openai(
            items=[
                {
                    "headline": "AAPL beats expectations",
                    "summary": "Revenue beats estimates.",
                }
            ],
            ticker="AAPL",
            user_query="AAPL news today",
        )

        self.assertIsNone(result)

    @patch.dict("os.environ", {}, clear=True)
    def test_openai_summary_result_reports_missing_api_key(self):
        result = summarize_news_with_openai_result(
            items=[
                {
                    "headline": "AAPL beats expectations",
                    "summary": "Revenue beats estimates.",
                }
            ],
            ticker="AAPL",
            user_query="AAPL news today",
        )

        self.assertIsNone(result["summary"])
        self.assertEqual(result["summary_source"], "rule_based")
        self.assertTrue(result["summary_fallback_used"])
        self.assertIn("OPENAI_API_KEY", result["summary_error"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    @patch("app.skills.news.urllib.request.urlopen")
    def test_openai_summary_uses_api_when_available(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"AAPL coverage focused on earnings strength."}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

        result = summarize_news_with_openai(
            items=[
                {
                    "headline": "AAPL beats expectations",
                    "summary": "Revenue beats estimates.",
                }
            ],
            ticker="AAPL",
            user_query="AAPL news today",
        )

        self.assertEqual(result, "AAPL coverage focused on earnings strength.")

    @patch.dict("os.environ", {}, clear=True)
    @patch("app.skills.news.finnhub_company_news_range")
    def test_run_news_agent_marks_rule_based_fallback(self, mock_news_range):
        mock_news_range.return_value = [
            {
                "datetime": 1777636800,
                "headline": "AAPL beats expectations",
                "summary": "Revenue beats estimates.",
                "related": "AAPL",
                "source": "Demo",
                "url": "https://example.com",
            }
        ]

        result = run_news_agent(
            ticker="AAPL",
            target_date="2026-05-01",
        )

        self.assertEqual(result["summary_source"], "rule_based")
        self.assertTrue(result["summary_fallback_used"])
        self.assertIn("OPENAI_API_KEY", result["summary_error"])


if __name__ == "__main__":
    unittest.main()
