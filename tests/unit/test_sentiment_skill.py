import unittest
from unittest.mock import patch

from app.skills.sentiment import analyze_news_sentiment, run_sentiment_skill


class FakeFinbert:
    def __call__(self, text):
        if "beats" in text.lower():
            return [[
                {"label": "positive", "score": 0.80},
                {"label": "neutral", "score": 0.15},
                {"label": "negative", "score": 0.05},
            ]]

        return [[
            {"label": "negative", "score": 0.70},
            {"label": "neutral", "score": 0.20},
            {"label": "positive", "score": 0.10},
        ]]


class SentimentSkillTests(unittest.TestCase):
    @patch("app.skills.sentiment.get_finbert")
    def test_analyze_news_sentiment_keeps_news_metadata(self, mock_get_finbert):
        mock_get_finbert.return_value = FakeFinbert()

        result = analyze_news_sentiment({
            "ticker": "AAPL",
            "requested_date": "2026-05-01",
            "start_date": "2026-05-01",
            "end_date": "2026-05-01",
            "items": [
                {
                    "headline": "AAPL beats expectations",
                    "summary": "Revenue beats estimates.",
                    "date": "2026-05-01 12:00 UTC",
                }
            ],
        })

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["requested_date"], "2026-05-01")
        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["positive_count"], 1)

    def test_run_sentiment_skill_requires_ticker_or_news_result(self):
        result = run_sentiment_skill()

        self.assertIn("error", result)
        self.assertEqual(result["sentiment"], "neutral")

    @patch("app.skills.sentiment.get_finbert")
    def test_analyze_news_sentiment_falls_back_when_model_unavailable(self, mock_get_finbert):
        mock_get_finbert.side_effect = RuntimeError("model download failed")

        result = analyze_news_sentiment({
            "ticker": "AAPL",
            "items": [
                {
                    "headline": "AAPL beats expectations",
                    "summary": "Revenue beats estimates.",
                    "date": "2026-05-01 12:00 UTC",
                }
            ],
        })

        self.assertEqual(result["sentiment"], "neutral")
        self.assertFalse(result["model_available"])
        self.assertEqual(result["article_count"], 1)
        self.assertIn("Sentiment model unavailable", result["summary"])

    @patch("app.skills.sentiment.run_news_skill")
    @patch("app.skills.sentiment.get_finbert")
    def test_run_sentiment_skill_can_fetch_news_for_target_date(self, mock_get_finbert, mock_run_news):
        mock_get_finbert.return_value = FakeFinbert()
        mock_run_news.return_value = {
            "ticker": "AAPL",
            "requested_date": "2026-05-01",
            "start_date": "2026-05-01",
            "end_date": "2026-05-01",
            "items": [
                {
                    "headline": "AAPL beats expectations",
                    "summary": "Revenue beats estimates.",
                    "date": "2026-05-01 12:00 UTC",
                }
            ],
        }

        result = run_sentiment_skill(
            ticker="AAPL",
            target_date="2026-05-01",
        )

        mock_run_news.assert_called_once()
        self.assertEqual(result["requested_date"], "2026-05-01")
        self.assertEqual(result["sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
