from datetime import datetime, timezone

import numpy as np
from transformers import pipeline


_finbert = None


def get_finbert():
    global _finbert

    if _finbert is None:
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
        )

    return _finbert


def analyze_news_sentiment(news_result: dict) -> dict:
    items = news_result.get("items", []) if news_result else []

    if not items:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "dispersion": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "article_count": 0,
            "summary": "No recent news available for sentiment analysis.",
        }

    finbert = get_finbert()

    scores = []
    weights = []
    now_ts = datetime.now(timezone.utc).timestamp()

    for item in items:
        headline = item.get("headline", "") or ""
        summary = item.get("summary", "") or ""
        text = f"{headline}. {summary}".strip()

        if not text:
            continue

        preds = finbert(text[:512])[0]
        best = max(preds, key=lambda d: d["score"])

        label = best["label"].lower()
        confidence = float(best["score"])

        if label == "positive":
            score = confidence
        elif label == "negative":
            score = -confidence
        else:
            score = 0.0

        weight = 1.0

        try:
            date_str = item.get("date", "")
            if "UTC" in date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M UTC").replace(
                    tzinfo=timezone.utc
                )
                age_hours = max(1.0, (now_ts - dt.timestamp()) / 3600.0)
                weight = 1.0 / (age_hours ** 0.5)
        except Exception:
            weight = 1.0

        scores.append(score)
        weights.append(weight)

    if not scores:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "dispersion": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "article_count": 0,
            "summary": "No usable news text available for sentiment analysis.",
        }

    scores = np.array(scores, dtype=float)
    weights = np.array(weights, dtype=float)

    avg = float(np.sum(scores * weights) / np.sum(weights))
    var = float(np.sum(weights * (scores - avg) ** 2) / np.sum(weights))
    dispersion = float(np.sqrt(var))

    positive_count = int(np.sum(scores > 0.1))
    negative_count = int(np.sum(scores < -0.1))
    neutral_count = int(len(scores) - positive_count - negative_count)

    if avg >= 0.20:
        sentiment = "positive"
    elif avg <= -0.20:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    summary = (
        f"Overall sentiment is {sentiment} "
        f"with score {avg:+.3f}. "
        f"Breakdown: {positive_count} positive, "
        f"{negative_count} negative, {neutral_count} neutral."
    )

    return {
        "sentiment": sentiment,
        "score": avg,
        "dispersion": dispersion,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "article_count": int(len(scores)),
        "summary": summary,
    }


def format_sentiment_output(sentiment_result: dict) -> str:
    if not sentiment_result:
        return "Sentiment unavailable."

    return (
        f"Sentiment: {sentiment_result.get('sentiment', 'neutral')}\n"
        f"Score: {sentiment_result.get('score', 0.0):+.3f}\n"
        f"Dispersion: {sentiment_result.get('dispersion', 0.0):.3f}\n"
        f"Articles: {sentiment_result.get('article_count', 0)}\n"
        f"{sentiment_result.get('summary', '')}"
    )