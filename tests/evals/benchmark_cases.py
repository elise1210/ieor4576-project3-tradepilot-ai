FAKE_SKILL_NAMES = ("news", "market", "fundamentals", "sentiment", "chart")


def fake_news_skill(
    ticker: str,
    query: str,
    days: int = 7,
    max_items: int = 8,
    target_date=None,
) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} received mostly positive recent coverage.",
        "items": [
            {"headline": f"{ticker} positive headline 1"},
            {"headline": f"{ticker} positive headline 2"},
        ],
        "article_count": 2,
        "query_echo": query,
        "days_used": days,
        "max_items_used": max_items,
        "target_date_used": target_date,
    }


def fake_market_skill(
    ticker: str,
    lookback_days: int = 7,
    requested_date=None,
) -> dict:
    return {
        "ticker": ticker,
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
        "current_price": 123.45,
        "lookback_days_used": lookback_days,
        "requested_date_used": requested_date,
        "history": [
            {"date": "2026-04-30", "close": 120.00},
            {"date": "2026-05-01", "close": 123.45},
        ],
    }


def fake_fundamentals_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} has a stable business profile.",
        "market_cap_bucket": "large_cap",
    }


def fake_sentiment_skill(news_result: dict) -> dict:
    return {
        "sentiment": "positive",
        "score": 0.42,
        "dispersion": 0.10,
        "source_ticker": news_result.get("ticker"),
    }


def fake_chart_skill(
    ticker: str,
    evidence: dict,
    query: str,
    reference_date=None,
) -> dict:
    market_history = evidence.get("market", {}).get("history", [])
    return {
        "ticker": ticker,
        "chart_id": f"{ticker.lower()}-price-trend",
        "kind": "seven_day_price_trend",
        "source_query": query,
        "reference_date_used": reference_date,
        "charts": [
            {
                "id": f"{ticker.lower()}-price",
                "type": "line",
                "data": market_history,
            }
        ],
    }


FAKE_SKILLS = {
    "news": fake_news_skill,
    "market": fake_market_skill,
    "fundamentals": fake_fundamentals_skill,
    "sentiment": fake_sentiment_skill,
    "chart": fake_chart_skill,
}


BENCHMARK_CASES = [
    {
        "id": "buy_aapl_week",
        "query": "Should I buy Apple this week?",
        "expected_intent": "buy_sell_decision",
        "expected_tickers": ["AAPL"],
        "required_tools": ["news", "market", "fundamentals", "sentiment"],
        "forbidden_tools": ["chart"],
        "required_evidence": ["news", "market", "fundamentals", "sentiment"],
        "expected_stop_reason": "decision_completed",
    },
    {
        "id": "compare_nvda_amd",
        "query": "Compare Nvidia vs AMD",
        "expected_intent": "comparison",
        "expected_tickers": ["NVDA", "AMD"],
        "required_tools": ["news", "market", "fundamentals", "sentiment"],
        "forbidden_tools": ["chart"],
        "required_evidence": ["news", "market", "fundamentals", "sentiment"],
        "expected_stop_reason": "decision_completed",
    },
    {
        "id": "summarize_nvda_news",
        "query": "Summarize Nvidia news from yesterday",
        "expected_intent": "explanation",
        "expected_tickers": ["NVDA"],
        "required_tools": ["news", "market", "fundamentals"],
        "forbidden_tools": ["sentiment", "chart"],
        "required_evidence": ["news", "market", "fundamentals"],
        "expected_stop_reason": "research_completed",
    },
    {
        "id": "nvda_sentiment_specific_date",
        "query": "What was the sentiment of Nvidia on 2026-04-02",
        "expected_intent": "general_research",
        "expected_tickers": ["NVDA"],
        "required_tools": ["news", "sentiment"],
        "forbidden_tools": ["market", "fundamentals", "chart"],
        "required_evidence": ["news", "sentiment"],
        "expected_stop_reason": "research_completed",
    },
    {
        "id": "aapl_price_today",
        "query": "What is AAPL stock price today?",
        "expected_intent": "general_research",
        "expected_tickers": ["AAPL"],
        "required_tools": ["market", "chart"],
        "forbidden_tools": ["news", "fundamentals", "sentiment"],
        "required_evidence": ["market", "chart"],
        "expected_stop_reason": "research_completed",
    },
    {
        "id": "nvda_price_specific_date",
        "query": "What was the price of Nvidia on 2026-04-02",
        "expected_intent": "general_research",
        "expected_tickers": ["NVDA"],
        "required_tools": ["market", "chart"],
        "forbidden_tools": ["news", "fundamentals", "sentiment"],
        "required_evidence": ["market", "chart"],
        "expected_stop_reason": "research_completed",
    },
    {
        "id": "clarify_buy_apple",
        "query": "Should I buy Apple?",
        "expected_intent": "buy_sell_decision",
        "expected_tickers": ["AAPL"],
        "required_tools": [],
        "forbidden_tools": list(FAKE_SKILL_NAMES),
        "required_evidence": [],
        "expected_stop_reason": "human_clarification_required",
    },
    {
        "id": "weather_out_of_scope",
        "query": "What is the weather in New York tomorrow?",
        "expected_intent": "general_research",
        "expected_tickers": [],
        "required_tools": [],
        "forbidden_tools": list(FAKE_SKILL_NAMES),
        "required_evidence": [],
        "expected_stop_reason": "out_of_scope",
    },
]


__all__ = [
    "BENCHMARK_CASES",
    "FAKE_SKILLS",
    "FAKE_SKILL_NAMES",
]
