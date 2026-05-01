import os
from datetime import date, timedelta
from typing import Optional

import finnhub
from dotenv import load_dotenv


load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

finnhub_client = (
    finnhub.Client(api_key=FINNHUB_API_KEY)
    if FINNHUB_API_KEY
    else None
)


def _client_missing_error() -> dict:
    return {"error": "FINNHUB_API_KEY not found. Please set it in .env"}


def _normalize_date(value: Optional[date | str]) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def finnhub_quote(ticker: str) -> dict:
    if finnhub_client is None:
        return _client_missing_error()

    try:
        return finnhub_client.quote(ticker.upper())
    except Exception as e:
        return {"error": str(e)}


def finnhub_company_news(
    ticker: str,
    days: int = 7,
    max_items: int = 20,
) -> list:
    end = date.today()
    start = end - timedelta(days=days)

    return finnhub_company_news_range(
        ticker=ticker,
        start_date=start,
        end_date=end,
        max_items=max_items,
    )


def finnhub_company_news_range(
    ticker: str,
    start_date: date | str,
    end_date: date | str,
    max_items: int = 30,
) -> list:
    if finnhub_client is None:
        return [_client_missing_error()]

    try:
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)

        items = finnhub_client.company_news(
            ticker.upper(),
            _from=start.isoformat(),
            to=end.isoformat(),
        ) or []

        return items[:max_items]

    except Exception as e:
        return [{"error": str(e)}]


def finnhub_fundamentals_basic(ticker: str) -> dict:
    if finnhub_client is None:
        return {"ticker": ticker.upper(), **_client_missing_error()}

    try:
        response = finnhub_client.company_basic_financials(
            ticker.upper(),
            "all",
        ) or {}

        metric = response.get("metric", {})

        keep = [
            "marketCapitalization",
            "peTTM",
            "pb",
            "epsTTM",
            "dividendYieldIndicatedAnnual",
            "52WeekHigh",
            "52WeekLow",
            "52WeekPriceReturnDaily",
            "beta",
        ]

        output = {key: metric.get(key) for key in keep}
        output["ticker"] = ticker.upper()

        return output

    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}
