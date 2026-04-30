import os
from datetime import date, timedelta

import finnhub
from dotenv import load_dotenv


load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found. Please set it in .env")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


def finnhub_quote(ticker: str) -> dict:
    try:
        return finnhub_client.quote(ticker.upper())
    except Exception as e:
        return {"error": str(e)}


def finnhub_company_news(
    ticker: str,
    days: int = 7,
    max_items: int = 20,
) -> list:
    try:
        end = date.today()
        start = end - timedelta(days=days)

        items = finnhub_client.company_news(
            ticker.upper(),
            _from=start.isoformat(),
            to=end.isoformat(),
        ) or []

        return items[:max_items]

    except Exception as e:
        return [{"error": str(e)}]


def finnhub_fundamentals_basic(ticker: str) -> dict:
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