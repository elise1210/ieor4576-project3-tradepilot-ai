import re
from datetime import date, datetime
from typing import Any, Optional

CHART_SPEC_VERSION = "1.0"
TODAY_TERMS = (
    "today",
    "latest",
    "current",
    "currently",
    "right now",
    "now",
)
PRICE_TERMS = (
    "price",
    "stock",
    "share",
    "shares",
    "trend",
    "movement",
    "move",
    "moving",
    "up",
    "down",
    "buy",
    "sell",
    "hold",
)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_ticker(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _normalize_reference_date(reference_date: Any = None) -> date:
    if reference_date is None:
        return date.today()

    if isinstance(reference_date, datetime):
        return reference_date.date()

    if isinstance(reference_date, date):
        return reference_date

    text = str(reference_date).strip()
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return date.today()


def _today_date_patterns(reference_date: date) -> tuple[str, ...]:
    return (
        reference_date.isoformat(),
        reference_date.strftime("%m/%d/%Y"),
        reference_date.strftime("%-m/%-d/%Y"),
        reference_date.strftime("%B %-d, %Y").lower(),
        reference_date.strftime("%B %-d %Y").lower(),
        reference_date.strftime("%b %-d, %Y").lower(),
        reference_date.strftime("%b %-d %Y").lower(),
    )


def _mentions_today_or_latest(query: str, reference_date: Any = None) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False

    if any(term in text for term in TODAY_TERMS):
        return True

    today = _normalize_reference_date(reference_date)
    return any(pattern in text for pattern in _today_date_patterns(today))


def _mentions_price_or_daily_signal(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False

    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in PRICE_TERMS)


def should_show_chart_for_query(query: str, reference_date: Any = None) -> bool:
    """
    Return True for today/latest price or daily-signal questions.

    Historical date-specific, news-only, or sentiment-only questions should not
    trigger a price chart. Future forecast-style questions should be handled by
    the planner or decision agents.
    """
    return (
        _mentions_today_or_latest(query, reference_date=reference_date)
        and _mentions_price_or_daily_signal(query)
    )


def _format_date(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, datetime):
        return value.date().isoformat()

    text = str(value).strip()
    if not text:
        return ""

    # yfinance/pandas timestamps often arrive as "YYYY-MM-DD 00:00:00".
    return text[:10]


def _extract_series_points(market_result: dict) -> list[dict]:
    """
    Convert common market-history payload shapes into chart points.

    The project currently stores compact market signals, but this helper also
    accepts richer future payloads such as history/prices/ohlc lists.
    """
    if not market_result:
        return []

    candidate_keys = ("history", "prices", "price_history", "ohlc")
    rows = []
    for key in candidate_keys:
        value = market_result.get(key)
        if isinstance(value, list):
            rows = value
            break

    points = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        date_value = (
            row.get("date")
            or row.get("datetime")
            or row.get("timestamp")
            or row.get("time")
        )
        price = _safe_float(
            row.get("close")
            or row.get("Close")
            or row.get("price")
            or row.get("adj_close")
            or row.get("Adj Close")
        )

        if price is None:
            continue

        points.append({
            "date": _format_date(date_value),
            "close": round(price, 2),
        })

    return [point for point in points if point["date"]]


def _fallback_trend_points(market_result: dict) -> list[dict]:
    start_price = _safe_float(market_result.get("start_price_7d"))
    current_price = _safe_float(market_result.get("current_price"))

    if start_price is None or current_price is None:
        return []

    return [
        {"date": "7 trading days ago", "close": round(start_price, 2)},
        {"date": "latest", "close": round(current_price, 2)},
    ]


def _build_price_chart(ticker: str, market_result: dict) -> Optional[dict]:
    points = _extract_series_points(market_result)
    chart_type = "line"

    if not points:
        points = _fallback_trend_points(market_result)
        chart_type = "slope"

    if not points:
        trend = _safe_float(market_result.get("trend_7d"))
        if trend is None:
            return None

        points = [{
            "label": "7-day trend",
            "value": round(trend * 100, 2),
        }]
        return {
            "id": f"{ticker.lower()}-trend",
            "type": "bar",
            "title": f"{ticker} 7-Day Price Trend",
            "x_field": "label",
            "y_field": "value",
            "unit": "percent",
            "data": points,
            "encoding": {
                "x": {"field": "label", "type": "nominal"},
                "y": {"field": "value", "type": "quantitative"},
            },
        }

    return {
        "id": f"{ticker.lower()}-price",
        "type": chart_type,
        "title": f"{ticker} Recent Price",
        "x_field": "date",
        "y_field": "close",
        "unit": "usd",
        "data": points,
        "encoding": {
            "x": {"field": "date", "type": "temporal"},
            "y": {"field": "close", "type": "quantitative"},
        },
    }


def _build_highlights(evidence: dict) -> list[str]:
    market = evidence.get("market") or {}
    highlights = []

    trend_label = market.get("trend_label")
    trend = _safe_float(market.get("trend_7d"))
    if trend_label and trend is not None:
        highlights.append(f"7-day price trend is {trend_label} ({trend:+.2%}).")

    current_price = _safe_float(market.get("current_price"))
    ma20 = _safe_float(market.get("ma20"))
    if current_price is not None and ma20 is not None:
        position = "above" if current_price > ma20 else "below"
        highlights.append(f"Latest price is {position} the 20-day moving average.")

    return highlights


def build_chart_spec(ticker: str, evidence: dict) -> dict:
    """
    Build frontend-ready chart specifications from structured evidence.

    The return value is intentionally JSON-serializable and library-neutral.
    A frontend can render these specs with Chart.js, Vega-Lite, Plotly, or a
    small custom component.
    """
    clean_ticker = _clean_ticker(ticker)
    if not clean_ticker:
        return {"error": "Ticker is required for chart output."}

    evidence = evidence or {}
    market = evidence.get("market") or {}

    price_chart = _build_price_chart(clean_ticker, market)
    charts = [price_chart] if price_chart is not None else []

    if not charts:
        return {
            "ticker": clean_ticker,
            "error": "No chartable evidence available.",
        }

    return {
        "ticker": clean_ticker,
        "chart_id": f"{clean_ticker.lower()}-seven-day-price-trend",
        "version": CHART_SPEC_VERSION,
        "kind": "seven_day_price_trend",
        "title": f"{clean_ticker} 7-Day Price Trend",
        "source": "TradePilot structured evidence",
        "charts": charts,
        "highlights": _build_highlights(evidence),
        "note": (
            "Chart output summarizes recent historical prices only. "
            "It is not a price forecast or investment advice."
        ),
    }


def run_chart_skill(
    ticker: str,
    evidence: dict,
    query: Optional[str] = None,
    reference_date: Any = None,
) -> dict:
    if query is not None and not should_show_chart_for_query(query, reference_date):
        return {
            "ticker": _clean_ticker(ticker),
            "chart_available": False,
            "reason": "Chart is only shown for today/latest price or daily signal questions.",
        }

    return build_chart_spec(ticker=ticker, evidence=evidence)


__all__ = ["build_chart_spec", "run_chart_skill", "should_show_chart_for_query"]
