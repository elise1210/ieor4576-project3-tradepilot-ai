import pandas as pd

from app.skills.yfinance_tool import (
    fetch_daily_prices,
    fetch_daily_prices_until,
)


def _price_history_from_df(df: pd.DataFrame) -> list[dict]:
    history = []
    for index, row in df.iterrows():
        item = {
            "date": index.date().isoformat(),
            "close": round(float(row["Close"]), 2),
        }
        for source_key, output_key in (
            ("Open", "open"),
            ("High", "high"),
            ("Low", "low"),
            ("Volume", "volume"),
        ):
            if source_key not in row or pd.isna(row[source_key]):
                continue
            value = float(row[source_key])
            item[output_key] = int(value) if output_key == "volume" else round(value, 2)
        history.append(item)
    return history


def fetch_market_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch daily market data from yfinance.
    """
    return fetch_daily_prices(ticker=ticker, period=period)


def run_market_skill(ticker: str, days: int = 7, end_date=None) -> dict:
    """
    Market skill for TradePilot AI.

    This agent:
    - retrieves recent price data using yfinance
    - computes 7-day trend
    - computes short-term volatility
    - computes moving average context

    It does NOT predict future prices.
    It only provides recent market context.
    """

    ticker = ticker.upper().strip()

    try:
        requested_date = str(end_date) if end_date is not None else None
        if end_date is not None:
            history = fetch_daily_prices_until(ticker=ticker, end_date=end_date, days=days)
            if not history:
                return {
                    "ticker": ticker,
                    "error": "No market data available",
                    "trend_7d": 0.0,
                    "trend_label": "sideways",
                    "volatility": 0.0,
                    "history": [],
                    "requested_date": requested_date,
                    "used_end_date": None,
                }

            current_price = float(history[-1]["close"])
            start_price = float(history[0]["close"])
            closes = pd.Series([item["close"] for item in history], dtype=float)
            returns = closes.pct_change().dropna()
            trend_7d = current_price / start_price - 1
            volatility = float(returns.std()) if len(returns) > 1 else 0.0
            ma20 = None
            above_ma20 = None
            used_end_date = history[-1]["date"]
        else:
            df = fetch_market_data(ticker, period="3mo")

            if df.empty or "Close" not in df.columns:
                return {
                    "ticker": ticker,
                    "error": "No market data available",
                    "trend_7d": 0.0,
                    "trend_label": "sideways",
                    "volatility": 0.0,
                    "history": [],
                    "requested_date": None,
                    "used_end_date": None,
                }

            df = df.dropna(subset=["Close"]).copy()

            if len(df) < 2:
                return {
                    "ticker": ticker,
                    "error": "Not enough market data",
                    "trend_7d": 0.0,
                    "trend_label": "sideways",
                    "volatility": 0.0,
                    "history": [],
                    "requested_date": None,
                    "used_end_date": None,
                }

            # Use latest available trading days
            recent = df.tail(days)

            start_price = float(recent["Close"].iloc[0])
            current_price = float(recent["Close"].iloc[-1])
            history = _price_history_from_df(recent)
            used_end_date = history[-1]["date"] if history else None

            trend_7d = current_price / start_price - 1

            returns = recent["Close"].pct_change().dropna()
            volatility = float(returns.std()) if len(returns) > 1 else 0.0

            ma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
            above_ma20 = current_price > ma20 if ma20 is not None else None

        if trend_7d > 0.02:
            trend_label = "upward"
        elif trend_7d < -0.02:
            trend_label = "downward"
        else:
            trend_label = "sideways"

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "start_price_7d": round(start_price, 2),
            "trend_7d": float(trend_7d),
            "trend_label": trend_label,
            "volatility": volatility,
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "above_ma20": above_ma20,
            "history": history,
            "requested_date": requested_date,
            "used_end_date": used_end_date,
            "start_date": history[0]["date"] if history else None,
            "end_date": history[-1]["date"] if history else None,
            "note": (
                "Market skill uses recent historical price behavior only. "
                "It does not forecast future stock prices."
            ),
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e),
            "trend_7d": 0.0,
            "trend_label": "sideways",
            "volatility": 0.0,
            "history": [],
            "requested_date": str(end_date) if end_date is not None else None,
            "used_end_date": None,
        }


def format_market_output(market_result: dict) -> str:
    if not market_result:
        return "Market data unavailable."

    if "error" in market_result:
        return f"Market data unavailable: {market_result['error']}"

    trend_pct = market_result.get("trend_7d", 0.0) * 100
    vol_pct = market_result.get("volatility", 0.0) * 100

    lines = [
        f"Market Signal for {market_result.get('ticker', '')}:",
        f"- Current price: ${market_result.get('current_price', 'N/A')}",
        f"- 7-day trend: {trend_pct:+.2f}% ({market_result.get('trend_label', 'sideways')})",
        f"- Short-term volatility: {vol_pct:.2f}%",
    ]

    if market_result.get("ma20") is not None:
        position = "above" if market_result.get("above_ma20") else "below"
        lines.append(f"- Price is {position} its 20-day moving average (${market_result['ma20']}).")

    lines.append("")
    lines.append("This is recent market context, not a price forecast.")

    return "\n".join(lines)


__all__ = ["run_market_skill", "format_market_output", "fetch_market_data"]
