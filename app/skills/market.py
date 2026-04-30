import pandas as pd
import yfinance as yf


def _normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output.
    Handles MultiIndex columns sometimes returned by yf.download().
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    return df


def fetch_market_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch daily market data from yfinance.
    """
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    return _normalize_yfinance_df(df)


def run_market_agent(ticker: str, days: int = 7) -> dict:
    """
    Market Agent for TradePilot AI.

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
        df = fetch_market_data(ticker, period="3mo")

        if df.empty or "Close" not in df.columns:
            return {
                "ticker": ticker,
                "error": "No market data available",
                "trend_7d": 0.0,
                "trend_label": "sideways",
                "volatility": 0.0,
            }

        df = df.dropna(subset=["Close"]).copy()

        if len(df) < 2:
            return {
                "ticker": ticker,
                "error": "Not enough market data",
                "trend_7d": 0.0,
                "trend_label": "sideways",
                "volatility": 0.0,
            }

        # Use latest available trading days
        recent = df.tail(days)

        start_price = float(recent["Close"].iloc[0])
        end_price = float(recent["Close"].iloc[-1])

        trend_7d = end_price / start_price - 1

        returns = recent["Close"].pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 1 else 0.0

        ma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
        current_price = end_price

        if trend_7d > 0.02:
            trend_label = "upward"
        elif trend_7d < -0.02:
            trend_label = "downward"
        else:
            trend_label = "sideways"

        above_ma20 = None
        if ma20 is not None:
            above_ma20 = current_price > ma20

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "start_price_7d": round(start_price, 2),
            "trend_7d": float(trend_7d),
            "trend_label": trend_label,
            "volatility": volatility,
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "above_ma20": above_ma20,
            "note": (
                "Market Agent uses recent historical price behavior only. "
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