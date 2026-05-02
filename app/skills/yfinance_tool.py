import os
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Union

import pandas as pd
from dotenv import load_dotenv
try:
    import yfinance as yf
except ImportError:  # pragma: no cover - environment-dependent dependency
    class _MissingYFinance:
        def download(self, *args, **kwargs):
            raise ModuleNotFoundError("yfinance is not installed")

        def Ticker(self, *args, **kwargs):
            raise ModuleNotFoundError("yfinance is not installed")

    yf = _MissingYFinance()

from app.skills.date_utils import parse_user_date

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_YFINANCE_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "py-yfinance-cache")


def _configure_yfinance_cache() -> None:
    """
    Point yfinance's local SQLite cache to a writable location.

    On some Windows setups, the default AppData cache path can be blocked or
    locked, which causes errors like "unable to open database file". We allow
    an override through YFINANCE_CACHE_DIR and otherwise fall back to C:\\tmp.
    """
    if not hasattr(yf, "__dict__"):
        return

    cache_dir = os.getenv("YFINANCE_CACHE_DIR", DEFAULT_YFINANCE_CACHE_DIR)

    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        return

    try:
        import yfinance.cache as yf_cache

        if hasattr(yf_cache, "set_cache_location"):
            yf_cache.set_cache_location(cache_dir)
        if hasattr(yf_cache, "set_tz_cache_location"):
            yf_cache.set_tz_cache_location(cache_dir)
    except Exception:
        return


_configure_yfinance_cache()


def normalize_reference_date(value: Optional[Union[date, datetime, str]] = None) -> date:
    """
    Normalize common user-facing date values.

    Accepted examples:
    - None, "today", "current", "latest", "now" -> date.today()
    - "yesterday" -> date.today() - 1 day
    - "2026-05-01" -> date(2026, 5, 1)
    """
    if value is None:
        return date.today()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    parsed = parse_user_date(str(value))
    if parsed is not None:
        return parsed

    return date.fromisoformat(str(value).strip()[:10])


def normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return None
        return float(value)
    except Exception:
        return None


def _download_daily(
    ticker: str,
    period: Optional[str] = None,
    start: Optional[Union[date, str]] = None,
    end: Optional[Union[date, str]] = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    kwargs = {
        "tickers": ticker.upper().strip(),
        "interval": "1d",
        "auto_adjust": auto_adjust,
        "progress": False,
    }

    if start is not None or end is not None:
        if start is not None:
            kwargs["start"] = normalize_reference_date(start).isoformat()
        if end is not None:
            kwargs["end"] = normalize_reference_date(end).isoformat()
    else:
        kwargs["period"] = period or "3mo"

    df = yf.download(**kwargs)
    return normalize_yfinance_df(df)


def fetch_daily_prices(
    ticker: str,
    period: str = "3mo",
    start: Optional[Union[date, str]] = None,
    end: Optional[Union[date, str]] = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from yfinance.

    Use period for latest data, or start/end for a specific date range.
    """
    return _download_daily(
        ticker=ticker,
        period=period,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
    )


def dataframe_to_price_history(df: pd.DataFrame) -> list[dict]:
    df = normalize_yfinance_df(df)
    if df.empty:
        return []

    rows = []
    for index, row in df.iterrows():
        item = {"date": index.date().isoformat()}
        for source_key, output_key in (
            ("Open", "open"),
            ("High", "high"),
            ("Low", "low"),
            ("Close", "close"),
            ("Volume", "volume"),
        ):
            if source_key not in row:
                continue
            value = _safe_float(row[source_key])
            if value is None:
                continue
            item[output_key] = round(value, 2) if output_key != "volume" else int(value)

        rows.append(item)

    return rows


def fetch_recent_daily_prices(
    ticker: str,
    days: int = 7,
    period: str = "3mo",
) -> list[dict]:
    df = fetch_daily_prices(ticker=ticker, period=period)
    if df.empty:
        return []

    df = df.dropna(subset=["Close"]).tail(days)
    return dataframe_to_price_history(df)


def fetch_daily_prices_until(
    ticker: str,
    end_date: Optional[Union[date, str]] = None,
    days: int = 7,
    lookback_calendar_days: int = 30,
) -> list[dict]:
    """
    Fetch the last N trading days ending at or before end_date.

    If end_date is a weekend/holiday, yfinance data falls back to the most
    recent previous trading day.
    """
    target = normalize_reference_date(end_date)
    start = target - timedelta(days=lookback_calendar_days)
    exclusive_end = target + timedelta(days=1)

    df = fetch_daily_prices(ticker=ticker, start=start, end=exclusive_end)
    if df.empty:
        return []

    df = df[df.index.date <= target].dropna(subset=["Close"]).tail(days)
    return dataframe_to_price_history(df)


def fetch_ohlcv_on_date(
    ticker: str,
    target_date: Union[date, str],
    fields: Optional[Iterable[str]] = None,
) -> dict:
    """
    Fetch OHLCV fields for a specific date.

    If the exact date is not a trading day, returns the most recent previous
    trading day in the nearby window.
    """
    target = normalize_reference_date(target_date)
    start = target - timedelta(days=7)
    exclusive_end = target + timedelta(days=7)
    requested_fields = list(fields or ["Open", "High", "Low", "Close", "Volume"])

    df = fetch_daily_prices(ticker=ticker, start=start, end=exclusive_end)
    if df.empty:
        return {"ticker": ticker.upper(), "target_date": target.isoformat(), "error": "price_unavailable"}

    df = df[df.index.date <= target]
    if df.empty:
        return {"ticker": ticker.upper(), "target_date": target.isoformat(), "error": "price_unavailable"}

    row = df.tail(1).iloc[0]
    used_date = df.tail(1).index[0].date()

    output = {
        "ticker": ticker.upper(),
        "target_date": target.isoformat(),
        "used_date": used_date.isoformat(),
    }

    for field in requested_fields:
        source_field = "Close" if field == "Price" else field
        value = _safe_float(row.get(source_field))
        output[field] = round(value, 2) if value is not None and field != "Volume" else value
        if field == "Volume" and value is not None:
            output[field] = int(value)

    return output


def fetch_today_intraday_fields(
    ticker: str,
    fields: Optional[Iterable[str]] = None,
) -> dict:
    requested_fields = list(fields or ["Price", "Open", "High", "Low", "Volume"])
    clean_ticker = ticker.upper().strip()
    ticker_obj = yf.Ticker(clean_ticker)

    try:
        intraday = ticker_obj.history(period="1d", interval="1m")
    except Exception:
        intraday = pd.DataFrame()
    intraday = normalize_yfinance_df(intraday)

    try:
        daily = ticker_obj.history(period="5d", interval="1d")
    except Exception:
        daily = pd.DataFrame()
    daily = normalize_yfinance_df(daily)

    last = None
    try:
        fast_info = getattr(ticker_obj, "fast_info", {}) or {}
        last = _safe_float(fast_info.get("last_price"))
    except Exception:
        last = None

    if last is None and not intraday.empty and "Close" in intraday.columns:
        last = _safe_float(intraday["Close"].iloc[-1])
    if last is None and not daily.empty and "Close" in daily.columns:
        last = _safe_float(daily["Close"].iloc[-1])

    output = {"ticker": clean_ticker, "date": date.today().isoformat()}

    for field in requested_fields:
        value = None
        if field in {"Price", "Close"}:
            value = last
        elif field == "Open" and not intraday.empty and "Open" in intraday.columns:
            value = _safe_float(intraday["Open"].iloc[0])
        elif field == "High" and not intraday.empty and "High" in intraday.columns:
            value = _safe_float(intraday["High"].max())
        elif field == "Low" and not intraday.empty and "Low" in intraday.columns:
            value = _safe_float(intraday["Low"].min())
        elif field == "Volume" and not intraday.empty and "Volume" in intraday.columns:
            value = _safe_float(intraday["Volume"].sum())

        if value is None and not daily.empty:
            source_field = "Close" if field == "Price" else field
            if source_field in daily.columns:
                value = _safe_float(daily[source_field].iloc[-1])

        output[field] = int(value) if field == "Volume" and value is not None else (
            round(value, 2) if value is not None else None
        )

    return output


def fetch_latest_price_snapshot(ticker: str) -> dict:
    clean_ticker = ticker.upper().strip()
    history = fetch_daily_prices(ticker=clean_ticker, period="5d")
    if history.empty or "Close" not in history.columns:
        return {"ticker": clean_ticker, "error": "price_unavailable"}

    close_price = _safe_float(history["Close"].iloc[-1])
    close_date = history.index[-1].date().isoformat()
    prev_close = _safe_float(history["Close"].iloc[-2]) if len(history) >= 2 else None

    pct_change = None
    if close_price is not None and prev_close not in (None, 0):
        pct_change = (close_price - prev_close) / prev_close

    return {
        "ticker": clean_ticker,
        "close_price": round(close_price, 2) if close_price is not None else None,
        "close_date": close_date,
        "previous_close": round(prev_close, 2) if prev_close is not None else None,
        "pct_change": pct_change,
    }
