from typing import Optional

from app.skills.finnhub_tool import (
    finnhub_company_profile,
    finnhub_fundamentals_basic,
)


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _market_cap_bucket(market_cap: Optional[float]) -> str:
    """
    Finnhub marketCapitalization is typically reported in USD millions.
    """
    if market_cap is None:
        return "unknown"
    if market_cap >= 200000:
        return "mega_cap"
    if market_cap >= 10000:
        return "large_cap"
    if market_cap >= 2000:
        return "mid_cap"
    if market_cap >= 300:
        return "small_cap"
    return "micro_cap"


def _bucket_phrase(bucket: str) -> str:
    return bucket.replace("_", "-")


def _compact_company_name(profile: dict, ticker: str) -> str:
    return (
        profile.get("name")
        or profile.get("ticker")
        or ticker.upper().strip()
    )


def _build_summary(
    ticker: str,
    company_name: str,
    industry: Optional[str],
    country: Optional[str],
    market_cap_bucket: str,
    pe_ttm: Optional[float],
    beta: Optional[float],
) -> str:
    pieces = [
        f"{company_name} ({ticker}) is a {_bucket_phrase(market_cap_bucket)} company",
    ]

    if industry:
        pieces.append(f"in the {industry} industry")

    if country:
        pieces.append(f"based in {country}")

    sentence = " ".join(pieces) + "."

    details = []
    if pe_ttm is not None:
        details.append(f"P/E TTM is {pe_ttm:.2f}")
    if beta is not None:
        details.append(f"beta is {beta:.2f}")

    if not details:
        return sentence

    return sentence + " Key valuation/risk context: " + "; ".join(details) + "."


def run_fundamentals_skill(ticker: str) -> dict:
    """
    Fundamentals skill for TradePilot AI.

    This skill owns company-context evidence. It fetches:
    - company profile data
    - basic financial metrics

    Then it merges them into one normalized fundamentals payload for the
    Research and Decision agents.
    """
    clean_ticker = ticker.upper().strip()

    profile = finnhub_company_profile(clean_ticker)
    metrics = finnhub_fundamentals_basic(clean_ticker)

    profile_error = profile.get("error")
    metrics_error = metrics.get("error")

    if profile_error and metrics_error:
        return {
            "ticker": clean_ticker,
            "error": f"profile_error={profile_error}; fundamentals_error={metrics_error}",
        }

    company_name = _compact_company_name(profile, clean_ticker)
    industry = profile.get("finnhubIndustry")
    country = profile.get("country")
    exchange = profile.get("exchange")
    currency = profile.get("currency")
    ipo = profile.get("ipo")
    website = profile.get("weburl")

    market_cap = _safe_float(
        profile.get("marketCapitalization")
        if profile.get("marketCapitalization") is not None
        else metrics.get("marketCapitalization")
    )
    pe_ttm = _safe_float(metrics.get("peTTM"))
    pb = _safe_float(metrics.get("pb"))
    eps_ttm = _safe_float(metrics.get("epsTTM"))
    dividend_yield = _safe_float(metrics.get("dividendYieldIndicatedAnnual"))
    week_52_high = _safe_float(metrics.get("52WeekHigh"))
    week_52_low = _safe_float(metrics.get("52WeekLow"))
    week_52_return = _safe_float(metrics.get("52WeekPriceReturnDaily"))
    beta = _safe_float(metrics.get("beta"))

    market_cap_bucket = _market_cap_bucket(market_cap)
    summary = _build_summary(
        ticker=clean_ticker,
        company_name=company_name,
        industry=industry,
        country=country,
        market_cap_bucket=market_cap_bucket,
        pe_ttm=pe_ttm,
        beta=beta,
    )

    return {
        "ticker": clean_ticker,
        "company_name": company_name,
        "industry": industry,
        "country": country,
        "exchange": exchange,
        "currency": currency,
        "ipo": ipo,
        "website": website,
        "market_cap": market_cap,
        "market_cap_bucket": market_cap_bucket,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "eps_ttm": eps_ttm,
        "dividend_yield_annual": dividend_yield,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        "week_52_return_daily": week_52_return,
        "beta": beta,
        "summary": summary,
        "source": "Finnhub company profile + basic financials",
        "profile_error": profile_error,
        "fundamentals_error": metrics_error,
        "note": (
            "Fundamentals skill summarizes company profile and basic financial "
            "context only. It does not predict future business performance."
        ),
    }


def format_fundamentals_output(fundamentals_result: dict) -> str:
    if not fundamentals_result:
        return "Fundamentals unavailable."

    if fundamentals_result.get("error"):
        return f"Fundamentals unavailable: {fundamentals_result['error']}"

    lines = [
        f"Company Context for {fundamentals_result.get('ticker', '')}:",
        f"- Summary: {fundamentals_result.get('summary', 'N/A')}",
    ]

    if fundamentals_result.get("exchange"):
        lines.append(f"- Exchange: {fundamentals_result['exchange']}")
    if fundamentals_result.get("market_cap_bucket"):
        lines.append(f"- Size bucket: {fundamentals_result['market_cap_bucket']}")
    if fundamentals_result.get("pe_ttm") is not None:
        lines.append(f"- P/E TTM: {fundamentals_result['pe_ttm']:.2f}")
    if fundamentals_result.get("beta") is not None:
        lines.append(f"- Beta: {fundamentals_result['beta']:.2f}")

    return "\n".join(lines)


__all__ = [
    "run_fundamentals_skill",
    "format_fundamentals_output",
]
