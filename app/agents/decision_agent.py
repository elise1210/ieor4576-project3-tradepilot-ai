def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _normalize_sentiment_score(sentiment_result: dict) -> float:
    """
    Expected sentiment_result examples:
    {"sentiment": "positive", "score": 0.42}
    {"sentiment": "negative", "score": -0.35}
    """
    if not sentiment_result:
        return 0.0

    score = _safe_float(sentiment_result.get("score"), 0.0)

    # If score missing, infer weak score from label
    if score == 0.0:
        label = str(
            sentiment_result.get("sentiment")
            or sentiment_result.get("label")
            or ""
        ).lower()

        if label == "positive":
            return 0.25
        if label == "negative":
            return -0.25

    return score


def _normalize_trend_score(market_result: dict) -> float:
    """
    Expected market_result examples:
    {"trend_7d": 0.023, "trend_label": "upward"}
    {"return_7d": -0.015}
    """
    if not market_result:
        return 0.0

    if "trend_7d" in market_result:
        return _safe_float(market_result.get("trend_7d"), 0.0)

    if "return_7d" in market_result:
        return _safe_float(market_result.get("return_7d"), 0.0)

    label = str(market_result.get("trend_label", "")).lower()

    if label == "upward":
        return 0.02
    if label == "downward":
        return -0.02

    return 0.0


def _risk_level(market_result: dict, sentiment_result: dict) -> str:
    volatility = _safe_float(market_result.get("volatility"), 0.0) if market_result else 0.0
    dispersion = _safe_float(sentiment_result.get("dispersion"), 0.0) if sentiment_result else 0.0

    if volatility >= 0.035 or dispersion >= 0.35:
        return "High"
    if volatility >= 0.02 or dispersion >= 0.20:
        return "Medium"
    return "Low"


def _confidence_label(abs_score: float, risk: str) -> str:
    if risk == "High":
        return "Low"
    if abs_score >= 0.45:
        return "High"
    if abs_score >= 0.25:
        return "Medium"
    return "Low"


def generate_decision(
    ticker: str,
    news_result: dict,
    sentiment_result: dict,
    market_result: dict,
    company_result: dict | None = None,
) -> dict:
    """
    TradePilot Decision Agent.

    This function does NOT predict future stock prices.
    It generates a daily informational buy/hold/sell signal using:
    - recent news sentiment
    - short-term price trend
    - volatility / risk
    - company context when available
    """

    ticker = ticker.upper()

    sentiment_score = _normalize_sentiment_score(sentiment_result)
    trend_score = _normalize_trend_score(market_result)

    risk = _risk_level(market_result or {}, sentiment_result or {})

    # Weighted daily signal:
    # sentiment matters slightly more than price trend because user asks about news-driven decision
    combined_score = 0.65 * sentiment_score + 0.35 * (trend_score * 10)

    if risk == "High":
        buy_threshold = 0.35
        sell_threshold = -0.35
    else:
        buy_threshold = 0.25
        sell_threshold = -0.25

    if combined_score >= buy_threshold:
        recommendation = "BUY"
    elif combined_score <= sell_threshold:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    confidence = _confidence_label(abs(combined_score), risk)

    sentiment_label = (
        sentiment_result.get("sentiment")
        or sentiment_result.get("label")
        or "neutral"
        if sentiment_result
        else "neutral"
    )

    trend_label = market_result.get("trend_label", "sideways") if market_result else "sideways"

    key_driver = "Mixed signals across news and price trend."
    if news_result and news_result.get("summary"):
        key_driver = news_result["summary"].split("\n\n")[0]

    reasoning = [
        f"News sentiment is {sentiment_label} with score {sentiment_score:+.3f}.",
        f"Recent price trend is {trend_label} with 7-day movement {trend_score:+.2%}.",
        f"Risk level is {risk}, based on volatility and sentiment dispersion.",
    ]

    if company_result and company_result.get("summary"):
        reasoning.append(f"Company context: {company_result['summary']}")

    return {
        "ticker": ticker,
        "recommendation": recommendation,
        "confidence": confidence,
        "risk_level": risk,
        "combined_score": round(combined_score, 3),
        "drivers": {
            "sentiment_score": round(sentiment_score, 3),
            "trend_score_7d": round(trend_score, 4),
            "trend_label": trend_label,
            "key_driver": key_driver,
        },
        "reasoning": reasoning,
        "disclaimer": (
            "This is an informational daily signal, not financial advice. "
            "The system does not predict future stock prices."
        ),
    }


def format_decision_output(decision: dict) -> str:
    """
    Human-readable output for frontend/demo.
    """

    lines = [
        f"Recommendation: {decision['recommendation']}",
        f"Confidence: {decision['confidence']}",
        f"Risk Level: {decision['risk_level']}",
        "",
        "Reason:",
    ]

    for r in decision.get("reasoning", []):
        lines.append(f"- {r}")

    lines.extend([
        "",
        f"Key Driver: {decision['drivers']['key_driver']}",
        "",
        decision["disclaimer"],
    ])

    return "\n".join(lines)