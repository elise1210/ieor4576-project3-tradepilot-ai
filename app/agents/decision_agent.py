from app.state import clone_state
from typing import Optional


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


def _downgrade_confidence(label: str, steps: int = 1) -> str:
    order = ["Low", "Medium", "High"]
    if label not in order:
        return label
    index = order.index(label)
    downgraded = max(0, index - max(0, steps))
    return order[downgraded]


def generate_decision(
    ticker: str,
    news_result: dict,
    sentiment_result: dict,
    market_result: dict,
    company_result: Optional[dict] = None,
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


def _extract_ticker_evidence(state: dict, evidence_type: str, ticker: str) -> dict:
    return state.get("evidence", {}).get(evidence_type, {}).get(ticker, {})


def _critic_items_for_ticker(state: dict, field: str, ticker: str) -> list[str]:
    items = state.get("critic_result", {}).get(field, []) or []
    suffix = f":{ticker}"
    output = []
    for item in items:
        if item.endswith(suffix):
            output.append(item.split(":", 1)[0])
    return output


def _augment_decision_with_critic_feedback(state: dict, ticker: str, decision: dict) -> dict:
    supporting_missing = sorted(set(_critic_items_for_ticker(state, "supporting_missing", ticker)))
    conflicts = sorted(set(_critic_items_for_ticker(state, "conflicts", ticker)))

    downgrade_steps = 0

    if supporting_missing:
        decision["reasoning"].append(
            "Some supporting evidence was unavailable: "
            + ", ".join(supporting_missing)
            + "."
        )
        downgrade_steps += 1

    if conflicts:
        decision["reasoning"].append(
            "Some evidence is mixed or conflicting, so this signal should be interpreted cautiously."
        )
        downgrade_steps += 1

    if downgrade_steps:
        decision["confidence"] = _downgrade_confidence(
            decision.get("confidence", "Medium"),
            steps=downgrade_steps,
        )

    decision["evidence_status"] = {
        "supporting_missing": supporting_missing,
        "conflicts": conflicts,
    }

    return decision


def _comparison_summary(decisions: dict) -> str:
    ordered = sorted(
        decisions.values(),
        key=lambda item: item.get("combined_score", 0.0),
        reverse=True,
    )

    if len(ordered) < 2:
        return f"{ordered[0]['ticker']} has the stronger overall signal right now."

    first = ordered[0]
    second = ordered[1]
    if first["combined_score"] == second["combined_score"]:
        return "The compared companies have very similar overall signals right now."

    return (
        f"{first['ticker']} has the stronger overall signal right now, "
        f"mainly due to a better combined sentiment and trend profile than {second['ticker']}."
    )


def run_decision_agent(state: dict) -> dict:
    next_state = clone_state(state)
    tickers = list(next_state.get("tickers", []))

    if not tickers:
        next_state["decision"] = {
            "error": "No ticker available for decision.",
        }
        return next_state

    decisions = {}
    for ticker in tickers:
        decision = generate_decision(
            ticker=ticker,
            news_result=_extract_ticker_evidence(next_state, "news", ticker),
            sentiment_result=_extract_ticker_evidence(next_state, "sentiment", ticker),
            market_result=_extract_ticker_evidence(next_state, "market", ticker),
            company_result=_extract_ticker_evidence(next_state, "fundamentals", ticker),
        )
        decision = _augment_decision_with_critic_feedback(next_state, ticker, decision)
        decisions[ticker] = decision

    if len(tickers) == 1:
        final_decision = decisions[tickers[0]]
    else:
        final_decision = {
            "type": "comparison",
            "tickers": tickers,
            "comparison_summary": _comparison_summary(decisions),
            "per_ticker": decisions,
            "confidence": next_state.get("confidence") or "Medium",
            "disclaimer": (
                "This comparison is informational only and not personalized financial advice."
            ),
        }

    next_state["decision"] = final_decision
    next_state["confidence"] = final_decision.get("confidence", next_state.get("confidence"))

    return next_state
