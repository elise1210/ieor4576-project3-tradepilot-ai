from copy import deepcopy

from app.agents.decision_agent import build_decision_preview
from app.skills.compliance import run_compliance_skill


def apply_compliance_to_state(state: dict) -> dict:
    """
    Wrap final decision payloads with compliance metadata for backend output.

    This keeps the agent loop unchanged and applies safety/uncertainty wording
    only at the response layer.
    """
    next_state = deepcopy(state)
    decision = next_state.get("decision")

    if not isinstance(decision, dict) or not decision:
        stopped_reason = next_state.get("metadata", {}).get("stopped_reason")
        if (
            stopped_reason == "iteration_budget_exhausted"
            and next_state.get("intent") in {"buy_sell_decision", "comparison"}
        ):
            preview = build_decision_preview(next_state)
            if preview:
                next_state["decision_preview"] = preview
        return next_state

    if decision.get("type") == "comparison":
        wrapped = run_compliance_skill(decision, state=next_state)
        per_ticker = {}
        for ticker, payload in decision.get("per_ticker", {}).items():
            per_ticker[ticker] = run_compliance_skill(payload, state=next_state)
        wrapped["per_ticker"] = per_ticker
        next_state["decision"] = wrapped
        return next_state

    next_state["decision"] = run_compliance_skill(decision, state=next_state)
    return next_state


def _first_ticker(state: dict) -> str:
    tickers = list(state.get("tickers", []))
    return tickers[0] if tickers else ""


def _is_sentiment_query(state: dict) -> bool:
    query = (state.get("query") or "").lower()
    required = set(state.get("plan", {}).get("required_evidence", []))
    return "sentiment" in query or ("sentiment" in required and "news" in required)


def _format_sentiment_answer(ticker: str, sentiment: dict) -> str:
    if not sentiment:
        return ""

    label = sentiment.get("sentiment", "neutral")
    score = sentiment.get("score", 0.0)
    article_count = sentiment.get("article_count", 0)
    positive = sentiment.get("positive_count", 0)
    neutral = sentiment.get("neutral_count", 0)
    negative = sentiment.get("negative_count", 0)
    requested_date = sentiment.get("requested_date")
    date_text = f" on {requested_date}" if requested_date else ""

    return (
        f"Sentiment for {ticker}{date_text}: {label}.\n"
        f"Score: {float(score):+.3f} based on {article_count} article(s).\n"
        f"Breakdown: {positive} positive, {neutral} neutral, {negative} negative."
    )


def _wants_price_answer(query: str) -> bool:
    text = (query or "").lower()
    return any(
        keyword in text
        for keyword in ("price", "stock price", "closing price", "close", "open", "high", "low", "volume")
    )


def _format_price_answer(ticker: str, market: dict) -> str:
    if not market or market.get("current_price") is None:
        return ""

    price = float(market["current_price"])
    requested_date = market.get("requested_date")
    used_date = market.get("used_end_date") or market.get("end_date")

    if requested_date:
        if used_date and used_date != requested_date:
            return (
                f"{ticker}'s closing price on the latest available trading day at or before "
                f"{requested_date} was ${price:.2f} ({used_date})."
            )
        return f"{ticker}'s closing price on {requested_date} was ${price:.2f}."

    if used_date:
        return f"{ticker}'s latest available closing price was ${price:.2f} ({used_date})."

    return f"{ticker}'s latest available closing price was ${price:.2f}."


def _format_research_answer(state: dict) -> str:
    ticker = _first_ticker(state)
    evidence = state.get("evidence", {})
    news = evidence.get("news", {}).get(ticker, {})
    market = evidence.get("market", {}).get(ticker, {})
    fundamentals = evidence.get("fundamentals", {}).get(ticker, {})
    sentiment = evidence.get("sentiment", {}).get(ticker, {})
    critic_result = state.get("critic_result", {})

    lines = []

    if _wants_price_answer(state.get("query", "")):
        price_answer = _format_price_answer(ticker, market)
        if price_answer:
            lines.append(price_answer)

    if _is_sentiment_query(state):
        sentiment_answer = _format_sentiment_answer(ticker, sentiment)
        if sentiment_answer:
            lines.append(sentiment_answer)

    news_summary = news.get("summary")
    if news_summary:
        if lines:
            lines.append("")
        lines.append(news_summary)

    trend_label = market.get("trend_label")
    trend_score = market.get("trend_7d")
    if trend_label or trend_score is not None:
        if trend_score is None:
            lines.extend([
                "",
                f"Recent market context: trend is {trend_label}.",
            ])
        else:
            lines.extend([
                "",
                f"Recent market context: trend is {trend_label} with 7-day movement {float(trend_score):+.2%}.",
            ])

    fundamentals_summary = fundamentals.get("summary")
    if fundamentals_summary:
        lines.extend([
            "",
            f"Company context: {fundamentals_summary}",
        ])

    supporting_missing = critic_result.get("supporting_missing", [])
    semantic_enough = critic_result.get("semantic_enough")
    critic_reasoning_brief = state.get("metadata", {}).get("critic_reasoning_brief")
    stopped_reason = state.get("metadata", {}).get("stopped_reason")

    if semantic_enough is False and critic_reasoning_brief:
        lines.extend([
            "",
            "Evidence note:",
            f"- {critic_reasoning_brief}",
        ])
    elif stopped_reason == "iteration_budget_exhausted" and critic_reasoning_brief:
        lines.extend([
            "",
            "Evidence note:",
            f"- {critic_reasoning_brief}",
        ])

    if supporting_missing:
        readable = []
        for item in supporting_missing:
            if ":" in item:
                readable.append(item.split(":", 1)[0])
            else:
                readable.append(item)
        lines.extend([
            "",
            "Caution:",
            "- Some supporting evidence was unavailable: " + ", ".join(sorted(set(readable))) + ".",
        ])

    if not lines:
        return "I collected evidence, but I do not yet have a clean research summary to return."

    return "\n".join(lines).strip()


def _first_summary_sentence(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    first_line = raw.splitlines()[0].strip()
    if not first_line:
        return ""

    for delimiter in (". ", ".\n"):
        if delimiter in first_line:
            return first_line.split(delimiter, 1)[0].strip() + "."

    return first_line


def _format_failed_decision_summary(state: dict) -> list[str]:
    ticker = _first_ticker(state)
    evidence = state.get("evidence", {})
    news = evidence.get("news", {}).get(ticker, {})
    market = evidence.get("market", {}).get(ticker, {})
    fundamentals = evidence.get("fundamentals", {}).get(ticker, {})
    sentiment = evidence.get("sentiment", {}).get(ticker, {})
    preview = state.get("decision_preview") or {}

    bullets = []

    if preview.get("type") == "provisional_single" and preview.get("risk_level"):
        bullets.append(
            f"Provisional risk level from the available evidence is {preview['risk_level']}."
        )

    if sentiment:
        label = sentiment.get("sentiment", "neutral")
        score = float(sentiment.get("score", 0.0))
        article_count = sentiment.get("article_count", 0)
        bullets.append(
            f"Sentiment is {label} with score {score:+.3f} across {article_count} article(s)."
        )

    trend_label = market.get("trend_label")
    trend_score = market.get("trend_7d")
    if trend_label or trend_score is not None:
        if trend_score is None:
            bullets.append(f"Recent price trend is {trend_label}.")
        else:
            bullets.append(
                f"Recent price trend is {trend_label} over 7 trading days ({float(trend_score):+.2%})."
            )

    news_summary = _first_summary_sentence(news.get("summary", ""))
    if news_summary:
        bullets.append(f"News snapshot: {news_summary}")

    fundamentals_summary = _first_summary_sentence(fundamentals.get("summary", ""))
    if fundamentals_summary:
        bullets.append(f"Company context: {fundamentals_summary}")

    return bullets


def _format_failed_decision_answer(state: dict) -> str:
    guardrails = state.get("guardrails", {})
    scope_note = guardrails.get("scope_note")
    stopped_reason = state.get("metadata", {}).get("stopped_reason")
    critic_reasoning_brief = state.get("metadata", {}).get("critic_reasoning_brief")
    summary_bullets = _format_failed_decision_summary(state)

    lines = []
    if scope_note:
        lines.append(scope_note)
        lines.append("")

    if stopped_reason == "iteration_budget_exhausted":
        lines.append(
            "I collected evidence, but I could not produce a reliable final recommendation within the iteration budget."
        )
    else:
        lines.append("I collected evidence, but I could not produce a reliable final recommendation.")

    if critic_reasoning_brief:
        lines.extend([
            "",
            "Why no final recommendation:",
            f"- {critic_reasoning_brief}",
        ])

    if summary_bullets:
        lines.extend([
            "",
            "Current evidence summary:",
        ])
        for bullet in summary_bullets:
            lines.append(f"- {bullet}")

    return "\n".join(lines).strip()


def format_pipeline_answer(state: dict) -> str:
    guardrails = state.get("guardrails", {})
    if guardrails.get("out_of_scope"):
        return guardrails.get("message") or "This request is outside the current system scope."

    if state.get("needs_human"):
        return state.get("clarification_question") or "More information is needed."

    if state.get("intent") not in {"buy_sell_decision", "comparison"}:
        return _format_research_answer(state)

    decision = state.get("decision") or {}
    if not decision:
        return _format_failed_decision_answer(state)

    scope_note = guardrails.get("scope_note")

    if decision.get("type") == "comparison":
        lines = [
            "Comparison Result:",
            decision.get("comparison_summary", "No comparison summary available."),
            "",
        ]

        per_ticker = decision.get("per_ticker", {})
        for ticker, result in per_ticker.items():
            lines.append(
                f"{ticker}: {result.get('recommendation', 'N/A')} "
                f"(Confidence: {result.get('confidence', 'N/A')})"
            )
            for note in result.get("uncertainty_notes", []):
                lines.append(f"- {ticker} note: {note}")

        for note in decision.get("uncertainty_notes", []):
            lines.extend(["", note])

        disclaimer = decision.get("disclaimer")
        if disclaimer:
            lines.extend(["", disclaimer])

        if scope_note:
            lines = [scope_note, ""] + lines

        return "\n".join(lines)

    lines = [
        f"Recommendation: {decision.get('recommendation', 'N/A')}",
        f"Confidence: {decision.get('confidence', 'N/A')}",
        f"Risk Level: {decision.get('risk_level', 'N/A')}",
        "",
        "Reason:",
    ]

    for item in decision.get("reasoning", []):
        lines.append(f"- {item}")

    key_driver = decision.get("drivers", {}).get("key_driver")
    if key_driver:
        lines.extend(["", f"Key Driver: {key_driver}"])

    uncertainty_notes = decision.get("uncertainty_notes", [])
    if uncertainty_notes:
        lines.extend(["", "Caution:"])
        for note in uncertainty_notes:
            lines.append(f"- {note}")

    disclaimer = decision.get("disclaimer")
    if disclaimer:
        lines.extend(["", disclaimer])

    if scope_note:
        lines = [scope_note, ""] + lines

    return "\n".join(lines)


__all__ = ["apply_compliance_to_state", "format_pipeline_answer"]
