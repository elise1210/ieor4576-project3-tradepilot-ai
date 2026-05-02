from copy import deepcopy

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


def _format_research_answer(state: dict) -> str:
    ticker = _first_ticker(state)
    evidence = state.get("evidence", {})
    news = evidence.get("news", {}).get(ticker, {})
    market = evidence.get("market", {}).get(ticker, {})
    fundamentals = evidence.get("fundamentals", {}).get(ticker, {})
    critic_result = state.get("critic_result", {})

    lines = []

    news_summary = news.get("summary")
    if news_summary:
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
        return "The pipeline did not produce a final decision."

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
