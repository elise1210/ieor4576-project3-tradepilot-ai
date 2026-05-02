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


def format_pipeline_answer(state: dict) -> str:
    guardrails = state.get("guardrails", {})
    if guardrails.get("out_of_scope"):
        return guardrails.get("message") or "This request is outside the current system scope."

    if state.get("needs_human"):
        return state.get("clarification_question") or "More information is needed."

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
