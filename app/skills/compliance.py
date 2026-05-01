from copy import deepcopy
from typing import Any


DEFAULT_DISCLAIMER = (
    "This is educational and informational analysis only, not personalized "
    "financial advice. It does not guarantee future performance."
)

DAILY_SIGNAL_DISCLAIMER = (
    "This is an informational daily signal based on available evidence, not "
    "financial advice or a prediction of future stock prices."
)

COMPARISON_DISCLAIMER = (
    "This comparison is informational only and not personalized financial "
    "advice. Relative signals can change as new data arrives."
)

LOW_CONFIDENCE_NOTE = (
    "Evidence confidence is limited, so the result should be interpreted "
    "cautiously."
)

CONFLICT_NOTE = (
    "Some evidence appears mixed or conflicting, so the conclusion should not "
    "be treated as a strong signal."
)

FORWARD_LOOKING_NOTE = (
    "The system does not forecast future returns; it summarizes current or "
    "historical evidence only."
)

UNSAFE_PHRASES = (
    "guaranteed",
    "guarantee",
    "risk-free",
    "sure thing",
    "will definitely",
    "certain to",
    "can't lose",
)


def _as_lower_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return " ".join(_as_lower_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_as_lower_text(item) for item in value.values())
    return str(value).lower()


def _has_unsafe_language(payload: Any) -> bool:
    text = _as_lower_text(payload)
    return any(phrase in text for phrase in UNSAFE_PHRASES)


def _confidence_value(payload: dict, state: dict | None = None) -> str:
    confidence = payload.get("confidence")
    if confidence is None and state:
        confidence = state.get("confidence")
    return str(confidence or "").strip().lower()


def _critic_conflicts(state: dict | None = None) -> list:
    if not state:
        return []
    return list(state.get("critic_result", {}).get("conflicts", []))


def _choose_disclaimer(payload: dict, state: dict | None = None) -> str:
    existing = payload.get("disclaimer")
    if existing:
        return existing

    if payload.get("type") == "comparison":
        return COMPARISON_DISCLAIMER

    if payload.get("recommendation") in {"BUY", "HOLD", "SELL"}:
        return DAILY_SIGNAL_DISCLAIMER

    intent = (state or {}).get("intent")
    if intent == "comparison":
        return COMPARISON_DISCLAIMER
    if intent == "buy_sell_decision":
        return DAILY_SIGNAL_DISCLAIMER

    return DEFAULT_DISCLAIMER


def _build_uncertainty_notes(payload: dict, state: dict | None = None) -> list[str]:
    notes = []
    confidence = _confidence_value(payload, state)

    if confidence in {"low", "limited", "weak"}:
        notes.append(LOW_CONFIDENCE_NOTE)

    if _critic_conflicts(state):
        notes.append(CONFLICT_NOTE)

    if _has_unsafe_language(payload):
        notes.append(FORWARD_LOOKING_NOTE)

    if not notes and payload.get("recommendation") in {"BUY", "HOLD", "SELL"}:
        notes.append(
            "The recommendation is a daily evidence-based signal and should be "
            "used alongside independent judgment."
        )

    return notes


def apply_compliance(
    response: dict | str,
    state: dict | None = None,
) -> dict:
    """
    Add safer wording, uncertainty language, and disclaimer metadata.

    This skill does not collect evidence or change the recommendation. It wraps
    an existing response/decision so the final answer is less overconfident.
    """
    if isinstance(response, str):
        payload = {"text": response}
    else:
        payload = deepcopy(response or {})

    payload["disclaimer"] = _choose_disclaimer(payload, state)
    payload["uncertainty_notes"] = _build_uncertainty_notes(payload, state)
    payload["safety"] = {
        "is_financial_advice": False,
        "personalized_advice": False,
        "predicts_future_prices": False,
        "unsafe_language_detected": _has_unsafe_language(response),
    }

    return payload


def run_compliance_skill(response: dict | str, state: dict | None = None) -> dict:
    return apply_compliance(response=response, state=state)


def add_disclaimer(response: dict | str, state: dict | None = None) -> dict:
    return apply_compliance(response=response, state=state)


__all__ = [
    "apply_compliance",
    "run_compliance_skill",
    "add_disclaimer",
]
