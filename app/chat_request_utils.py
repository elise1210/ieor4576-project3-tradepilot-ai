from typing import Optional


def clarification_value_to_text(clarification_type: Optional[str], clarification_value: str) -> str:
    value = (clarification_value or "").strip()
    if not value:
        return ""

    if clarification_type == "time_horizon":
        normalized = value.lower()
        if normalized == "short_term":
            return "I want a short-term trading view."
        if normalized == "long_term":
            return "I want a longer-term investment view."

    if clarification_type == "ticker":
        return f"The company or ticker is {value}."

    return value


def resolve_effective_query(
    query: Optional[str] = None,
    original_query: Optional[str] = None,
    clarification_type: Optional[str] = None,
    clarification_value: Optional[str] = None,
) -> str:
    direct_query = (query or "").strip()
    if direct_query:
        return direct_query

    original = (original_query or "").strip()
    clarification = (clarification_value or "").strip()
    if original and clarification:
        clarification_text = clarification_value_to_text(clarification_type, clarification)
        if clarification_text:
            return f"{original}\n\nClarification: {clarification_text}"

    raise ValueError("Provide either query, or original_query plus clarification_value.")


__all__ = ["clarification_value_to_text", "resolve_effective_query"]
