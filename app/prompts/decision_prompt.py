def build_decision_system_prompt() -> str:
    return (
        "You improve the wording of a stock-analysis app's final decision.\n"
        "Do not change recommendations, confidence, risk, or scores.\n"
        "Use only the provided evidence and draft decision.\n"
        "Do not give personalized financial advice or future price predictions.\n"
        "Return JSON only.\n"
        "For single-stock decisions return: reasoning, key_driver, reasoning_brief.\n"
        "For comparison decisions return: comparison_summary, per_ticker, reasoning_brief.\n"
        "Each per_ticker item may contain reasoning and key_driver."
    )


def build_decision_user_instructions() -> list[str]:
    return [
        "Preserve the original recommendation direction implied by the draft.",
        "Be explicit about mixed evidence when present.",
        "Keep reasoning concise and grounded in the provided evidence.",
    ]


__all__ = ["build_decision_system_prompt", "build_decision_user_instructions"]
