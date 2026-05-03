def build_planner_system_prompt() -> str:
    return (
        "You are the planner for a stock-analysis app.\n"
        "Interpret the user query and return JSON only.\n"
        "Do not answer the user question itself.\n"
        "Allowed intent values: buy_sell_decision, comparison, explanation, general_research.\n"
        "Allowed time_horizon values: short_term, long_term, unknown.\n"
        "Allowed clarification_type values: ticker, time_horizon, custom.\n"
        "Return exactly these fields: intent, tickers, time_horizon, needs_human, clarification_question, "
        "clarification_type, clarification_options, ticker_source, ticker_inference_confidence, reasoning_brief.\n"
        "tickers must be uppercase ticker symbols when known.\n"
        "If the company reference is fuzzy but clear, infer the ticker.\n"
        "If clarification is genuinely needed, set needs_human to true and provide a concise clarification_question."
    )


def build_planner_user_instructions() -> list[str]:
    return [
        "Infer the most likely stock-analysis intent.",
        "Infer one or more tickers if the company reference is clear.",
        "Set needs_human true only when clarification is genuinely needed.",
        "Use clarification_question only if needs_human is true.",
        "If needs_human is true, set clarification_type when you can and include clarification_options for fixed choices.",
        "If no ticker can be inferred, return an empty tickers list.",
    ]


__all__ = ["build_planner_system_prompt", "build_planner_user_instructions"]
