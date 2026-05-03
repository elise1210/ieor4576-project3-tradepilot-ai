def _planner_few_shot_examples() -> str:
    return (
        "Few-shot examples:\n"
        "Example 1\n"
        'Query: "Should I buy the Windows maker this month?"\n'
        'Output: {"intent":"buy_sell_decision","tickers":["MSFT"],"time_horizon":"short_term","needs_human":false,"clarification_question":null,"clarification_type":null,"clarification_options":[],"ticker_source":"llm_inference","ticker_inference_confidence":"medium","required_evidence":["news","market","fundamentals","sentiment"],"reasoning_brief":"The Windows maker refers to Microsoft and this is a buy/sell request."}\n'
        "Example 2\n"
        'Query: "Which has the stronger setup right now, Amazon or Meta?"\n'
        'Output: {"intent":"comparison","tickers":["AMZN","META"],"time_horizon":"short_term","needs_human":false,"clarification_question":null,"clarification_type":null,"clarification_options":[],"ticker_source":"llm_inference","ticker_inference_confidence":"medium","required_evidence":["news","market","fundamentals","sentiment"],"reasoning_brief":"The user is comparing two companies and asking for relative strength."}\n'
        "Example 3\n"
        'Query: "What is the market mood around Tesla lately?"\n'
        'Output: {"intent":"general_research","tickers":["TSLA"],"time_horizon":"unknown","needs_human":false,"clarification_question":null,"clarification_type":null,"clarification_options":[],"ticker_source":"company_name","ticker_inference_confidence":"medium","required_evidence":["news","sentiment"],"reasoning_brief":"The user is effectively asking for sentiment-style context, so news and sentiment are the minimum evidence."}\n'
        "Example 4\n"
        'Query: "What was Alphabet\'s closing price on 2026-03-18?"\n'
        'Output: {"intent":"general_research","tickers":["GOOGL"],"time_horizon":"unknown","needs_human":false,"clarification_question":null,"clarification_type":null,"clarification_options":[],"ticker_source":"company_name","ticker_inference_confidence":"medium","required_evidence":["market","chart"],"reasoning_brief":"This is a narrow price lookup, so market and chart are sufficient."}\n'
        "Example 5\n"
        'Query: "Why did the online retail giant move today?"\n'
        'Output: {"intent":"explanation","tickers":["AMZN"],"time_horizon":"short_term","needs_human":false,"clarification_question":null,"clarification_type":null,"clarification_options":[],"ticker_source":"llm_inference","ticker_inference_confidence":"medium","required_evidence":["news","market","fundamentals","chart"],"reasoning_brief":"This asks for a causal explanation of a recent move, so explanation evidence should include recent price behavior."}\n'
    )


def build_planner_system_prompt() -> str:
    return (
        "You are the planner for a stock-analysis app.\n"
        "Interpret the user query and return JSON only.\n"
        "Do not answer the user question itself.\n"
        "Allowed intent values: buy_sell_decision, comparison, explanation, general_research.\n"
        "Allowed time_horizon values: short_term, long_term, unknown.\n"
        "Allowed clarification_type values: ticker, time_horizon, custom.\n"
        "Allowed required_evidence values: news, market, fundamentals, sentiment, chart.\n"
        "Return exactly these fields: intent, tickers, time_horizon, needs_human, clarification_question, "
        "clarification_type, clarification_options, ticker_source, ticker_inference_confidence, "
        "required_evidence, reasoning_brief.\n"
        "tickers must be uppercase ticker symbols when known.\n"
        "If the company reference is indirect but clearly points to a well-known public company, infer the ticker.\n"
        "Examples of indirect references include product-based, role-based, or colloquial descriptions.\n"
        "Classify softer comparison wording such as 'which is better' as comparison when two companies are being contrasted.\n"
        "Classify softer opinion or mood wording such as 'how do you feel about' or 'what is the mood around' as "
        "general_research with required_evidence including news and sentiment when the user is effectively asking about sentiment.\n"
        "For narrow lookup questions, keep required_evidence narrow instead of requesting every tool.\n"
        "If clarification is genuinely needed, set needs_human to true and provide a concise clarification_question.\n"
        + _planner_few_shot_examples()
    )


def build_planner_user_instructions() -> list[str]:
    return [
        "Infer the most likely stock-analysis intent.",
        "Infer one or more tickers if the company reference is clear.",
        "Use required_evidence to express the minimum evidence the downstream system should gather.",
        "For sentiment-style questions, prefer required_evidence like news and sentiment.",
        "For price lookup questions, prefer required_evidence like market and chart.",
        "For explanation questions, include chart only when the user is asking about recent movement or price behavior.",
        "Set needs_human true only when clarification is genuinely needed.",
        "Use clarification_question only if needs_human is true.",
        "If needs_human is true, set clarification_type when you can and include clarification_options for fixed choices.",
        "If no ticker can be inferred, return an empty tickers list.",
    ]


__all__ = ["build_planner_system_prompt", "build_planner_user_instructions"]
