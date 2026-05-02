from typing import Optional


SKILL_SCHEMAS = {
    "news": {
        "purpose": (
            "Collect recent company-specific news and summarize the main current catalysts."
        ),
        "dependencies": [],
        "inputs": {
            "ticker": "Required uppercase ticker symbol.",
            "query": "Optional raw user query for relevance filtering.",
            "days": "Optional recent lookback window in calendar days. Default is 7.",
            "target_date": "Optional explicit date for date-specific news retrieval.",
            "max_items": "Optional cap on number of selected articles. Default is 8.",
        },
        "outputs": {
            "ticker": "Ticker used for the fetch.",
            "summary": "News summary string for downstream reasoning.",
            "items": "List of selected article objects.",
            "article_count": "Number of selected articles.",
            "requested_date": "Explicit requested date when date-specific mode is used.",
            "start_date": "News window start date.",
            "end_date": "News window end date.",
            "summary_source": "openai or rule_based.",
        },
        "quality_hints": [
            "article_count under 2 may indicate thin evidence.",
            "If the query asks about a specific date, target_date should usually be set.",
            "If articles are sparse, widening the days window can help.",
        ],
    },
    "market": {
        "purpose": (
            "Collect recent price behavior and derive short-term trend and volatility context."
        ),
        "dependencies": [],
        "inputs": {
            "ticker": "Required uppercase ticker symbol.",
            "lookback_days": "Optional number of recent trading days to analyze. Default is 7.",
            "requested_date": "Optional explicit end date for historical lookback mode.",
        },
        "outputs": {
            "ticker": "Ticker used for the fetch.",
            "current_price": "Latest close in the selected window.",
            "start_price_7d": "Starting close for the chosen lookback window.",
            "trend_7d": "Relative price move over the window.",
            "trend_label": "upward, downward, or sideways.",
            "volatility": "Short-term volatility estimate.",
            "history": "List of OHLCV-like recent points for charting.",
            "requested_date": "Explicit requested end date when historical mode is used.",
            "used_end_date": "Actual last date represented in the returned history.",
            "start_date": "First date in the returned history.",
            "end_date": "Last date in the returned history.",
        },
        "quality_hints": [
            "history with fewer than 3 points weakens trend interpretation.",
            "For date-specific questions, requested_date should usually be set.",
            "For latest or today-style questions, a short lookback window is usually enough.",
        ],
    },
    "fundamentals": {
        "purpose": (
            "Provide lightweight company context, valuation, and risk information."
        ),
        "dependencies": [],
        "inputs": {
            "ticker": "Required uppercase ticker symbol.",
        },
        "outputs": {
            "ticker": "Ticker used for the fetch.",
            "company_name": "Resolved company name.",
            "industry": "Company industry classification.",
            "country": "Company country.",
            "market_cap": "Market capitalization from provider data.",
            "market_cap_bucket": "Small/large/mega-cap style bucket.",
            "pe_ttm": "Trailing price-to-earnings ratio when available.",
            "beta": "Beta when available.",
            "summary": "Compact company-context summary.",
        },
        "quality_hints": [
            "Fundamentals are supporting evidence for most current-signal queries.",
            "If fundamentals fail, a cautious answer may still be possible when market and news are strong.",
        ],
    },
    "sentiment": {
        "purpose": (
            "Score the tone of selected company news and summarize aggregate sentiment."
        ),
        "dependencies": ["news"],
        "inputs": {
            "news_result": "Preferred input: the already selected news payload for a ticker.",
        },
        "outputs": {
            "ticker": "Ticker propagated from the news payload.",
            "sentiment": "positive, neutral, or negative.",
            "score": "Aggregate sentiment score.",
            "dispersion": "How mixed article-level sentiment is.",
            "article_count": "Number of articles scored.",
            "summary": "Compact sentiment summary string.",
            "model_available": "Whether the sentiment model was available.",
        },
        "quality_hints": [
            "sentiment should normally be run after news and on the same selected article set.",
            "Very low article_count can make sentiment less reliable.",
            "High dispersion means sentiment is mixed and confidence should be lower.",
        ],
    },
    "chart": {
        "purpose": (
            "Convert recent market evidence into a frontend-ready chart specification."
        ),
        "dependencies": ["market"],
        "inputs": {
            "ticker": "Required uppercase ticker symbol.",
            "evidence": "Ticker evidence bundle, especially market history.",
            "query": "Raw user query used for chart gating.",
            "reference_date": "Optional explicit date context for date parsing.",
        },
        "outputs": {
            "ticker": "Ticker represented by the chart.",
            "chart_id": "Stable chart identifier.",
            "kind": "Chart semantic type.",
            "charts": "Frontend-renderable chart specs.",
            "highlights": "Short bullets explaining the price pattern.",
            "note": "Safety note about chart interpretation.",
        },
        "quality_hints": [
            "Charts are presentation artifacts, not primary evidence.",
            "Chart should usually run only for latest/today price or daily-signal style questions.",
        ],
    },
}


def get_skill_schema(skill_name: str) -> Optional[dict]:
    return SKILL_SCHEMAS.get(skill_name)


def format_skill_schema(skill_name: str) -> str:
    schema = get_skill_schema(skill_name)
    if not schema:
        return ""

    lines = [
        f"Skill: {skill_name}",
        f"Purpose: {schema['purpose']}",
    ]

    dependencies = schema.get("dependencies", [])
    if dependencies:
        lines.append("Dependencies: " + ", ".join(dependencies))
    else:
        lines.append("Dependencies: none")

    lines.append("Inputs:")
    for field, description in schema.get("inputs", {}).items():
        lines.append(f"- {field}: {description}")

    lines.append("Outputs:")
    for field, description in schema.get("outputs", {}).items():
        lines.append(f"- {field}: {description}")

    quality_hints = schema.get("quality_hints", [])
    if quality_hints:
        lines.append("Quality hints:")
        for hint in quality_hints:
            lines.append(f"- {hint}")

    return "\n".join(lines)


def format_skill_schemas(skill_names: Optional[list[str]] = None) -> str:
    names = skill_names or list(SKILL_SCHEMAS.keys())
    sections = []
    for skill_name in names:
        section = format_skill_schema(skill_name)
        if section:
            sections.append(section)
    return "\n\n".join(sections)


__all__ = ["SKILL_SCHEMAS", "get_skill_schema", "format_skill_schema", "format_skill_schemas"]
