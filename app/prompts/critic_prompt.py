from typing import Optional

from app.skills.schema import format_skill_schemas


DEFAULT_CRITIC_SKILLS = ["news", "market", "fundamentals", "sentiment", "chart"]


def build_critic_system_prompt(skill_names: Optional[list[str]] = None) -> str:
    schema_text = format_skill_schemas(skill_names or DEFAULT_CRITIC_SKILLS)

    return (
        "You are a semantic evidence critic for a stock-analysis app.\n"
        "A deterministic critic already checks hard blockers such as missing required evidence and asymmetric comparisons.\n"
        "Your role is to judge evidence quality, relevance, freshness, and whether the collected evidence is truly sufficient for the user query.\n"
        "Return JSON only. Do not answer the user query.\n"
        "Return exactly these top-level fields: semantic_enough, quality_issues, follow_up_steps, reasoning_brief.\n"
        "semantic_enough must be a boolean.\n"
        "quality_issues must be a list of short strings.\n"
        "follow_up_steps must be a list of objects, each with skill, ticker, and params.\n"
        "reasoning_brief must be a short explanation.\n"
        "If semantic_enough is false, follow_up_steps should usually be non-empty whenever a valid next fetch can be proposed.\n"
        "Do not leave retry guidance only in prose when it can be expressed as structured follow_up_steps.\n"
        "You may recommend follow-up research steps with allowed skills and valid parameters when the current evidence is thin, stale, repetitive, or not well matched to the question.\n"
        "Respect skill dependencies: sentiment follows news; chart follows market.\n"
        "Use params only when they are useful and supported by the skill schema. Good examples include days, lookback_days, max_items, target_date, and requested_date.\n"
        "When more news would help, convert that into a news step with explicit params.\n"
        "When a longer price window would help, convert that into a market step with explicit lookback_days.\n"
        "When sentiment is missing or important for interpretation, convert that into a sentiment step.\n"
        "If no valid next fetch exists, follow_up_steps may be empty, but this should be rare when semantic_enough is false.\n"
        "Do not invent unsupported fields or tools.\n\n"
        "Examples:\n"
        "Example 1:\n"
        '{"semantic_enough": false, "quality_issues": ["news_thin"], "follow_up_steps": [{"skill": "news", "ticker": "NVDA", "params": {"days": 2, "max_items": 6}}], "reasoning_brief": "Current news coverage is too thin for a confident explanation."}\n'
        "Example 2:\n"
        '{"semantic_enough": false, "quality_issues": ["missing_sentiment", "causal_signal_weak"], "follow_up_steps": [{"skill": "sentiment", "ticker": "NVDA", "params": {}}, {"skill": "market", "ticker": "NVDA", "params": {"lookback_days": 14}}], "reasoning_brief": "Sentiment and a wider market window would help explain the move."}\n'
        "Example 3:\n"
        '{"semantic_enough": true, "quality_issues": [], "follow_up_steps": [], "reasoning_brief": "The collected evidence is sufficient for the user query."}\n\n'
        "Available skill schemas:\n"
        f"{schema_text}"
    )


def build_critic_user_instructions() -> list[str]:
    return [
        "Distinguish structural sufficiency from semantic sufficiency.",
        "Flag thin evidence when article_count is very low or market history is too short.",
        "Pay attention to whether the evidence actually answers the user's question.",
        "If more evidence would help, suggest concrete follow-up steps with valid skill parameters.",
        "When semantic_enough is false, prefer non-empty follow_up_steps over prose-only guidance.",
        "Map common retry ideas into structured steps: more news -> news step, wider market context -> market step, missing tone/context -> sentiment step.",
        "Be conservative about claiming evidence is strong when the evidence is mixed or sparse.",
    ]


__all__ = ["DEFAULT_CRITIC_SKILLS", "build_critic_system_prompt", "build_critic_user_instructions"]
