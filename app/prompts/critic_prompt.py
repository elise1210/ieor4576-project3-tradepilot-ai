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
        "You may recommend follow-up research steps with allowed skills and valid parameters when the current evidence is thin, stale, repetitive, or not well matched to the question.\n"
        "Respect skill dependencies: sentiment follows news; chart follows market.\n"
        "Do not invent unsupported fields or tools.\n\n"
        "Available skill schemas:\n"
        f"{schema_text}"
    )


def build_critic_user_instructions() -> list[str]:
    return [
        "Distinguish structural sufficiency from semantic sufficiency.",
        "Flag thin evidence when article_count is very low or market history is too short.",
        "Pay attention to whether the evidence actually answers the user's question.",
        "If more evidence would help, suggest concrete follow-up steps with valid skill parameters.",
        "Be conservative about claiming evidence is strong when the evidence is mixed or sparse.",
    ]


__all__ = ["DEFAULT_CRITIC_SKILLS", "build_critic_system_prompt", "build_critic_user_instructions"]
