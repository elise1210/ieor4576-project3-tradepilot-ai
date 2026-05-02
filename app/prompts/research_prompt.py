from typing import Optional

from app.skills.schema import format_skill_schemas


DEFAULT_RESEARCH_SKILLS = ["news", "market", "fundamentals", "sentiment", "chart"]


def build_research_system_prompt(skill_names: Optional[list[str]] = None) -> str:
    schema_text = format_skill_schemas(skill_names or DEFAULT_RESEARCH_SKILLS)

    return (
        "You plan data-collection steps for a stock-analysis app.\n"
        "Your job is to choose which allowed skills to call, in what order, and with what parameters.\n"
        "Return JSON only. Do not answer the user query.\n"
        "Use the fewest steps that can still produce good evidence.\n"
        "Respect skill dependencies: sentiment should follow news; chart should follow market.\n"
        "If evidence is thin or the critic asked for follow-up work, you may widen the relevant interval or request a date-specific fetch.\n"
        "Do not invent new skills or parameters.\n\n"
        "Available skill schemas:\n"
        f"{schema_text}"
    )


def build_research_user_instructions() -> list[str]:
    return [
        "Prefer the minimum useful step set.",
        "If evidence is missing from a prior pass, you may widen the lookback window.",
        "For latest or today-style queries, keep windows short unless the evidence is thin.",
        "For explanation or trend questions, include market and news when useful.",
        "When a query is explicitly about a date, prefer target_date or requested_date style parameters.",
    ]


__all__ = ["DEFAULT_RESEARCH_SKILLS", "build_research_system_prompt", "build_research_user_instructions"]
