import json
import os
import re
import urllib.request
from typing import Optional

from dotenv import load_dotenv

from app.prompts.research_prompt import (
    build_research_system_prompt,
    build_research_user_instructions,
)


load_dotenv()


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RESEARCH_MODEL = os.getenv("OPENAI_RESEARCH_MODEL", "gpt-4o-mini")
ALLOWED_SKILLS = {"news", "market", "fundamentals", "sentiment", "chart"}
ALLOWED_DEPENDENCIES = {"news", "market"}


def _extract_json_object(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_step(step: object, valid_tickers: list[str]) -> Optional[dict]:
    if not isinstance(step, dict):
        return None

    skill = step.get("skill")
    if skill not in ALLOWED_SKILLS:
        return None

    ticker = step.get("ticker")
    if not isinstance(ticker, str):
        return None
    ticker = ticker.upper().strip()
    if ticker not in valid_tickers:
        return None

    params = step.get("params", {})
    if not isinstance(params, dict):
        params = {}

    clean_params = {}
    for key in ("days", "lookback_days", "max_items"):
        value = params.get(key)
        if isinstance(value, int) and value > 0:
            clean_params[key] = value

    for key in ("target_date", "requested_date"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            clean_params[key] = value.strip()

    depends_on = step.get("depends_on")
    if depends_on not in ALLOWED_DEPENDENCIES:
        depends_on = None

    normalized = {
        "skill": skill,
        "ticker": ticker,
        "params": clean_params,
    }
    if depends_on is not None:
        normalized["depends_on"] = depends_on
    return normalized


def _normalize_research_plan(data: dict, tickers: list[str]) -> Optional[dict]:
    if not isinstance(data, dict):
        return None

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    steps = []
    for step in raw_steps:
        normalized = _normalize_step(step, valid_tickers=tickers)
        if normalized is not None:
            steps.append(normalized)

    if not steps:
        return None

    reasoning_brief = data.get("reasoning_brief")
    if not isinstance(reasoning_brief, str) or not reasoning_brief.strip():
        reasoning_brief = None

    return {
        "steps": steps,
        "reasoning_brief": reasoning_brief,
    }


def run_llm_research_planner(
    state: dict,
    model: Optional[str] = None,
    timeout: int = 12,
) -> Optional[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    tickers = list(state.get("tickers", []))
    if not tickers:
        return None

    payload = {
        "model": model or os.getenv("OPENAI_RESEARCH_MODEL", OPENAI_RESEARCH_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": build_research_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": state.get("query", ""),
                        "intent": state.get("intent"),
                        "tickers": tickers,
                        "required_evidence": state.get("plan", {}).get("required_evidence", []),
                        "existing_evidence_keys": {
                            key: list(value.keys()) if isinstance(value, dict) else []
                            for key, value in state.get("evidence", {}).items()
                            if key != "charts"
                        },
                        "gaps": state.get("gaps", []),
                        "critic_result": state.get("critic_result", {}),
                        "instructions": build_research_user_instructions(),
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    }

    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if parsed is None:
            return None
        return _normalize_research_plan(parsed, tickers=tickers)
    except Exception:
        return None


__all__ = ["run_llm_research_planner"]
