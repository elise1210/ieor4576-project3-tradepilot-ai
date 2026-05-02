import json
import os
import re
import urllib.request
from typing import Optional

from dotenv import load_dotenv

from app.prompts.critic_prompt import (
    build_critic_system_prompt,
    build_critic_user_instructions,
)


load_dotenv()


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_CRITIC_MODEL = os.getenv("OPENAI_CRITIC_MODEL", "gpt-4o-mini")
ALLOWED_SKILLS = {"news", "market", "fundamentals", "sentiment", "chart"}


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


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
    return output


def _normalize_follow_up_steps(value: object, tickers: list[str]) -> list[dict]:
    if not isinstance(value, list):
        return []

    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        skill = item.get("skill")
        ticker = item.get("ticker")
        params = item.get("params", {})
        if skill not in ALLOWED_SKILLS:
            continue
        if not isinstance(ticker, str) or ticker.upper().strip() not in tickers:
            continue
        if not isinstance(params, dict):
            params = {}

        clean_params = {}
        for key in ("days", "lookback_days", "max_items"):
            val = params.get(key)
            if isinstance(val, int) and val > 0:
                clean_params[key] = val
        for key in ("target_date", "requested_date"):
            val = params.get(key)
            if isinstance(val, str) and val.strip():
                clean_params[key] = val.strip()

        output.append({
            "skill": skill,
            "ticker": ticker.upper().strip(),
            "params": clean_params,
        })

    return output


def _normalize_critic_output(data: dict, tickers: list[str]) -> Optional[dict]:
    if not isinstance(data, dict):
        return None

    semantic_enough = data.get("semantic_enough")
    if not isinstance(semantic_enough, bool):
        return None

    reasoning_brief = data.get("reasoning_brief")
    if not isinstance(reasoning_brief, str) or not reasoning_brief.strip():
        reasoning_brief = None

    return {
        "semantic_enough": semantic_enough,
        "quality_issues": _normalize_string_list(data.get("quality_issues")),
        "follow_up_steps": _normalize_follow_up_steps(data.get("follow_up_steps"), tickers),
        "reasoning_brief": reasoning_brief,
    }


def run_llm_critic(state: dict, deterministic_critic_result: dict, model: Optional[str] = None, timeout: int = 12) -> Optional[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    tickers = list(state.get("tickers", []))
    if not tickers:
        return None

    payload = {
        "model": model or os.getenv("OPENAI_CRITIC_MODEL", OPENAI_CRITIC_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": build_critic_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": state.get("query", ""),
                        "intent": state.get("intent"),
                        "tickers": tickers,
                        "deterministic_critic_result": deterministic_critic_result,
                        "evidence": state.get("evidence", {}),
                        "instructions": build_critic_user_instructions(),
                        "required_json_fields": [
                            "semantic_enough",
                            "quality_issues",
                            "follow_up_steps",
                            "reasoning_brief",
                        ],
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
        return _normalize_critic_output(parsed, tickers)
    except Exception:
        return None


__all__ = ["run_llm_critic"]
