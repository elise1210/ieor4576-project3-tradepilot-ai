import json
import os
import re
import urllib.request
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DECISION_MODEL = os.getenv("OPENAI_DECISION_MODEL", "gpt-4o-mini")


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


def _normalize_reasoning_list(value: object) -> Optional[list[str]]:
    if not isinstance(value, list):
        return None

    output = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            output.append(text)

    return output or None


def _normalize_per_ticker(data: object, tickers: list[str]) -> dict:
    if not isinstance(data, dict):
        return {}

    output = {}
    for ticker in tickers:
        payload = data.get(ticker)
        if not isinstance(payload, dict):
            continue

        normalized = {}
        reasoning = _normalize_reasoning_list(payload.get("reasoning"))
        if reasoning is not None:
            normalized["reasoning"] = reasoning

        key_driver = payload.get("key_driver")
        if isinstance(key_driver, str) and key_driver.strip():
            normalized["key_driver"] = key_driver.strip()

        if normalized:
            output[ticker] = normalized

    return output


def _normalize_decision_output(data: dict, tickers: list[str], is_comparison: bool) -> Optional[dict]:
    if not isinstance(data, dict):
        return None

    normalized = {
        "reasoning_brief": None,
    }

    reasoning_brief = data.get("reasoning_brief")
    if isinstance(reasoning_brief, str) and reasoning_brief.strip():
        normalized["reasoning_brief"] = reasoning_brief.strip()

    if is_comparison:
        comparison_summary = data.get("comparison_summary")
        if isinstance(comparison_summary, str) and comparison_summary.strip():
            normalized["comparison_summary"] = comparison_summary.strip()

        per_ticker = _normalize_per_ticker(data.get("per_ticker"), tickers)
        if per_ticker:
            normalized["per_ticker"] = per_ticker

        return normalized if len(normalized) > 1 else None

    reasoning = _normalize_reasoning_list(data.get("reasoning"))
    if reasoning is not None:
        normalized["reasoning"] = reasoning

    key_driver = data.get("key_driver")
    if isinstance(key_driver, str) and key_driver.strip():
        normalized["key_driver"] = key_driver.strip()

    return normalized if len(normalized) > 1 else None


def run_llm_decision_synthesizer(state: dict, draft_decision: dict, model: Optional[str] = None, timeout: int = 12) -> Optional[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    tickers = list(state.get("tickers", []))
    is_comparison = draft_decision.get("type") == "comparison"

    payload = {
        "model": model or os.getenv("OPENAI_DECISION_MODEL", OPENAI_DECISION_MODEL),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You improve the wording of a stock-analysis app's final decision. "
                    "Do not change recommendations, confidence, risk, or scores. "
                    "Use only the provided evidence and draft decision. "
                    "Do not give personalized financial advice or future price predictions. "
                    "Return JSON only. "
                    "For single-stock decisions return: reasoning, key_driver, reasoning_brief. "
                    "For comparison decisions return: comparison_summary, per_ticker, reasoning_brief. "
                    "Each per_ticker item may contain reasoning and key_driver."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": state.get("query", ""),
                        "intent": state.get("intent"),
                        "tickers": tickers,
                        "critic_result": state.get("critic_result", {}),
                        "evidence": state.get("evidence", {}),
                        "draft_decision": draft_decision,
                        "instructions": [
                            "Preserve the original recommendation direction implied by the draft.",
                            "Be explicit about mixed evidence when present.",
                            "Keep reasoning concise and grounded in the provided evidence.",
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
        return _normalize_decision_output(parsed, tickers=tickers, is_comparison=is_comparison)
    except Exception:
        return None


__all__ = ["run_llm_decision_synthesizer"]
