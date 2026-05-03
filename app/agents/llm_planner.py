import json
import os
import re
import urllib.request
from typing import Optional

from dotenv import load_dotenv

from app.prompts.planner_prompt import (
    build_planner_system_prompt,
    build_planner_user_instructions,
)


load_dotenv()


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", "gpt-4o-mini")
ALLOWED_INTENTS = {
    "buy_sell_decision",
    "comparison",
    "explanation",
    "general_research",
}
ALLOWED_TIME_HORIZONS = {"short_term", "long_term", "unknown"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_CLARIFICATION_TYPES = {"ticker", "time_horizon", "custom"}
ALLOWED_REQUIRED_EVIDENCE = {"news", "market", "fundamentals", "sentiment", "chart"}


def _normalize_clarification_options(value: object) -> Optional[list[dict]]:
    if value is None:
        return None
    if not isinstance(value, list):
        return None

    options = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        option_value = item.get("value")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(option_value, str) or not option_value.strip():
            continue
        options.append({
            "label": label.strip(),
            "value": option_value.strip(),
        })
    return options


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


def _normalize_tickers(value: object) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        return []

    tickers = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        ticker = re.sub(r"[^A-Za-z]", "", candidate).upper().strip()
        if not ticker or len(ticker) > 5:
            continue
        if ticker not in tickers:
            tickers.append(ticker)

    return tickers


def _normalize_required_evidence(value: object) -> Optional[list[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        return None

    cleaned = []
    for item in value:
        if not isinstance(item, str):
            continue
        evidence_type = item.strip().lower()
        if evidence_type not in ALLOWED_REQUIRED_EVIDENCE:
            continue
        if evidence_type not in cleaned:
            cleaned.append(evidence_type)
    return cleaned or None


def _normalize_llm_planner_output(data: dict, provided_ticker: Optional[str] = None) -> Optional[dict]:
    if not isinstance(data, dict):
        return None

    intent = data.get("intent")
    if intent not in ALLOWED_INTENTS:
        return None

    time_horizon = data.get("time_horizon", "unknown")
    if time_horizon not in ALLOWED_TIME_HORIZONS:
        time_horizon = "unknown"

    tickers = _normalize_tickers(data.get("tickers", []))
    if provided_ticker:
        tickers = [provided_ticker.upper().strip()]

    needs_human = bool(data.get("needs_human", False))
    clarification_question = data.get("clarification_question")
    if not isinstance(clarification_question, str) or not clarification_question.strip():
        clarification_question = None
    else:
        clarification_question = clarification_question.strip()

    clarification_type = data.get("clarification_type")
    if clarification_type is not None:
        clarification_type = str(clarification_type).strip().lower()
    if clarification_type not in ALLOWED_CLARIFICATION_TYPES:
        clarification_type = None

    clarification_options = _normalize_clarification_options(data.get("clarification_options"))

    ticker_source = data.get("ticker_source")
    if not isinstance(ticker_source, str) or not ticker_source.strip():
        ticker_source = "provided" if provided_ticker else ("llm" if tickers else "unknown")

    confidence = str(data.get("ticker_inference_confidence", "")).lower()
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "high" if provided_ticker else ("medium" if tickers else None)

    reasoning_brief = data.get("reasoning_brief")
    if not isinstance(reasoning_brief, str) or not reasoning_brief.strip():
        reasoning_brief = None

    required_evidence = _normalize_required_evidence(data.get("required_evidence"))

    return {
        "intent": intent,
        "tickers": tickers,
        "time_horizon": time_horizon,
        "needs_human": needs_human,
        "clarification_question": clarification_question,
        "clarification_type": clarification_type,
        "clarification_options": clarification_options,
        "ticker_source": ticker_source,
        "ticker_inference_confidence": confidence,
        "required_evidence": required_evidence,
        "reasoning_brief": reasoning_brief,
    }


def run_llm_planner(
    query: str,
    provided_ticker: Optional[str] = None,
    company_name_to_ticker: Optional[dict] = None,
    model: Optional[str] = None,
    timeout: int = 12,
) -> Optional[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    known_companies = company_name_to_ticker or {}

    payload = {
        "model": model or os.getenv("OPENAI_PLANNER_MODEL", OPENAI_PLANNER_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": build_planner_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "provided_ticker": provided_ticker,
                        "known_company_map": known_companies,
                        "instructions": build_planner_user_instructions(),
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
        return _normalize_llm_planner_output(parsed, provided_ticker=provided_ticker)
    except Exception:
        return None


__all__ = ["run_llm_planner"]
