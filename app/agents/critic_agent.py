import os

from dotenv import load_dotenv

from app.agents.llm_critic import run_llm_critic
from app.state import clone_state


load_dotenv()


def _has_usable_payload(payload: object) -> bool:
    if payload is None:
        return False
    if isinstance(payload, dict):
        return bool(payload) and not payload.get("error")
    if isinstance(payload, list):
        return bool(payload)
    return True


def _required_evidence(state: dict) -> list[str]:
    return list(state.get("plan", {}).get("required_evidence", []))


def _required_without_chart(state: dict) -> list[str]:
    return [item for item in _required_evidence(state) if item != "chart"]


def _requires_evidence_type(state: dict, evidence_type: str) -> bool:
    return evidence_type in set(_required_without_chart(state))


def _usable_for(state: dict, evidence_type: str, ticker: str) -> bool:
    payload = state.get("evidence", {}).get(evidence_type, {}).get(ticker, {})
    return _has_usable_payload(payload)


def _single_ticker_missing_breakdown(state: dict) -> tuple[list[str], list[str]]:
    tickers = list(state.get("tickers", []))
    if not tickers:
        return [], []

    ticker = tickers[0]
    blocking_missing = []
    supporting_missing = []

    has_market = _usable_for(state, "market", ticker)
    has_news = _usable_for(state, "news", ticker)
    has_sentiment = _usable_for(state, "sentiment", ticker)
    has_fundamentals = _usable_for(state, "fundamentals", ticker)

    if _requires_evidence_type(state, "market") and not has_market:
        blocking_missing.append(f"market:{ticker}")

    if _requires_evidence_type(state, "news") and not has_news:
        blocking_missing.append(f"news:{ticker}")

    if _requires_evidence_type(state, "sentiment") and not has_sentiment:
        blocking_missing.append(f"sentiment:{ticker}")

    if _requires_evidence_type(state, "fundamentals") and not has_fundamentals:
        blocking_missing.append(f"fundamentals:{ticker}")

    return blocking_missing, supporting_missing


def _comparison_missing_breakdown(state: dict) -> tuple[list[str], list[str]]:
    tickers = list(state.get("tickers", []))
    blocking_missing = []
    supporting_missing = []

    for ticker in tickers:
        has_market = _usable_for(state, "market", ticker)
        has_news = _usable_for(state, "news", ticker)
        has_sentiment = _usable_for(state, "sentiment", ticker)
        has_fundamentals = _usable_for(state, "fundamentals", ticker)

        if _requires_evidence_type(state, "market") and not has_market:
            blocking_missing.append(f"market:{ticker}")

        if _requires_evidence_type(state, "news") and not has_news:
            blocking_missing.append(f"news:{ticker}")

        if _requires_evidence_type(state, "sentiment") and not has_sentiment:
            blocking_missing.append(f"sentiment:{ticker}")

        if _requires_evidence_type(state, "fundamentals") and not has_fundamentals:
            blocking_missing.append(f"fundamentals:{ticker}")

    return blocking_missing, supporting_missing


def _missing_breakdown(state: dict) -> tuple[list[str], list[str]]:
    tickers = list(state.get("tickers", []))
    if len(tickers) <= 1:
        return _single_ticker_missing_breakdown(state)
    return _comparison_missing_breakdown(state)


def _comparison_fairness_issues(state: dict) -> list[str]:
    tickers = list(state.get("tickers", []))
    required = _required_without_chart(state)
    if len(tickers) < 2:
        return []

    evidence = state.get("evidence", {})
    issues = []

    for evidence_type in required:
        present = []
        for ticker in tickers:
            payload = evidence.get(evidence_type, {}).get(ticker, {})
            present.append(_has_usable_payload(payload))
        if any(present) and not all(present):
            issues.append(f"asymmetric_{evidence_type}")

    return issues


def _conflict_flags(state: dict) -> list[str]:
    tickers = list(state.get("tickers", []))
    evidence = state.get("evidence", {})
    conflicts = []

    for ticker in tickers:
        market = evidence.get("market", {}).get(ticker, {})
        sentiment = evidence.get("sentiment", {}).get(ticker, {})

        trend = str(market.get("trend_label", "")).lower()
        sentiment_label = str(sentiment.get("sentiment", "")).lower()

        if trend == "upward" and sentiment_label == "negative":
            conflicts.append(f"trend_vs_sentiment:{ticker}")
        elif trend == "downward" and sentiment_label == "positive":
            conflicts.append(f"trend_vs_sentiment:{ticker}")

    return conflicts


def _follow_up_tasks(missing: list[str], fairness_issues: list[str]) -> list[str]:
    tasks = []
    for item in missing:
        evidence_type, ticker = item.split(":", 1)
        tasks.append(f"collect_{evidence_type}:{ticker}")

    for issue in fairness_issues:
        tasks.append(f"resolve_{issue}")

    return tasks


def _llm_critic_enabled() -> bool:
    value = os.getenv("USE_LLM_CRITIC")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(os.getenv("OPENAI_API_KEY"))


def _deterministic_critic_result(state: dict) -> dict:
    if state.get("needs_human"):
        return {
            "enough_evidence": False,
            "missing": ["human_clarification_required"],
            "blocking_missing": ["human_clarification_required"],
            "supporting_missing": [],
            "fairness_issues": [],
            "conflicts": [],
            "follow_up_tasks": [],
            "confidence": "Low",
        }

    blocking_missing, supporting_missing = _missing_breakdown(state)
    missing = sorted(set(blocking_missing + supporting_missing))
    fairness_issues = _comparison_fairness_issues(state)
    conflicts = _conflict_flags(state)

    enough_evidence = not blocking_missing and not fairness_issues and bool(state.get("tickers"))
    follow_up_tasks = _follow_up_tasks(missing, fairness_issues)
    confidence = "High"
    if not enough_evidence or blocking_missing or fairness_issues:
        confidence = "Low"
    elif supporting_missing or conflicts:
        confidence = "Medium"

    return {
        "enough_evidence": enough_evidence,
        "missing": missing,
        "blocking_missing": blocking_missing,
        "supporting_missing": supporting_missing,
        "fairness_issues": fairness_issues,
        "conflicts": conflicts,
        "follow_up_tasks": follow_up_tasks,
        "confidence": confidence,
    }


def run_critic_agent(state: dict) -> dict:
    next_state = clone_state(state)

    deterministic = _deterministic_critic_result(next_state)
    semantic_enough = None
    quality_issues = []
    llm_follow_up_steps = []
    critic_mode = "deterministic_only"
    reasoning_brief = None

    if _llm_critic_enabled() and next_state.get("tickers"):
        llm_result = run_llm_critic(next_state, deterministic)
        if llm_result is None:
            critic_mode = "deterministic_fallback"
        else:
            critic_mode = "llm"
            semantic_enough = llm_result["semantic_enough"]
            quality_issues = llm_result.get("quality_issues", [])
            llm_follow_up_steps = llm_result.get("follow_up_steps", [])
            reasoning_brief = llm_result.get("reasoning_brief")

    enough_evidence = deterministic["enough_evidence"]
    if semantic_enough is False:
        enough_evidence = False

    confidence = deterministic["confidence"]
    if semantic_enough is False:
        confidence = "Low"
    elif quality_issues and confidence == "High":
        confidence = "Medium"

    critic_result = dict(deterministic)
    critic_result["enough_evidence"] = enough_evidence
    critic_result["semantic_enough"] = semantic_enough
    critic_result["quality_issues"] = quality_issues
    critic_result["llm_follow_up_steps"] = llm_follow_up_steps
    critic_result["confidence"] = confidence

    next_state["critic_result"] = critic_result
    next_state["confidence"] = confidence
    next_state["metadata"]["critic_mode"] = critic_mode
    next_state["metadata"]["critic_reasoning_brief"] = reasoning_brief

    return next_state
