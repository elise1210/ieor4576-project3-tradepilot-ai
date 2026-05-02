import os
from typing import Callable, Dict, Optional

from dotenv import load_dotenv

from app.agents.llm_research import run_llm_research_planner
from app.state import clone_state


load_dotenv()


SkillRegistry = Dict[str, Callable]
EVIDENCE_KEYS = ("news", "market", "fundamentals", "sentiment", "charts")
PRIMARY_EVIDENCE_KEYS = {"news", "market", "fundamentals", "sentiment"}


def _empty_evidence_result(result: object) -> bool:
    if result is None:
        return True
    if isinstance(result, dict):
        if not result:
            return True
        if result.get("error"):
            return True
    if isinstance(result, list) and not result:
        return True
    return False


def _ensure_evidence_slots(state: dict) -> None:
    evidence = state.setdefault("evidence", {})
    for key in EVIDENCE_KEYS:
        if key == "charts":
            evidence.setdefault(key, [])
        else:
            evidence.setdefault(key, {})


def _set_gap(next_state: dict, gap: str) -> None:
    gaps = next_state.setdefault("gaps", [])
    if gap not in gaps:
        gaps.append(gap)


def _store_ticker_evidence(next_state: dict, evidence_type: str, ticker: str, payload: dict) -> None:
    evidence_bucket = next_state["evidence"].setdefault(evidence_type, {})
    evidence_bucket[ticker] = payload


def _store_chart_evidence(next_state: dict, ticker: str, payload: dict) -> None:
    charts = next_state["evidence"].setdefault("charts", [])
    chart_id = payload.get("chart_id")
    chart_kind = payload.get("kind")

    filtered = []
    for existing in charts:
        if not isinstance(existing, dict):
            filtered.append(existing)
            continue

        same_ticker = existing.get("ticker") == ticker
        same_chart_id = chart_id and existing.get("chart_id") == chart_id
        same_kind = chart_kind and existing.get("kind") == chart_kind

        if same_ticker and (same_chart_id or same_kind):
            continue
        filtered.append(existing)

    filtered.append(payload)
    next_state["evidence"]["charts"] = filtered


def _build_ticker_evidence_bundle(state: dict, ticker: str) -> dict:
    evidence = state.get("evidence", {})
    bundle = {}

    for evidence_type in ("news", "market", "fundamentals", "sentiment"):
        payload = evidence.get(evidence_type, {}).get(ticker)
        if isinstance(payload, dict) and payload:
            bundle[evidence_type] = payload

    return bundle


def _get_required_evidence(state: dict) -> list[str]:
    return list(state.get("plan", {}).get("required_evidence", []))


def _run_news_skill(skill: Callable, ticker: str, query: str, params: Optional[dict] = None) -> dict:
    params = params or {}
    call_args = {"ticker": ticker, "query": query}
    if "days" in params:
        call_args["days"] = params["days"]
    if "max_items" in params:
        call_args["max_items"] = params["max_items"]
    if "target_date" in params:
        call_args["target_date"] = params["target_date"]
    return skill(**call_args)


def _run_market_skill(skill: Callable, ticker: str, params: Optional[dict] = None) -> dict:
    params = params or {}
    call_args = {"ticker": ticker}
    if "lookback_days" in params:
        call_args["lookback_days"] = params["lookback_days"]
    elif "days" in params:
        call_args["lookback_days"] = params["days"]
    if "requested_date" in params:
        call_args["requested_date"] = params["requested_date"]
    elif "target_date" in params:
        call_args["requested_date"] = params["target_date"]
    return skill(**call_args)


def _run_fundamentals_skill(skill: Callable, ticker: str, params: Optional[dict] = None) -> dict:
    _ = params
    return skill(ticker=ticker)


def _run_sentiment_skill(skill: Callable, news_result: dict, params: Optional[dict] = None) -> dict:
    _ = params
    return skill(news_result=news_result)


def _run_chart_skill(
    skill: Callable,
    ticker: str,
    ticker_evidence: dict,
    query: str,
    params: Optional[dict] = None,
) -> dict:
    params = params or {}
    call_args = {
        "ticker": ticker,
        "evidence": ticker_evidence,
        "query": query,
    }
    if "requested_date" in params:
        call_args["reference_date"] = params["requested_date"]
    elif "target_date" in params:
        call_args["reference_date"] = params["target_date"]

    try:
        return skill(**call_args)
    except TypeError:
        call_args.pop("reference_date", None)
        return skill(**call_args)


def _execute_skill_step(next_state: dict, registry: SkillRegistry, query: str, step: dict) -> None:
    skill_name = step["skill"]
    ticker = step["ticker"]
    params = step.get("params", {})

    if skill_name == "news":
        skill = registry.get("news")
        if skill is None:
            _set_gap(next_state, f"missing_skill:news:{ticker}")
            return
        result = _run_news_skill(skill, ticker, query, params=params)
        if _empty_evidence_result(result):
            _set_gap(next_state, f"missing_evidence:news:{ticker}")
            return
        _store_ticker_evidence(next_state, "news", ticker, result)
        next_state["metadata"]["executed_research_steps"].append(step)
        return

    if skill_name == "market":
        skill = registry.get("market")
        if skill is None:
            _set_gap(next_state, f"missing_skill:market:{ticker}")
            return
        result = _run_market_skill(skill, ticker, params=params)
        if _empty_evidence_result(result):
            _set_gap(next_state, f"missing_evidence:market:{ticker}")
            return
        _store_ticker_evidence(next_state, "market", ticker, result)
        next_state["metadata"]["executed_research_steps"].append(step)
        return

    if skill_name == "fundamentals":
        skill = registry.get("fundamentals")
        if skill is None:
            _set_gap(next_state, f"missing_skill:fundamentals:{ticker}")
            return
        result = _run_fundamentals_skill(skill, ticker, params=params)
        if _empty_evidence_result(result):
            _set_gap(next_state, f"missing_evidence:fundamentals:{ticker}")
            return
        _store_ticker_evidence(next_state, "fundamentals", ticker, result)
        next_state["metadata"]["executed_research_steps"].append(step)
        return

    if skill_name == "sentiment":
        skill = registry.get("sentiment")
        news_result = next_state["evidence"].get("news", {}).get(ticker, {})
        if skill is None:
            _set_gap(next_state, f"missing_skill:sentiment:{ticker}")
            return
        if _empty_evidence_result(news_result):
            _set_gap(next_state, f"missing_dependency:news_for_sentiment:{ticker}")
            return
        result = _run_sentiment_skill(skill, news_result, params=params)
        if _empty_evidence_result(result):
            _set_gap(next_state, f"missing_evidence:sentiment:{ticker}")
            return
        _store_ticker_evidence(next_state, "sentiment", ticker, result)
        next_state["metadata"]["executed_research_steps"].append(step)
        return

    if skill_name == "chart":
        skill = registry.get("chart")
        if skill is None:
            _set_gap(next_state, f"missing_skill:chart:{ticker}")
            return
        ticker_bundle = _build_ticker_evidence_bundle(next_state, ticker)
        result = _run_chart_skill(skill, ticker, ticker_bundle, query, params=params)
        if _empty_evidence_result(result):
            _set_gap(next_state, f"missing_evidence:chart:{ticker}")
            return
        _store_chart_evidence(next_state, ticker, result)
        next_state["metadata"]["executed_research_steps"].append(step)


def _build_default_research_steps(state: dict) -> list[dict]:
    steps = []
    tickers = list(state.get("tickers", []))
    required_evidence = _get_required_evidence(state)

    for ticker in tickers:
        for evidence_type in ("news", "market", "fundamentals", "sentiment", "chart"):
            if evidence_type in required_evidence:
                steps.append({
                    "skill": evidence_type,
                    "ticker": ticker,
                    "params": {},
                })

    return steps


def _covers_required_evidence(state: dict, steps: list[dict]) -> bool:
    required = set(_get_required_evidence(state))
    tickers = list(state.get("tickers", []))

    for ticker in tickers:
        available = {
            step["skill"]
            for step in steps
            if step.get("ticker") == ticker
        }
        if not required.issubset(available):
            return False

    return True


def _llm_research_enabled() -> bool:
    value = os.getenv("USE_LLM_RESEARCH")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    return bool(os.getenv("OPENAI_API_KEY"))


def _build_research_steps(state: dict) -> tuple[list[dict], str, Optional[str]]:
    default_steps = _build_default_research_steps(state)

    if not _llm_research_enabled():
        return default_steps, "deterministic_only", None

    llm_plan = run_llm_research_planner(state)
    if llm_plan is None:
        return default_steps, "deterministic_fallback", None

    steps = llm_plan.get("steps", [])
    if not _covers_required_evidence(state, steps):
        return default_steps, "deterministic_fallback", llm_plan.get("reasoning_brief")

    return steps, "llm", llm_plan.get("reasoning_brief")


def run_research_agent(state: dict, skills: Optional[SkillRegistry] = None) -> dict:
    """
    Hybrid Research Agent.

    The agent optionally asks an LLM to build a structured fetch plan, but
    actual skill execution stays deterministic and validated in code.
    """
    next_state = clone_state(state)
    _ensure_evidence_slots(next_state)

    if next_state.get("needs_human"):
        _set_gap(next_state, "human_clarification_required")
        return next_state

    tickers = list(next_state.get("tickers", []))
    if not tickers:
        _set_gap(next_state, "missing_ticker")
        return next_state

    query = next_state.get("query", "")
    registry = skills or {}

    next_state["gaps"] = []
    next_state["metadata"]["executed_research_steps"] = []
    steps, research_mode, reasoning_brief = _build_research_steps(next_state)
    next_state["metadata"]["research_mode"] = research_mode
    next_state["metadata"]["research_reasoning_brief"] = reasoning_brief
    next_state["metadata"]["research_plan_steps"] = steps

    for step in steps:
        _execute_skill_step(next_state, registry, query, step)

    return next_state
