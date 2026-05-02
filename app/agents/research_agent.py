from typing import Callable, Dict, Optional

from app.state import clone_state


SkillRegistry = Dict[str, Callable]
EVIDENCE_KEYS = ("news", "market", "fundamentals", "sentiment", "charts")


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


def _run_news_skill(skill: Callable, ticker: str, query: str) -> dict:
    return skill(ticker=ticker, query=query)


def _run_market_skill(skill: Callable, ticker: str) -> dict:
    return skill(ticker=ticker)


def _run_fundamentals_skill(skill: Callable, ticker: str) -> dict:
    return skill(ticker=ticker)


def _run_sentiment_skill(skill: Callable, news_result: dict) -> dict:
    return skill(news_result=news_result)


def _run_chart_skill(skill: Callable, ticker: str, ticker_evidence: dict, query: str) -> dict:
    try:
        return skill(ticker=ticker, evidence=ticker_evidence, query=query)
    except TypeError:
        return skill(ticker=ticker, evidence=ticker_evidence)


def run_research_agent(state: dict, skills: Optional[SkillRegistry] = None) -> dict:
    """
    Deterministic Research Agent.

    The agent does not contain the content logic itself. Instead it calls
    injected skill functions, stores their outputs in RunState, and records
    any missing evidence gaps.
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
    required_evidence = _get_required_evidence(next_state)
    registry = skills or {}

    next_state["gaps"] = []

    for ticker in tickers:
        if "news" in required_evidence:
            news_skill = registry.get("news")
            if news_skill is None:
                _set_gap(next_state, f"missing_skill:news:{ticker}")
            else:
                news_result = _run_news_skill(news_skill, ticker, query)
                if _empty_evidence_result(news_result):
                    _set_gap(next_state, f"missing_evidence:news:{ticker}")
                else:
                    _store_ticker_evidence(next_state, "news", ticker, news_result)

        if "market" in required_evidence:
            market_skill = registry.get("market")
            if market_skill is None:
                _set_gap(next_state, f"missing_skill:market:{ticker}")
            else:
                market_result = _run_market_skill(market_skill, ticker)
                if _empty_evidence_result(market_result):
                    _set_gap(next_state, f"missing_evidence:market:{ticker}")
                else:
                    _store_ticker_evidence(next_state, "market", ticker, market_result)

        if "fundamentals" in required_evidence:
            fundamentals_skill = registry.get("fundamentals")
            if fundamentals_skill is None:
                _set_gap(next_state, f"missing_skill:fundamentals:{ticker}")
            else:
                fundamentals_result = _run_fundamentals_skill(fundamentals_skill, ticker)
                if _empty_evidence_result(fundamentals_result):
                    _set_gap(next_state, f"missing_evidence:fundamentals:{ticker}")
                else:
                    _store_ticker_evidence(next_state, "fundamentals", ticker, fundamentals_result)

        if "sentiment" in required_evidence:
            sentiment_skill = registry.get("sentiment")
            news_result = next_state["evidence"].get("news", {}).get(ticker, {})
            if sentiment_skill is None:
                _set_gap(next_state, f"missing_skill:sentiment:{ticker}")
            elif _empty_evidence_result(news_result):
                _set_gap(next_state, f"missing_dependency:news_for_sentiment:{ticker}")
            else:
                sentiment_result = _run_sentiment_skill(sentiment_skill, news_result)
                if _empty_evidence_result(sentiment_result):
                    _set_gap(next_state, f"missing_evidence:sentiment:{ticker}")
                else:
                    _store_ticker_evidence(next_state, "sentiment", ticker, sentiment_result)

        if "chart" in required_evidence:
            chart_skill = registry.get("chart")
            if chart_skill is None:
                _set_gap(next_state, f"missing_skill:chart:{ticker}")
            else:
                ticker_bundle = _build_ticker_evidence_bundle(next_state, ticker)
                chart_result = _run_chart_skill(chart_skill, ticker, ticker_bundle, query)
                if _empty_evidence_result(chart_result):
                    _set_gap(next_state, f"missing_evidence:chart:{ticker}")
                else:
                    next_state["evidence"]["charts"].append(chart_result)

    return next_state
