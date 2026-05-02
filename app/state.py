from copy import deepcopy
from typing import Optional


DEFAULT_MAX_ITERATIONS = 2


def build_initial_state(query: str, ticker: Optional[str] = None) -> dict:
    """
    Build the shared RunState used by the agent pipeline.

    The state is intentionally a plain dict for now so the first
    implementation stays simple and easy to test.
    """
    clean_query = (query or "").strip()
    clean_ticker = (ticker or "").strip().upper()

    return {
        "query": clean_query,
        "user_inputs": {
            "raw_query": clean_query,
            "provided_ticker": clean_ticker or None,
        },
        "intent": None,
        "tickers": [clean_ticker] if clean_ticker else [],
        "time_horizon": "unknown",
        "plan": {
            "required_evidence": [],
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        },
        "evidence": {
            "market": {},
            "news": {},
            "fundamentals": {},
            "profile": {},
            "sentiment": {},
            "charts": [],
        },
        "gaps": [],
        "critic_result": {},
        "decision": None,
        "confidence": None,
        "needs_human": False,
        "clarification_question": None,
        "guardrails": {
            "out_of_scope": False,
            "message": None,
            "scope_note": None,
        },
        "metadata": {
            "ticker_source": "provided" if clean_ticker else "unknown",
            "ticker_inference_confidence": None,
            "planner_mode": None,
            "planner_reasoning_brief": None,
            "research_mode": None,
            "research_reasoning_brief": None,
            "research_plan_steps": [],
            "executed_research_steps": [],
            "decision_mode": None,
            "decision_reasoning_brief": None,
        },
    }


def clone_state(state: dict) -> dict:
    """Return a deep copy so agents can update state safely in tests."""
    return deepcopy(state)
