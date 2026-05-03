import re
from typing import Callable, Dict, Optional

from app.agents.critic_agent import run_critic_agent
from app.agents.decision_agent import run_decision_agent
from app.agents.planner_agent import infer_tickers, infer_time_horizon, run_planner_agent
from app.agents.research_agent import run_research_agent
from app.state import clone_state


SkillRegistry = Dict[str, Callable]
DECISION_INTENTS = {"buy_sell_decision", "comparison"}


def planner_node(state: dict) -> dict:
    next_state = run_planner_agent(state)
    next_state.setdefault("metadata", {}).setdefault("iterations_used", 0)
    return next_state


def _clear_clarification_state(next_state: dict) -> dict:
    next_state["needs_human"] = False
    next_state["clarification_question"] = None
    next_state["clarification_type"] = None
    next_state["clarification_options"] = []
    return next_state


def _normalize_ticker_reply(reply: str) -> str:
    cleaned = (reply or "").strip().upper()
    return cleaned if re.fullmatch(r"[A-Z]{1,5}", cleaned) else ""


def _resolve_ticker_from_reply(reply: str) -> tuple[list[str], Optional[str]]:
    tickers, _, confidence = infer_tickers(reply)
    if tickers:
        return tickers, confidence

    explicit_ticker = _normalize_ticker_reply(reply)
    if explicit_ticker:
        return [explicit_ticker], "high"

    return [], None


def _resolve_time_horizon_from_reply(reply: str) -> str:
    normalized = (reply or "").strip().lower().replace("-", " ")
    if normalized in {"short term", "short_term", "shortterm"}:
        return "short_term"
    if normalized in {"long term", "long_term", "longterm"}:
        return "long_term"
    return infer_time_horizon(reply)


def apply_clarification_to_state(state: dict, clarification_value: str) -> dict:
    next_state = clone_state(state)
    clarification_type = next_state.get("clarification_type")
    reply = (clarification_value or "").strip()
    metadata = next_state.setdefault("metadata", {})
    metadata["clarification_answer"] = reply
    metadata["clarification_resolved_via"] = "langgraph_interrupt"

    if clarification_type == "time_horizon":
        resolved_horizon = _resolve_time_horizon_from_reply(reply)
        if resolved_horizon == "unknown":
            next_state["needs_human"] = True
            next_state["clarification_question"] = (
                "I still need your time horizon. Do you want a short-term trading view "
                "or a longer-term investment view?"
            )
            next_state["clarification_type"] = "time_horizon"
            next_state["clarification_options"] = [
                {"label": "Short-term", "value": "short_term"},
                {"label": "Long-term", "value": "long_term"},
            ]
            metadata["clarification_error"] = "unresolved_time_horizon"
            return next_state

        next_state["time_horizon"] = resolved_horizon
        metadata["clarification_error"] = None
        return _clear_clarification_state(next_state)

    if clarification_type == "ticker":
        tickers, confidence = _resolve_ticker_from_reply(reply)
        if not tickers:
            next_state["needs_human"] = True
            next_state["clarification_question"] = (
                "I still couldn't determine the company or ticker. Please reply with a "
                "stock ticker like AAPL or a company name like Apple."
            )
            next_state["clarification_type"] = "ticker"
            next_state["clarification_options"] = []
            metadata["clarification_error"] = "unresolved_ticker"
            return next_state

        next_state["tickers"] = tickers
        metadata["ticker_source"] = "clarification"
        metadata["ticker_inference_confidence"] = confidence or "high"
        metadata["clarification_error"] = None
        return _clear_clarification_state(next_state)

    metadata["clarification_error"] = None
    return _clear_clarification_state(next_state)


def build_interrupt_payload(state: dict) -> dict:
    return {
        "clarification_question": state.get("clarification_question"),
        "clarification_type": state.get("clarification_type"),
        "clarification_options": state.get("clarification_options", []),
        "original_query": state.get("query"),
        "tickers": state.get("tickers", []),
    }


def make_clarification_node(interrupt_fn):
    def clarification_node(state: dict) -> dict:
        resume_value = interrupt_fn(build_interrupt_payload(state))
        return apply_clarification_to_state(state, str(resume_value or "").strip())

    return clarification_node


def research_node(state: dict, skills: Optional[SkillRegistry] = None) -> dict:
    return run_research_agent(state, skills=skills)


def critic_node(state: dict) -> dict:
    next_state = run_critic_agent(state)
    metadata = next_state.setdefault("metadata", {})
    metadata["iterations_used"] = int(metadata.get("iterations_used", 0) or 0) + 1
    return next_state


def decision_node(state: dict) -> dict:
    next_state = run_decision_agent(state)
    next_state.setdefault("metadata", {})["stopped_reason"] = "decision_completed"
    return next_state


def out_of_scope_stop_node(state: dict) -> dict:
    next_state = dict(state)
    metadata = dict(next_state.get("metadata", {}))
    metadata["stopped_reason"] = "out_of_scope"
    metadata["iterations_used"] = 0
    next_state["metadata"] = metadata
    return next_state


def human_clarification_stop_node(state: dict) -> dict:
    next_state = dict(state)
    metadata = dict(next_state.get("metadata", {}))
    metadata["stopped_reason"] = "human_clarification_required"
    metadata["iterations_used"] = 0
    next_state["metadata"] = metadata
    return next_state


def research_complete_node(state: dict) -> dict:
    next_state = dict(state)
    next_state.setdefault("metadata", {})["stopped_reason"] = "research_completed"
    return next_state


def exhausted_stop_node(state: dict) -> dict:
    next_state = dict(state)
    next_state.setdefault("metadata", {})["stopped_reason"] = "iteration_budget_exhausted"
    return next_state


def planner_route(state: dict) -> str:
    if state.get("guardrails", {}).get("out_of_scope"):
        return "out_of_scope"
    if state.get("needs_human"):
        return "needs_human"
    return "research"


def clarification_route(state: dict) -> str:
    if state.get("needs_human"):
        return "needs_human"
    return "research"


def _iteration_budget(state: dict) -> int:
    metadata = state.get("metadata", {})
    override = metadata.get("requested_max_iterations")
    if isinstance(override, int) and override > 0:
        return override

    plan_budget = state.get("plan", {}).get("max_iterations")
    if isinstance(plan_budget, int) and plan_budget > 0:
        return plan_budget

    return 2


def critic_route(state: dict) -> str:
    enough_evidence = bool(state.get("critic_result", {}).get("enough_evidence"))
    if enough_evidence:
        if state.get("intent") in DECISION_INTENTS:
            return "decision"
        return "research_completed"

    iterations_used = int(state.get("metadata", {}).get("iterations_used", 0) or 0)
    if iterations_used >= _iteration_budget(state):
        return "exhausted"
    return "research"


__all__ = [
    "SkillRegistry",
    "apply_clarification_to_state",
    "build_interrupt_payload",
    "clarification_route",
    "critic_node",
    "critic_route",
    "decision_node",
    "exhausted_stop_node",
    "human_clarification_stop_node",
    "make_clarification_node",
    "out_of_scope_stop_node",
    "planner_node",
    "planner_route",
    "research_complete_node",
    "research_node",
]
