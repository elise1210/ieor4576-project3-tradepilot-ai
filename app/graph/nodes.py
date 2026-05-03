from typing import Callable, Dict, Optional

from app.agents.critic_agent import run_critic_agent
from app.agents.decision_agent import run_decision_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.research_agent import run_research_agent


SkillRegistry = Dict[str, Callable]
DECISION_INTENTS = {"buy_sell_decision", "comparison"}


def planner_node(state: dict) -> dict:
    next_state = run_planner_agent(state)
    next_state.setdefault("metadata", {}).setdefault("iterations_used", 0)
    return next_state


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
    "critic_node",
    "critic_route",
    "decision_node",
    "exhausted_stop_node",
    "human_clarification_stop_node",
    "out_of_scope_stop_node",
    "planner_node",
    "planner_route",
    "research_complete_node",
    "research_node",
]
