from typing import Callable, Dict, Optional

from app.agents.critic_agent import run_critic_agent
from app.agents.decision_agent import run_decision_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.research_agent import run_research_agent
from app.state import build_initial_state


SkillRegistry = Dict[str, Callable]


def run_tradepilot_pipeline(
    query: str,
    ticker: Optional[str] = None,
    skills: Optional[SkillRegistry] = None,
    max_iterations: Optional[int] = None,
) -> dict:
    """
    Reusable deterministic orchestrator for the 4-agent pipeline.

    Flow:
    1. Build initial state
    2. Planner
    3. If clarification needed, stop
    4. Research -> Critic loop
    5. If enough evidence, Decision
    6. Otherwise stop when iteration budget is exhausted
    """
    state = build_initial_state(query=query, ticker=ticker)
    state = run_planner_agent(state)

    if state.get("guardrails", {}).get("out_of_scope"):
        state["metadata"]["stopped_reason"] = "out_of_scope"
        state["metadata"]["iterations_used"] = 0
        return state

    if state.get("needs_human"):
        state["metadata"]["stopped_reason"] = "human_clarification_required"
        state["metadata"]["iterations_used"] = 0
        return state

    iteration_budget = max_iterations or state.get("plan", {}).get("max_iterations", 2)
    state["metadata"]["iterations_used"] = 0

    for iteration in range(iteration_budget):
        state = run_research_agent(state, skills=skills)
        state = run_critic_agent(state)
        state["metadata"]["iterations_used"] = iteration + 1

        if state.get("critic_result", {}).get("enough_evidence"):
            state = run_decision_agent(state)
            state["metadata"]["stopped_reason"] = "decision_completed"
            return state

    state["metadata"]["stopped_reason"] = "iteration_budget_exhausted"
    return state
