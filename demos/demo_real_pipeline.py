import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.agents.critic_agent import run_critic_agent
from app.agents.decision_agent import run_decision_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.research_agent import run_research_agent
from app.orchestrator import run_tradepilot_pipeline
from app.skills.registry import REAL_SKILLS
from app.state import build_initial_state


def print_section(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def summarize_state(state: dict) -> dict:
    evidence = state.get("evidence", {})
    return {
        "query": state.get("query"),
        "intent": state.get("intent"),
        "tickers": state.get("tickers"),
        "time_horizon": state.get("time_horizon"),
        "plan": state.get("plan"),
        "gaps": state.get("gaps"),
        "guardrails": state.get("guardrails"),
        "needs_human": state.get("needs_human"),
        "clarification_question": state.get("clarification_question"),
        "critic_result": state.get("critic_result"),
        "confidence": state.get("confidence"),
        "evidence_keys": {
            "news": list(evidence.get("news", {}).keys()),
            "market": list(evidence.get("market", {}).keys()),
            "fundamentals": list(evidence.get("fundamentals", {}).keys()),
            "sentiment": list(evidence.get("sentiment", {}).keys()),
            "charts": len(evidence.get("charts", [])),
        },
        "decision": state.get("decision"),
        "metadata": state.get("metadata"),
    }


def run_step_by_step(query: str, ticker: str = None) -> None:
    state = build_initial_state(query=query, ticker=ticker)
    print_section("Initial State", summarize_state(state))

    state = run_planner_agent(state)
    print_section("Planner Output", summarize_state(state))

    if state.get("guardrails", {}).get("out_of_scope") or state.get("needs_human"):
        return

    iteration_budget = state.get("plan", {}).get("max_iterations", 2)

    for iteration in range(iteration_budget):
        state = run_research_agent(state, skills=REAL_SKILLS)
        print_section(f"Research Output Pass {iteration + 1}", summarize_state(state))

        state = run_critic_agent(state)
        print_section(f"Critic Output Pass {iteration + 1}", summarize_state(state))

        if state.get("critic_result", {}).get("enough_evidence"):
            state = run_decision_agent(state)
            print_section("Decision Output", summarize_state(state))
            return

    print_section("Stopped Before Decision", summarize_state(state))


def run_full_pipeline(query: str, ticker: str = None) -> None:
    result = run_tradepilot_pipeline(query=query, ticker=ticker, skills=REAL_SKILLS)
    print_section("Full Pipeline Result", summarize_state(result))
    print_section("Full Pipeline Raw State", result)


def main() -> None:
    query = "Should I buy Apple this week?"

    print_section("Using REAL_SKILLS", sorted(REAL_SKILLS.keys()))
    run_step_by_step(query)
    run_full_pipeline(query)


if __name__ == "__main__":
    main()
