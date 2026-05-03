from typing import Callable, Dict, Optional

from app.graph.nodes import (
    SkillRegistry,
    critic_node,
    critic_route,
    decision_node,
    exhausted_stop_node,
    human_clarification_stop_node,
    out_of_scope_stop_node,
    planner_node,
    planner_route,
    research_complete_node,
    research_node,
)
from app.graph.state_schema import TradePilotGraphState


def _import_langgraph():
    from langgraph.graph import END, START, StateGraph

    return StateGraph, START, END


def build_tradepilot_graph(skills: Optional[SkillRegistry] = None):
    StateGraph, START, END = _import_langgraph()

    graph = StateGraph(TradePilotGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("research", lambda state: research_node(state, skills=skills))
    graph.add_node("critic", critic_node)
    graph.add_node("decision", decision_node)
    graph.add_node("out_of_scope_stop", out_of_scope_stop_node)
    graph.add_node("human_clarification_stop", human_clarification_stop_node)
    graph.add_node("research_complete", research_complete_node)
    graph.add_node("exhausted_stop", exhausted_stop_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        planner_route,
        {
            "out_of_scope": "out_of_scope_stop",
            "needs_human": "human_clarification_stop",
            "research": "research",
        },
    )
    graph.add_edge("research", "critic")
    graph.add_conditional_edges(
        "critic",
        critic_route,
        {
            "decision": "decision",
            "research_completed": "research_complete",
            "exhausted": "exhausted_stop",
            "research": "research",
        },
    )
    graph.add_edge("decision", END)
    graph.add_edge("out_of_scope_stop", END)
    graph.add_edge("human_clarification_stop", END)
    graph.add_edge("research_complete", END)
    graph.add_edge("exhausted_stop", END)

    return graph.compile()


def run_tradepilot_graph(state: dict, skills: Optional[SkillRegistry] = None) -> dict:
    app = build_tradepilot_graph(skills=skills)
    return app.invoke(state)


__all__ = ["build_tradepilot_graph", "run_tradepilot_graph"]
