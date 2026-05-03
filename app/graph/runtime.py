from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.graph.tradepilot_graph import build_tradepilot_graph
from app.state import build_initial_state


_RUNTIME_CACHE: dict[int, dict[str, Any]] = {}


def _import_langgraph_runtime():
    from langgraph.types import Command

    return Command


def _skills_cache_key(skills) -> int:
    return id(skills)


def _runtime_for_skills(skills):
    key = _skills_cache_key(skills)
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        app = build_tradepilot_graph(skills=skills)
        runtime = {"app": app}
        _RUNTIME_CACHE[key] = runtime
    return runtime


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _snapshot_state(app, thread_id: str) -> dict:
    snapshot = app.get_state(_thread_config(thread_id))
    values = getattr(snapshot, "values", {}) or {}
    return deepcopy(values)


def _interrupt_payload(result: Any) -> Optional[dict]:
    if not isinstance(result, dict):
        return None

    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    value = getattr(first_interrupt, "value", None)
    return value if isinstance(value, dict) else {"message": value}


def _mark_waiting_for_user(state: dict) -> dict:
    next_state = deepcopy(state)
    metadata = next_state.setdefault("metadata", {})
    metadata["stopped_reason"] = "human_clarification_required"
    metadata.setdefault("iterations_used", 0)
    return next_state


def start_tradepilot_graph_run(
    query: str,
    ticker: Optional[str] = None,
    skills=None,
    max_iterations: Optional[int] = None,
) -> Tuple[str, dict, Optional[dict], str]:
    runtime = _runtime_for_skills(skills)
    app = runtime["app"]
    thread_id = uuid4().hex

    initial_state = build_initial_state(query=query, ticker=ticker)
    metadata = initial_state.setdefault("metadata", {})
    if max_iterations is not None:
        metadata["requested_max_iterations"] = max_iterations

    result = app.invoke(initial_state, config=_thread_config(thread_id))
    interrupt = _interrupt_payload(result)
    if interrupt is not None:
        state = _mark_waiting_for_user(_snapshot_state(app, thread_id))
        return thread_id, state, interrupt, "waiting_for_user"

    return thread_id, deepcopy(result), None, result.get("metadata", {}).get("stopped_reason", "completed")


def resume_tradepilot_graph_run(
    thread_id: str,
    clarification_value: str,
    skills=None,
) -> Tuple[str, dict, Optional[dict], str]:
    runtime = _runtime_for_skills(skills)
    app = runtime["app"]
    Command = _import_langgraph_runtime()

    result = app.invoke(
        Command(resume=clarification_value),
        config=_thread_config(thread_id),
    )
    interrupt = _interrupt_payload(result)
    if interrupt is not None:
        state = _mark_waiting_for_user(_snapshot_state(app, thread_id))
        return thread_id, state, interrupt, "waiting_for_user"

    return thread_id, deepcopy(result), None, result.get("metadata", {}).get("stopped_reason", "completed")


__all__ = [
    "resume_tradepilot_graph_run",
    "start_tradepilot_graph_run",
]
