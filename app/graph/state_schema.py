from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class TradePilotGraphState(TypedDict, total=False):
    query: str
    user_inputs: Dict[str, Any]
    intent: Optional[str]
    tickers: List[str]
    time_horizon: str
    plan: Dict[str, Any]
    evidence: Dict[str, Any]
    gaps: List[str]
    critic_result: Dict[str, Any]
    decision: Optional[Dict[str, Any]]
    confidence: Optional[str]
    needs_human: bool
    clarification_question: Optional[str]
    clarification_type: Optional[str]
    clarification_options: List[Dict[str, Any]]
    guardrails: Dict[str, Any]
    metadata: Dict[str, Any]


__all__ = ["TradePilotGraphState"]
