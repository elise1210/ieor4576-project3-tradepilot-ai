# TradePilot AI

IEOR 4576 Project 3

TradePilot AI is a financial decision support system that helps a user analyze a stock, compare companies, and understand the evidence behind a recommendation. The system is designed around a multi-agent workflow, tool calling, shared state, and human-in-the-loop checkpoints instead of a single monolithic prompt.

## Project Goal

Users often have to jump across multiple sources to answer simple investing questions:

- What happened to this stock recently?
- Is the recent news positive or negative?
- Is the company fundamentally strong?
- Should I buy, hold, or wait?

TradePilot AI combines market data, company news, fundamentals, sentiment analysis, and structured reasoning into one decision-oriented workflow.

This system is for informational and educational use only. It is not financial advice.

## Core Design

The system uses 4 reasoning agents and a shared run state.

```text
User Query
  ->
Planner / Supervisor Agent
  ->
Research / Tool-Using Analyst Agent
  ->
Critic / Verifier Agent
  ->
Decision / Response Agent
```

The workflow is iterative rather than strictly one-pass:

```text
Planner -> Research -> Critic
                    -> if gaps remain: Research again
                    -> if evidence is sufficient: Decision
                    -> if ambiguity or risk is high: Human-in-the-Loop
```

This design lets the system gather evidence, inspect whether that evidence is enough, and ask for clarification when confidence is too low.

## Why This Architecture

The original project structure treated agents almost like function wrappers such as news, market, and company agents. In this version, those become reusable skills or tools, while agents handle reasoning stages.

- Agents decide what to do
- Skills perform reusable work
- Shared state keeps the workflow consistent

This makes the system easier to extend, debug, and explain in a course setting.

## Agent Responsibilities

### 1. Planner / Supervisor Agent

Responsibilities:

- understand user intent
- extract company names and tickers
- identify time horizon such as short-term vs long-term
- decide what categories of evidence are needed
- define stop criteria for "enough evidence"
- trigger human clarification when the query is ambiguous

Example outputs:

- single-stock recommendation
- company comparison
- explanatory company analysis
- deeper research request

### 2. Research / Tool-Using Analyst Agent

Responsibilities:

- inspect the current run state
- reason about missing evidence
- choose which tool or skill to call next
- fetch and normalize market, news, and company data
- compute technical signals
- request chart generation when useful

This is not just a fetcher. It is allowed to think about what evidence is still missing and what action best fills the gap.

### 3. Critic / Verifier Agent

Responsibilities:

- check whether evidence is sufficient
- check whether news and price data are fresh enough
- check whether comparisons are fair across tickers
- detect conflicting signals
- send structured follow-up requests back to the research agent
- trigger human review when confidence is too low

### 4. Decision / Response Agent

Responsibilities:

- synthesize the evidence
- produce a BUY / HOLD / SELL style output, or a comparison result
- explain the major drivers clearly
- calibrate uncertainty
- add financial-risk and compliance language

## Human-in-the-Loop

Human-in-the-loop is a first-class part of the system.

The system should pause and ask the user when:

- the ticker is ambiguous
- the user goal is unclear
- evidence is sparse or conflicting
- confidence is too low for a strong recommendation
- the system needs the user to choose between short-term and long-term analysis

This prevents the system from pretending certainty when it should ask for guidance.

## Skills and Tools

The system separates reasoning agents from reusable skills.

### Planner Skills

- `intent_classification`: figure out what kind of question the user is asking
- `entity_extraction`: pull out the company names, tickers, and time horizon
- `task_planning`: decide what information the system needs before answering
- `routing`: choose the right workflow, such as single-stock analysis or comparison
- `human_gate_check`: decide whether the system should pause and ask the user to clarify something

### Research Skills

- `gap_reasoning`: look at the current evidence and decide what is still missing
- `tool_selection`: choose the best tool or skill to call next
- `market_data_skill`: fetch stock prices, returns, and trading history
- `news_skill`: fetch recent company news
- `fundamentals_skill`: fetch company profile details and basic financial numbers
- `sentiment_skill`: score whether the recent news is positive, neutral, or negative
- `technical_signal_skill`: turn raw price data into simple signals like trend, volatility, and RSI
- `chart_skill`: turn structured market data into a chart specification for the frontend
- `comparison_alignment`: make sure both companies are judged using the same kinds of evidence

### Critic Skills

- `evidence_sufficiency_check`: decide whether there is enough information to answer well
- `freshness_check`: make sure the data is recent enough to trust
- `fairness_check`: make sure a comparison is balanced across both companies
- `conflict_detection`: spot signals that disagree with each other
- `followup_generation`: list the exact extra information the research agent should collect
- `confidence_scoring`: estimate how confident the system should be in its answer

### Decision Skills

- `recommendation_reasoning`: turn the evidence into a final recommendation or comparison
- `uncertainty_calibration`: avoid sounding more certain than the evidence allows
- `compliance_skill`: add careful wording so the answer does not sound like reckless financial advice
- `response_generation`: write the final answer in a clear and helpful way
- `evidence_citation`: point to the key facts that support the recommendation

## Data Stack

The MVP uses lightweight, high-value financial data sources.

### 1. Market Data: `yfinance`

Used for:

- price history
- daily returns
- volatility
- moving averages
- RSI
- drawdown
- chart-ready series

### 2. News and Fundamentals: `Finnhub`

Used for:

- recent company news
- quote data
- company profile
- basic financial metrics
- 52-week range
- market capitalization
- valuation context

### 3. Sentiment: `FinBERT`

Used for:

- headline and summary sentiment
- aggregate positive / neutral / negative signal
- sentiment score for decision support

### 4. Derived Internal Signals

Used for:

- trend classification
- short-term risk label
- comparison summaries
- evidence confidence estimation

## Shared State

All agents operate over a shared `RunState` instead of passing only free-form text.

Example structure:

```json
{
  "query": "Should I buy AAPL this week?",
  "intent": "buy_sell_decision",
  "tickers": ["AAPL"],
  "time_horizon": "short_term",
  "plan": {
    "required_evidence": ["news", "market", "fundamentals", "sentiment"],
    "max_iterations": 2
  },
  "evidence": {
    "market": {},
    "news": [],
    "fundamentals": {},
    "profile": {},
    "sentiment": {},
    "charts": []
  },
  "gaps": [],
  "critic_result": {},
  "decision": null,
  "confidence": null,
  "needs_human": false
}
```

This shared state supports better orchestration, debugging, and evaluation.

## Evidence Loop

The system is built around a bounded research loop.

```text
1. Planner creates an initial task plan
2. Research gathers and normalizes evidence
3. Critic checks sufficiency and fairness
4. If gaps remain, Research fetches additional evidence
5. Decision produces the final response
6. Human-in-the-loop interrupts when needed
```

To keep the workflow safe and efficient, the loop should include:

- max iteration limits
- tool budgets
- freshness checks
- explicit stop conditions

## Visualization

Charts are treated as a reusable skill, not arbitrary code generation.

The preferred pattern is:

- research agent decides whether a chart is useful
- chart skill produces a structured chart specification
- frontend renders the chart

This is more reliable than having an agent write plotting code from scratch during each run.

## Optional Future Extension: RAG

RAG is not required for the MVP recommendation loop, but it can be added later for long-horizon company context such as:

- company milestones
- major M&A history
- strategic background
- product launches
- earnings-history themes

For the default recommendation flow, live APIs and derived signals are enough.

## Course Concepts Demonstrated

This project is intentionally aligned with the main themes of IEOR 4576.

### Multi-Agent Patterns

- planner / supervisor
- research agent
- critic / verifier
- decision agent
- human-in-the-loop checkpointing

### Tool Calling

- external market and news APIs
- sentiment model invocation
- chart generation as a structured tool

### Context Engineering

- shared run state
- selective evidence passing
- role-specific prompts
- normalized evidence instead of raw tool dumps

### State, Context, and Memory

- persistent run state across agent steps
- explicit tracking of evidence, gaps, and confidence

### Evaluation

- critic-based evidence checks
- confidence scoring
- later support for rubric-based evaluation or LLM-as-judge

### Agents as Functions

Each agent can be viewed as a state transformation:

- `plan(state) -> state`
- `research(state) -> state`
- `critic(state) -> state`
- `decide(state) -> state`

### RAG

- optional future layer for long-term company history
- not required for core real-time recommendation flow

## Proposed Repository Shape

```text
tradepilot-ai/
|-- app/
|   |-- main.py
|   |-- state.py
|   |-- config.py
|   |-- agents/
|   |   |-- planner_agent.py
|   |   |-- research_agent.py
|   |   |-- critic_agent.py
|   |   `-- decision_agent.py
|   |-- skills/
|   |   |-- market.py
|   |   |-- news.py
|   |   |-- fundamentals.py
|   |   |-- sentiment.py
|   |   |-- chart.py
|   |   `-- compliance.py
|   `-- prompts/
|-- frontend/
|   `-- index.html
|-- README.md
|-- Dockerfile
`-- cloudbuild.yaml
```

## MVP Scope

The first working version should support:

- single-stock analysis
- BUY / HOLD / SELL style recommendation
- two-company comparison
- evidence-backed reasoning
- human clarification when needed
- chart generation from structured market data

The MVP should not try to do everything at once. In particular, it should avoid:

- large-scale patent retrieval
- complex long-term knowledge bases
- too many specialized agents
- unbounded autonomous loops

## Next Build Priorities

1. define `RunState`
2. implement market, news, fundamentals, and sentiment skills
3. implement the planner agent
4. implement the research agent
5. implement the critic agent
6. implement the decision agent
7. add human-in-the-loop checkpoints
8. add chart rendering
9. add optional RAG only if time remains

## Security Note

API keys should never be stored directly in notebooks, prompts, or committed files. All credentials should be loaded from environment variables or a secure secret manager.

## Disclaimer

TradePilot AI provides educational and informational analysis only. It does not provide personalized investment advice, guarantee future performance, or replace professional financial judgment.
