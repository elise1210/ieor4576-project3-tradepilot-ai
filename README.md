# TradePilot AI

IEOR 4576 Project 3

TradePilot AI is a multi-agent financial decision support system. It helps a user analyze a stock, compare companies, and understand the evidence behind a recommendation such as `BUY`, `HOLD`, or `SELL`.

The key design idea is simple:

- `agents` decide what to do
- `skill modules` handle content-related work such as fetching, analysis, and formatting
- `sub-skills` handle the supporting assistant work each agent uses to do its job

This README uses that structure consistently.

## Project Goal

Users often have to switch between multiple websites and tools to answer simple questions:

- What happened to this stock recently?
- Is the news positive or negative?
- Is the company fundamentally strong?
- Should I buy now, hold, or wait?

TradePilot AI combines market data, company news, fundamentals, sentiment analysis, and structured reasoning into one workflow.

This system is for educational and informational use only. It is not financial advice.

## System Overview

The system uses 4 reasoning agents:

```text
User Query
  ->
Planner / Supervisor Agent
  ->
Research / Evidence-Building Agent
  ->
Critic / Verifier Agent
  ->
Decision / Response Agent
```

The workflow can loop when the evidence is not strong enough:

```text
Planner -> Research -> Critic
                    -> if gaps remain: Research again
                    -> if enough evidence: Decision
                    -> if ambiguity is high: Human-in-the-Loop
```

## Why This Design

The project originally treated domain pieces like news, market, and company context as if they were top-level agents. In this design, they are handled as reusable skill modules instead.

This gives us a cleaner separation:

- agents handle reasoning stages
- skill modules handle fetching, analysis, or formatting work
- sub-skills explain each agent's concrete abilities

## Current External Data Sources

Right now the system uses:

- `yfinance`
  - raw stock price and volume history
- `Finnhub`
  - company news, quote data, company profile, and basic fundamentals

It also uses:

- `FinBERT`
  - sentiment analysis model for financial news

So:

- external data providers: `2`
- outside analysis model: `1`

## Skill Modules

These are the main reusable modules in the system.

### Fetching Skill Modules

- `yfinance_tool`
  - fetches raw stock history such as date, open, high, low, close, and volume
  - this is mainly a fetcher

- `finnhub_tool`
  - fetches raw news, quote, company profile, and basic financial data from Finnhub
  - this is mainly a fetcher

### Analysis Skill Modules

- `market`
  - turns raw price history into signals such as trend, volatility, moving-average context, and later RSI or drawdown
  - this is an analysis skill

- `news`
  - filters noisy articles, keeps company-relevant news, ranks price-relevant items, and summarizes the coverage
  - this is an analysis skill

- `fundamentals`
  - turns raw company profile and financial fields into usable company context
  - this is an analysis skill

- `sentiment`
  - runs FinBERT on recent news and aggregates the overall sentiment
  - this is an analysis skill

### Presentation / Safety Skill Modules

- `chart`
  - turns structured evidence into chart-ready output for the frontend
  - this is a presentation skill

- `compliance`
  - adds safer wording, uncertainty language, and disclaimer logic
  - this is a response-safety skill

## Agent Responsibilities and Skills

This is the most important section. Each agent has its own role, its own sub-skills, and its own skill modules.

### 1. Planner / Supervisor Agent

Main job:

- understand the question
- identify the ticker or company
- decide what evidence is needed
- decide whether the system should ask the user for clarification

Sub-skills owned by the Planner:

- `intent_classification`
  - figure out what kind of question the user is asking
- `entity_extraction`
  - pull out ticker, company name, and time horizon
- `task_planning`
  - decide what evidence categories are needed before answering
- `routing`
  - choose the right workflow such as single-stock analysis, comparison, or deeper research
- `human_gate_check`
  - decide whether the system should pause and ask the user a clarifying question

Skill modules used by the Planner:

- none of the data skills directly by default
- the Planner mainly works on the query and the shared state

Planner output:

- intent
- ticker(s)
- required evidence
- stop criteria
- whether human clarification is needed

### 2. Research / Evidence-Building Agent

Main job:

- gather the evidence
- analyze the evidence
- organize the evidence into shared state

Important note:

The Research Agent is not just a fetcher. It does both:

- fetching raw data
- building usable evidence from that data

But it does not make the final recommendation.

Sub-skills owned by the Research Agent:

- `gap_reasoning`
  - look at the current state and decide what evidence is still missing
- `tool_selection`
  - choose which skill module to call next
- `comparison_alignment`
  - make sure both companies are evaluated using the same kinds of evidence
- `technical_signal_skill`
  - turn price history into simple market signals
- `news_filtering`
  - keep only relevant and useful company news
- `sentiment_aggregation`
  - turn article-level sentiment into one overall signal
- `evidence_normalization`
  - store all collected results in a clean shared structure
- `chart_requesting`
  - ask for a chart when a graph would improve the answer

Skill modules used by the Research Agent:

- `yfinance_tool`
  - fetch raw stock price history
- `finnhub_tool`
  - fetch raw news, quote, profile, and fundamentals
- `market`
  - analyze market behavior
- `news`
  - filter and summarize company news
- `fundamentals`
  - structure company background and financial context
- `sentiment`
  - score recent news sentiment
- `chart`
  - optionally produce graph-ready output

Research output:

- market evidence
- news evidence
- company/fundamental evidence
- sentiment evidence
- chart spec when useful
- updated gaps if more research is needed

### 3. Critic / Verifier Agent

Main job:

- check whether the collected evidence is strong enough, current enough, and fair enough

The Critic usually does not fetch data itself. It inspects what the Research Agent already produced.

Sub-skills owned by the Critic:

- `evidence_sufficiency_check`
  - decide whether there is enough information to answer well
- `freshness_check`
  - make sure the data is recent enough to trust
- `fairness_check`
  - make sure comparisons are balanced across both companies
- `conflict_detection`
  - spot signals that disagree with each other
- `followup_generation`
  - list the exact extra evidence the Research Agent should collect
- `confidence_scoring`
  - estimate how confident the system should be

Skill modules used by the Critic:

- no direct fetching modules by default
- the Critic mainly reads shared evidence produced by the Research Agent

Critic output:

- enough evidence or not
- missing evidence list
- follow-up tasks
- confidence estimate
- human-review trigger when needed

### 4. Decision / Response Agent

Main job:

- turn evidence into the final answer
- explain the answer clearly
- avoid overconfident wording

The Decision Agent should not be responsible for collecting evidence. It should work from the structured results already prepared.

Sub-skills owned by the Decision Agent:

- `recommendation_reasoning`
  - turn the evidence into a final recommendation or comparison
- `uncertainty_calibration`
  - avoid sounding more certain than the evidence allows
- `response_generation`
  - write the final answer in a clear and useful way
- `evidence_citation`
  - point to the key facts that support the conclusion
- `risk_language_control`
  - keep the tone careful when the evidence is weak or mixed

Skill modules used by the Decision Agent:

- `compliance`
  - add disclaimer logic and safer wording
- `chart`
  - optionally include chart-based explanation if already prepared

Decision output:

- final recommendation or comparison
- reasoning
- confidence level
- uncertainty language
- disclaimer

## Human-in-the-Loop

Human-in-the-loop is part of the system, not an afterthought.

The system should pause and ask the user when:

- the ticker is ambiguous
- the user goal is unclear
- the user did not specify short-term vs long-term intent
- the evidence is thin or conflicting
- confidence is too low for a strong answer

## Shared State

All agents work over a shared `RunState` rather than only passing free-form text.

Example:

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

This shared state helps with:

- context management
- debugging
- multi-step orchestration
- evaluation

## Evidence Flow

The evidence flow is intentionally bounded.

```text
1. Planner decides what evidence is needed
2. Research fetches and analyzes evidence
3. Critic checks if the evidence is sufficient
4. If needed, Research runs again for missing pieces
5. Decision produces the final answer
6. Human-in-the-loop interrupts when needed
```

To keep the system safe and efficient, we should include:

- max iteration limits
- tool budgets
- freshness rules
- explicit stop conditions

## Proposed Repository Structure

This is the consistent structure we want to build toward:

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
|   |   |-- compliance.py
|   |   |-- yfinance_tool.py
|   |   `-- finnhub_tool.py
|   `-- prompts/
|-- frontend/
|   `-- index.html
|-- README.md
|-- Dockerfile
`-- cloudbuild.yaml
```

## Current Build Status

The current work in the repo mostly covers the research-side functionality first.

Implemented or partly implemented:

- `news`
- `market`
- `sentiment`
- `finnhub_tool`
- `decision_agent`

Still mostly placeholders:

- `planner_agent`
- `research_agent`
- `critic_agent`
- `fundamentals`
- `chart`
- `compliance`
- `yfinance_tool`
- `state`

This means the project already has important evidence-processing pieces, but the full orchestration layer is not finished yet.

## Course Concepts Demonstrated

This project is designed to connect clearly to IEOR 4576 themes.

### Multi-Agent Patterns

- planner / supervisor
- research / evidence-building
- critic / verifier
- decision / response
- human-in-the-loop

### Tool Calling

- fetch live stock and company data
- run a sentiment model
- generate chart-ready outputs

### Context Engineering

- shared run state
- role-specific prompts
- normalized evidence instead of raw dumps

### State, Context, and Memory

- explicit tracking of evidence, gaps, confidence, and human checkpoints

### Evaluation

- evidence sufficiency checks
- fairness checks
- confidence scoring

### Agents as Functions

Each agent can be viewed as a state transformation:

- `plan(state) -> state`
- `research(state) -> state`
- `critic(state) -> state`
- `decide(state) -> state`

### RAG

- optional future extension for long-term company history, milestones, or M&A context
- not required for the core MVP loop

## MVP Scope

The first working version should support:

- single-stock analysis
- evidence-backed `BUY / HOLD / SELL` style output
- two-company comparison
- human clarification when needed
- optional simple chart output

The MVP should avoid doing too much at once. In particular, it should not start with:

- patent search
- heavy long-term knowledge retrieval
- too many specialized agents
- unbounded autonomous loops

## Next Build Priorities

1. finish `state.py`
2. fix imports so the current skill modules are wired correctly
3. finish `yfinance_tool.py`
4. finish `fundamentals.py`
5. implement `research_agent.py`
6. implement `planner_agent.py`
7. implement `critic_agent.py`
8. add `compliance.py`
9. add `chart.py`

## Security Note

API keys should never be stored directly in notebooks, prompts, or committed files. They should be loaded from environment variables or a secure secret manager.

## Disclaimer

TradePilot AI provides educational and informational analysis only. It does not provide personalized investment advice, guarantee future performance, or replace professional financial judgment.
