# Agentic AI Learning Journey

A hands-on, incremental learning repository for agentic AI frameworks — built example by example, concept by concept.

## Frameworks

| Folder | Framework | Examples | Status |
|--------|-----------|----------|--------|
| [lang-graph/](lang-graph/) | LangGraph | 12 examples — from basics to production agent | ✅ Complete |
| [crew-ai/](crew-ai/) | CrewAI | Role-based multi-agent framework | ✅ Complete |

## What's Inside

Each framework folder contains:
- Numbered Python files (`01_`, `02_`, ...) — each teaching one concept
- Its own `README.md` with a full example table
- Its own `requirements.txt` and `.env.example`
- Its own Python virtual environment (`venv/`) — not committed to git

## Getting Started

```bash
# Clone the repo
git clone https://github.com/vasumai/agentic-learnings.git
cd agentic-learnings

# Pick a framework and set it up
cd lang-graph
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in your API keys

# Run any example
python 01_simple_graph.py
```

## What You'll Learn

### LangGraph — Graph-based stateful agents

| # | Concept |
|---|---------|
| 01 | StateGraph basics — nodes, edges, invoke |
| 02 | Tool calling + conditional routing |
| 03 | Memory + checkpointing (MemorySaver) |
| 04 | Human-in-the-loop — pause, approve, reject |
| 05 | Multi-agent supervisor pattern |
| 06 | Streaming — tokens, updates, snapshots |
| 07 | Subgraphs — modular, reusable graph components |
| 08 | Parallel execution — static fan-out + Send API |
| 09 | SQLite persistence — memory across restarts |
| 10 | ReAct agent from scratch — no prebuilts |
| 11 | End-to-end production agent (capstone) |
| 12 | Real-world APIs — live weather, crypto, Wikipedia |

### CrewAI — Role-based multi-agent teams

| # | Concept |
|---|---------|
| 01 | First Agent — role, goal, backstory, task, crew |
| 02 | Agent with tools — SerperDevTool, dynamic inputs |
| 03 | Two-agent sequential crew — automatic context handoff |
| 04 | Task dependencies — explicit context=[] wiring |
| 05 | Hierarchical crew — Manager agent delegates dynamically |
| 06 | Custom tools — @tool decorator and BaseTool class |
| 07 | Structured output — Pydantic models from tasks |
| 08 | Memory — short-term, long-term, entity across runs |
| 09 | Human-in-the-loop — review gates mid-pipeline |
| 10 | Flow basics — @start, @listen, @router, FlowState |
| 11 | Flow + Crew — Flow orchestrating multiple Crews |
| 12 | Real-world pipeline — live APIs, Pydantic, memory, Flow (capstone) |

## Framework Comparison

Not sure which framework fits your use case? See the detailed side-by-side breakdown:

**[LangGraph vs CrewAI — A Practical Comparison](LANGGRAPH_VS_CREWAI.md)**

Covers: core philosophy, state management, routing, memory, code examples, honest trade-offs, and a quick decision guide — built from hands-on experience with 12 examples in each framework.

---

## API Keys Needed

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | LangGraph 01–11, CrewAI all examples |
| `OPENAI_API_KEY` | LangGraph 12 |
| `SERPER_API_KEY` | CrewAI 02, 12 (optional — free tier at serper.dev) |

Free APIs used with no key: Open-Meteo, Wikipedia, REST Countries, CoinGecko, HackerNews, GitHub Search.
