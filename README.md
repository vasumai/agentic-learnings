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

## LangGraph — What You'll Learn

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

## Framework Comparison

Not sure which framework fits your use case? See the detailed side-by-side breakdown:

**[LangGraph vs CrewAI — A Practical Comparison](LANGGRAPH_VS_CREWAI.md)**

Covers: core philosophy, state management, routing, memory, code examples, honest trade-offs, and a quick decision guide — built from hands-on experience with 12 examples in each framework.

---

## API Keys Needed

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | LangGraph examples 01–11 |
| `OPENAI_API_KEY` | LangGraph example 12 |

All other APIs in example 12 (Open-Meteo, Wikipedia, REST Countries, CoinGecko) are free with no key.
