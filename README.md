# Agentic AI Learning Journey

A hands-on, incremental learning repository for three major agentic AI frameworks —
built example by example, concept by concept, in a deliberate learning sequence.

---

## Recommended Learning Sequence

> **Start here if you are new to agentic AI frameworks.**

The three frameworks are ordered intentionally. Each one builds on mental models
you develop in the previous one. Jumping straight to Semantic Kernel without
LangGraph context, for example, makes the orchestration patterns much harder to internalize.

```
LangGraph  →  CrewAI  →  Semantic Kernel
  (how)        (who)         (what + enterprise)
```

| Step | Framework | Why this order |
|------|-----------|----------------|
| **1. LangGraph** | Graph-based stateful agents | Teaches the *mechanics* — state, edges, routing, checkpointing. You build agents from first principles. No magic. |
| **2. CrewAI** | Role-based multi-agent teams | Teaches the *team metaphor* — agents with roles, goals, and delegated tasks. Higher abstraction than LangGraph. |
| **3. Semantic Kernel** | Microsoft's enterprise AI SDK | Teaches the *plugin/kernel model* — the most structured, enterprise-ready approach. References to LangGraph and CrewAI patterns appear throughout the code comments. |

Each framework has 12 lessons (simple → advanced → capstone). Complete all 12
in one framework before moving to the next.

---

## Frameworks

| Folder | Framework | Lessons | Status |
|--------|-----------|---------|--------|
| [lang-graph/](lang-graph/) | LangGraph + LangChain | 12 lessons — graph basics to production agent | ✅ Complete |
| [crew-ai/](crew-ai/) | CrewAI | 12 lessons — role-based multi-agent teams | ✅ Complete |
| [semantic-kernel/](semantic-kernel/) | Semantic Kernel (Microsoft) | 12 lessons — plugins, memory, agents, MCP | 🔄 In Progress (03/12) |

---

## What's Inside Each Framework Folder

- Numbered Python files (`01_`, `02_`, ...) — one concept per file
- Its own `README.md` with a full lesson table
- Its own `requirements.txt` and `.env.example`
- Its own Python virtual environment (`venv/`) — not committed to git

---

## Getting Started

### Step 1 — LangGraph

```bash
cd lang-graph
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add ANTHROPIC_API_KEY
python 01_simple_graph.py
```

### Step 2 — CrewAI

```bash
cd crew-ai
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add ANTHROPIC_API_KEY
python 01_first_agent.py
```

### Step 3 — Semantic Kernel

```bash
cd semantic-kernel
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add ANTHROPIC_API_KEY
python 01_hello_semantic_kernel.py
```

---

## Curriculum Summary

### LangGraph — Graph-based stateful agents

> Build agents from scratch using explicit state machines. Best for understanding *how* agent loops work.

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

> Assign roles, goals, and tasks to agents. Best for *who does what* in a team of agents.

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

### Semantic Kernel — Microsoft's enterprise AI SDK

> Plugin-based architecture with structured prompt templates, memory, planners, and MCP support. Best for production enterprise AI apps.

| # | Concept |
|---|---------|
| 01 | Kernel, ChatHistory, invoke_prompt, multi-turn chat |
| 02 | Native Plugins — @kernel_function, direct & auto invocation |
| 03 | Prompt Templates — {{$variable}}, reuse, PromptTemplateConfig |
| 04 | Chat History & Conversation State _(coming soon)_ |
| 05 | Semantic Memory + Embeddings _(coming soon)_ |
| 06 | Planners — auto function selection _(coming soon)_ |
| 07 | Filters & Middleware _(coming soon)_ |
| 08 | Agents — ChatCompletionAgent _(coming soon)_ |
| 09 | Multi-Agent Collaboration _(coming soon)_ |
| 10 | Human-in-the-Loop _(coming soon)_ |
| 11 | MCP Integration _(coming soon)_ |
| 12 | Capstone — Real-World Pipeline _(coming soon)_ |

---

## Framework Comparison

Not sure which framework fits your use case? See the detailed side-by-side breakdown:

**[LangGraph vs CrewAI — A Practical Comparison](LANGGRAPH_VS_CREWAI.md)**

Covers: core philosophy, state management, routing, memory, code examples, honest trade-offs,
and a quick decision guide — built from hands-on experience with 12 lessons in each framework.

| Concept | LangGraph | CrewAI | Semantic Kernel |
|---------|-----------|--------|-----------------|
| Core unit | StateGraph | Crew | Kernel |
| Functions / Tools | Node tool calls | @tool / BaseTool | @kernel_function Plugin |
| Prompt definition | PromptTemplate (LangChain) | Task description string | kernel.add_function(prompt=...) |
| Multi-agent | Supervisor node | Crew(agents) | AgentGroup |
| Memory | MemorySaver / SQLite | memory=True | VectorStore / SK Memory |
| Orchestration | Edge routing | Task pipeline | Planner / invoke() |
| Human-in-loop | interrupt_before | human_input=True | Custom filter |
| Best for | Stateful graph pipelines | Role-based agent teams | Enterprise plugin architecture |

---

## API Keys Needed

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | All three frameworks |
| `OPENAI_API_KEY` | LangGraph lesson 12 only |
| `SERPER_API_KEY` | CrewAI lessons 02, 12 (optional — free tier at serper.dev) |

Free APIs used with no key: Open-Meteo, Wikipedia, REST Countries, CoinGecko, HackerNews, GitHub Search.
