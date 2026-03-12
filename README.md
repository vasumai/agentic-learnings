# Agentic AI Learning Journey

A hands-on, incremental learning repository for four major agentic AI frameworks —
built example by example, concept by concept, in a deliberate learning sequence.

---

## Recommended Learning Sequence

> **Start here if you are new to agentic AI frameworks.**

The frameworks are ordered intentionally. Each one builds on mental models
you develop in the previous one.

```
LangGraph  →  CrewAI  →  Semantic Kernel  →  Google ADK
  (how)        (who)      (what + enterprise)   (Gemini-native)
```

| Step | Framework | Why this order |
|------|-----------|----------------|
| **1. LangGraph** | Graph-based stateful agents | Teaches the *mechanics* — state, edges, routing, checkpointing. You build agents from first principles. No magic. |
| **2. CrewAI** | Role-based multi-agent teams | Teaches the *team metaphor* — agents with roles, goals, and delegated tasks. Higher abstraction than LangGraph. |
| **3. Semantic Kernel** | Microsoft's enterprise AI SDK | Teaches the *plugin/kernel model* — the most structured, enterprise-ready approach. |
| **4. Google ADK** | Google's open-source agent framework | Teaches *Gemini-native multi-agent orchestration* — SequentialAgent, ParallelAgent, LoopAgent, MCP, and HITL callbacks. |

Each framework has 12 lessons (simple → advanced → capstone). Complete all 12
in one framework before moving to the next.

---

## Frameworks

| Folder | Framework | Lessons | Status |
|--------|-----------|---------|--------|
| [lang-graph/](lang-graph/) | LangGraph + LangChain | 12 lessons — graph basics to production agent | ✅ Complete |
| [crew-ai/](crew-ai/) | CrewAI | 12 lessons — role-based multi-agent teams | ✅ Complete |
| [semantic-kernel/](semantic-kernel/) | Semantic Kernel (Microsoft) | 12 lessons — plugins, memory, agents, MCP | ✅ Complete |
| [google-adk/](google-adk/) | Google ADK | 12 lessons — Gemini-native multi-agent pipelines | ✅ Complete |

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

### Step 4 — Google ADK

```bash
cd google-adk
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add GOOGLE_API_KEY (enable billing — see API-KEYS.md)
python 01_hello_adk.py
```

> **Google ADK note:** The free tier has a 20 requests/day hard cap — you will hit
> it mid-lesson. Enable billing on your Google Cloud project first (2 minutes,
> costs pennies). See [API-KEYS.md](API-KEYS.md) for step-by-step instructions.

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
| 04 | Chat History — multi-turn, persona, windowing, serialise, template |
| 05 | Semantic Memory — embeddings, save/search, collections, TextMemoryPlugin |
| 06 | Function Calling — Auto/Required/NoneInvoke, filters, loop depth |
| 07 | Filters & Middleware — FUNCTION_INVOCATION, PROMPT_RENDERING, AUTO_FUNCTION_INVOCATION |
| 08 | ChatCompletionAgent — name/instructions, plugins, ChatHistoryAgentThread, streaming |
| 09 | Multi-Agent — manual orchestration (Anthropic), AgentGroupChat (OpenAI) |
| 10 | Human-in-the-Loop — AUTO_FUNCTION_INVOCATION filter, risk tiers, audit log |
| 11 | MCP Integration — MCPStdioPlugin, MCPSsePlugin, resources, real-world servers |
| 12 | Capstone — multi-agent Research & Report pipeline (all concepts combined) |

### Google ADK — Gemini-native multi-agent framework

> Google's open-source Python SDK for building multi-agent systems with Gemini. Best for Gemini-native pipelines with structured orchestration primitives.

| # | Concept |
|---|---------|
| 01 | Agent + Runner + InMemorySessionService — the three core ADK objects |
| 02 | Plain Python functions as tools — docstring = description, type hints = schema |
| 03 | FunctionTool, LongRunningFunctionTool, google_search; built-in tool mixing restriction |
| 04 | Multi-turn conversation, isolated sessions, session history inspection |
| 05 | output_schema (Pydantic structured output), output_key (saves to session state) |
| 06 | Callbacks — before/after agent, model, and tool hooks (ADK's middleware layer) |
| 07 | SequentialAgent — pipeline: ResearchAgent → AnalysisAgent → SummaryAgent |
| 08 | ParallelAgent — concurrent fan-out + merger agent pattern |
| 09 | LoopAgent — exit_loop tool, max_iterations safety net, writer/checker pattern |
| 10 | Human-in-the-Loop — before_tool_callback approval gate, risk-based routing |
| 11 | MCP Integration — McpToolset, StdioConnectionParams, tool_filter |
| 12 | Capstone — editorial pipeline: MCP + parallel analysis + structured output + HITL |

---

## Framework Comparison

Not sure which framework fits your use case? See the detailed side-by-side breakdown:

**[LangGraph vs CrewAI — A Practical Comparison](LANGGRAPH_VS_CREWAI.md)**

| Concept | LangGraph | CrewAI | Semantic Kernel | Google ADK |
|---------|-----------|--------|-----------------|------------|
| Core unit | StateGraph | Crew | Kernel | Agent + Runner |
| Tools | Node tool calls | @tool / BaseTool | @kernel_function Plugin | Plain Python functions |
| Multi-agent | Supervisor node | Task pipeline | AgentGroupChat | SequentialAgent / ParallelAgent / LoopAgent |
| Memory | MemorySaver / SQLite | memory=True | VectorStore / SK Memory | InMemorySessionService / session.state |
| HITL | interrupt_before | human_input=True | AUTO_FUNCTION_INVOCATION filter | before_tool_callback |
| MCP | Custom | Custom | MCPStdioPlugin | McpToolset (native) |
| Model | Any | Any | Any | Gemini (native) |
| Best for | Stateful graph pipelines | Role-based agent teams | Enterprise plugin architecture | Gemini-native multi-agent pipelines |

---

## API Keys

For full details on where to get each key, free vs paid tiers, and billing setup:

**[API-KEYS.md](API-KEYS.md)**

| Key | Used by | Free tier |
|-----|---------|-----------|
| `ANTHROPIC_API_KEY` | LangGraph, CrewAI, Semantic Kernel | Limited credits |
| `GOOGLE_API_KEY` | Google ADK | 20 req/day — **enable billing** |
| `OPENAI_API_KEY` | LangGraph 12, SK 09 & 12 | No — load $5 credits |
| `SERPER_API_KEY` | CrewAI 02 & 12 | 2,500 free searches |

Free APIs (no key needed): Open-Meteo, Wikipedia, REST Countries, CoinGecko, HackerNews, GitHub Search.

---

## Community

Have a question about a lesson? Stuck on an error? Want to share something you built?

**[Join the Discussions](https://github.com/vasumai/agentic-learnings/discussions)**

| Category | Use it for |
|----------|-----------|
| [Q&A](https://github.com/vasumai/agentic-learnings/discussions/categories/q-a) | Questions about lessons, errors, concepts |
| [Show and Tell](https://github.com/vasumai/agentic-learnings/discussions/categories/show-and-tell) | Share what you built using these lessons |
| [Ideas](https://github.com/vasumai/agentic-learnings/discussions/categories/ideas) | Suggest new lessons, frameworks, or improvements |
| [General](https://github.com/vasumai/agentic-learnings/discussions/categories/general) | Anything else |
