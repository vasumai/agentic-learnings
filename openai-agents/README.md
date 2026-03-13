# OpenAI Agents SDK — Learning Curriculum

A 12-lesson hands-on curriculum for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).
Each lesson is a self-contained Python file that teaches one core concept, building from basics to a full capstone.

---

## Setup

```bash
cd openai-agents
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your key:

```
OPENAI_API_KEY=sk-...
```

---

## Lessons

| # | File | Concept | Status |
|---|------|---------|--------|
| 01 | `01_hello_agents.py` | **Hello Agents** — `Agent`, `Runner.run()`, single-turn Q&A, multi-turn with `to_input_list()` | ✅ Done |
| 02 | `02_function_tools.py` | **Function Tools** — `@function_tool`, type hints as schema, docstring as description, `RunContextWrapper[T]` | ✅ Done |
| 03 | `03_handoffs.py` | **Handoffs** — agent-to-agent delegation, triage routing, `on_handoff` callback, `result.last_agent` | ✅ Done |
| 04 | `04_context.py` | **Context** — `RunContextWrapper[T]` typed context, passing state across tools and agents in a run | 🔜 Next |
| 05 | `05_structured_output.py` | **Structured Output** — `output_type=PydanticModel`, `result.final_output_as()`, typed agent responses | ⬜ |
| 06 | `06_multi_agent_triage.py` | **Multi-Agent Triage** — orchestrator + multiple specialists, chained handoffs, escalation patterns | ⬜ |
| 07 | `07_guardrails.py` | **Guardrails** — `@input_guardrail`, `@output_guardrail`, `tripwire_triggered`, `InputGuardrailTripwireTriggered` | ⬜ |
| 08 | `08_streaming.py` | **Streaming** — `Runner.run_streamed()`, `stream_events()`, event types, cancellation | ⬜ |
| 09 | `09_lifecycle_hooks.py` | **Lifecycle Hooks** — `RunHooks`, `AgentHooks`, `on_tool_start`, `on_tool_end`, `on_handoff`, observability | ⬜ |
| 10 | `10_human_in_the_loop.py` | **Human-in-the-Loop** — approval gates on tools, `RunState`, pausing and resuming a run | ⬜ |
| 11 | `11_mcp_integration.py` | **MCP Integration** — `MCPServerStdio`, `MCPServerStdioParams`, connect/cleanup lifecycle | ⬜ |
| 12 | `12_capstone.py` | **Capstone** — all patterns combined: triage + tools + context + guardrails + streaming + HITL | ⬜ |

---

## Concept Map

```
Agent + Runner.run()          → Lesson 01  (the foundation)
@function_tool + context      → Lesson 02  (giving agents abilities)
handoff()                     → Lesson 03  (multi-agent routing)
RunContextWrapper[T]          → Lesson 04  (shared typed state)
output_type=PydanticModel     → Lesson 05  (structured responses)
Orchestrator + specialists    → Lesson 06  (real-world triage)
@input/output_guardrail       → Lesson 07  (safety + validation)
run_streamed() + events       → Lesson 08  (real-time output)
RunHooks + AgentHooks         → Lesson 09  (observability)
Approval gates + RunState     → Lesson 10  (human oversight)
MCPServerStdio                → Lesson 11  (external tool servers)
Everything together           → Lesson 12  (capstone)
```

---

## Key SDK Primitives

| Primitive | Description |
|-----------|-------------|
| `Agent` | A configured LLM persona — name, instructions, model, tools, handoffs |
| `Runner.run()` | Async entry point — runs an agent against a list of messages |
| `RunResult` | Returned by `Runner.run()` — holds `final_output`, `last_agent`, `to_input_list()` |
| `@function_tool` | Decorator that turns a Python function into an agent tool |
| `RunContextWrapper[T]` | Typed wrapper giving tools/hooks access to shared run context |
| `handoff()` | Delegates control from one agent to another |
| `@input_guardrail` | Validates/screens user input before the agent processes it |
| `@output_guardrail` | Validates/screens agent output before it reaches the user |
| `Runner.run_streamed()` | Streaming variant — yields events as the run progresses |
| `RunHooks` / `AgentHooks` | Lifecycle callbacks for observability and side effects |
| `MCPServerStdio` | Connects to an external MCP tool server over stdio |

---

## Models Used

- `gpt-4o-mini` — used in most lessons (fast, cheap, capable enough)
- `gpt-4o` — used in the capstone (lesson 12)
