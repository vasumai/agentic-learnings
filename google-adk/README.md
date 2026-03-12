# Google ADK — Agentic AI with Gemini

12-lesson hands-on curriculum for **Google Agent Development Kit (ADK)** —
Google's open-source Python framework for building multi-agent AI systems powered by Gemini.

---

## Setup

```bash
cd google-adk
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add your GOOGLE_API_KEY (see API-KEYS.md)
python 01_hello_adk.py
```

**Model used throughout:** `gemini-2.5-flash`
**API key:** `GOOGLE_API_KEY` from Google AI Studio — see [API-KEYS.md](../API-KEYS.md)

> **Tip:** Use a paid billing account (not the free tier). The free tier has
> a hard 20 requests/day cap that you will hit mid-lesson. Billing setup
> takes 2 minutes and costs pennies per lesson.

---

## Lessons

| # | File | Concept |
|---|------|---------|
| 01 | `01_hello_adk.py` | Agent + Runner + InMemorySessionService — the three core ADK objects |
| 02 | `02_tools.py` | Plain Python functions as tools — docstring = description, type hints = schema |
| 03 | `03_tool_types.py` | FunctionTool, LongRunningFunctionTool, google_search (built-in); mixing restriction |
| 04 | `04_multi_turn.py` | Multi-turn conversation, isolated sessions, session history inspection |
| 05 | `05_structured_output.py` | output_schema (Pydantic), output_key (saves to session state) |
| 06 | `06_callbacks.py` | before/after agent, model, and tool callbacks — ADK's middleware layer |
| 07 | `07_sequential_agent.py` | SequentialAgent — pipeline: ResearchAgent → AnalysisAgent → SummaryAgent |
| 08 | `08_parallel_agent.py` | ParallelAgent — concurrent fan-out + merger agent pattern |
| 09 | `09_loop_agent.py` | LoopAgent — exit_loop tool, max_iterations safety net, writer/checker pattern |
| 10 | `10_human_in_the_loop.py` | HITL via before_tool_callback approval gate and risk-based routing |
| 11 | `11_mcp_integration.py` | McpToolset — connect any MCP server to an ADK agent; tool_filter |
| 12 | `12_capstone.py` | Capstone: full editorial pipeline combining all concepts (MCP + parallel + HITL + structured output) |

---

## Key Concepts

### The three core objects

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()
agent = Agent(name="my_agent", model="gemini-2.5-flash", instruction="...", tools=[...])
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)
```

### Tools are plain Python functions

```python
def get_weather(city: str) -> dict:
    """Returns current weather for a city."""
    return {"city": city, "temp_c": 22}

agent = Agent(..., tools=[get_weather])
```

### Multi-agent patterns

| Pattern | Class | Use when |
|---------|-------|----------|
| Pipeline | `SequentialAgent` | Steps must run in order, each feeding the next |
| Fan-out | `ParallelAgent` | Independent tasks that can run concurrently |
| Loop | `LoopAgent` | Repeat until a condition is met (checker calls `exit_loop`) |

### Callbacks — intercept any stage

| Callback | Signature | Return to block |
|----------|-----------|-----------------|
| `before_agent_callback` | `(callback_context)` | `types.Content` |
| `after_agent_callback` | `(callback_context)` | — (ignored) |
| `before_model_callback` | `(callback_context, llm_request)` | `LlmResponse` |
| `before_tool_callback` | `(tool, args, tool_context)` | `dict` |
| `after_tool_callback` | `(tool, args, tool_context, tool_response)` | `dict` |

> Return `None` from any callback to proceed normally.

### MCP Integration

```python
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command="python", args=["my_server.py"]),
        timeout=30.0,
    ),
    tool_filter=["tool_a", "tool_b"],  # optional
)
agent = Agent(..., tools=[toolset])
# Always clean up:
await toolset.close()
```

---

## ADK vs Other Frameworks

| Feature | Google ADK | LangGraph | CrewAI | Semantic Kernel |
|---------|-----------|-----------|--------|-----------------|
| Core unit | Agent + Runner | StateGraph | Crew | Kernel |
| Multi-agent | SequentialAgent / ParallelAgent / LoopAgent | Supervisor node | Task pipeline | AgentGroupChat |
| State | InMemorySessionService / session.state | MemorySaver | memory=True | VectorStore |
| HITL | before_tool_callback | interrupt_before | human_input=True | AUTO_FUNCTION_INVOCATION filter |
| MCP | McpToolset (native) | Custom | Custom | MCPStdioPlugin |
| Model | Gemini (native), any via LiteLLM | Any | Any | Any |
| Best for | Gemini-native multi-agent pipelines | Stateful graph control flow | Role-based agent teams | Enterprise plugin architecture |
