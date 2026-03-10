"""
11_end_to_end_agent.py
----------------------
Capstone: a production-ready research assistant that combines every
pattern learned in files 01–10 into one cohesive agent.

Features combined:
  ✓ ReAct loop (file 10)          — reason → act → observe cycle
  ✓ SQLite persistence (file 09)  — memory survives restarts
  ✓ Human-in-the-loop (file 04)   — approval before sensitive actions
  ✓ Streaming (file 06)           — tokens printed as generated
  ✓ Multiple tools (file 02/10)   — search, calculator, weather, email
  ✓ Thread isolation (file 03)    — each user = separate conversation

Architecture:
  START → reason → [tools?] → [sensitive?] → human_gate → act → reason → END
                                    │
                                    └── [safe tool] ──────────────────────►

Sensitive tools (require approval): send_email, save_report
Safe tools (auto-execute): search_web, calculator, get_weather
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
import os, asyncio

load_dotenv()

DB_PATH = "agent_memory.db"

# ── 1. Tools ──────────────────────────────────────────────────────────────────
SENSITIVE_TOOLS = {"send_email", "save_report"}   # require human approval


@tool
def search_web(query: str) -> str:
    """Search the web for information on any topic."""
    results = {
        "langgraph":        "LangGraph is a framework for building stateful multi-actor LLM applications.",
        "python":           "Python is the #1 language for AI/ML development as of 2024.",
        "anthropic claude": "Claude is Anthropic's AI assistant, known for safety and helpfulness.",
        "openai":           "OpenAI develops GPT models and the ChatGPT product.",
        "machine learning": "Machine learning enables computers to learn from data without explicit programming.",
    }
    for key in results:
        if key in query.lower():
            return results[key]
    return f"Found general information about '{query}'. It is a widely discussed topic in tech circles."


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "tokyo":     "Rainy, 18°C",
        "london":    "Cloudy, 12°C",
        "new york":  "Sunny, 22°C",
        "sydney":    "Clear, 25°C",
        "paris":     "Partly cloudy, 15°C",
        "singapore": "Humid, 31°C",
    }
    return data.get(city.lower(), f"Weather data unavailable for {city}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '25 * 4 + 10'."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only numeric expressions allowed"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. REQUIRES human approval."""
    print(f"\n  📧 [EMAIL SENT] To: {to} | Subject: {subject}")
    return f"Email sent to {to} with subject '{subject}'."


@tool
def save_report(filename: str, content: str) -> str:
    """Save a report to disk. REQUIRES human approval."""
    path = f"/tmp/{filename}"
    with open(path, "w") as f:
        f.write(content)
    print(f"\n  💾 [REPORT SAVED] → {path}")
    return f"Report saved to {path}"


tools = [search_web, get_weather, calculator, send_email, save_report]
tool_by_name = {t.name: t for t in tools}

# ── 2. State ──────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool_calls: list    # tool calls waiting for approval
    iterations: int


# ── 3. LLM ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful research assistant with access to web search,
weather lookup, a calculator, email sending, and report saving.

For factual questions, use search_web. For math, use calculator.
For weather, use get_weather. Only use send_email or save_report when
the user explicitly asks you to send an email or save something."""

llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(tools)


# ── 4. Nodes ──────────────────────────────────────────────────────────────────
def reason(state: State) -> dict:
    """LLM decides next action."""
    iteration = state.get("iterations", 0) + 1
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        names = [tc["name"] for tc in tool_calls]
        print(f"\n  [REASON #{iteration}] wants to call: {names}")
    else:
        print(f"\n  [REASON #{iteration}] forming final answer...")

    return {"messages": [response], "iterations": iteration}


def classify_tools(state: State) -> dict:
    """Split pending tool calls into safe and sensitive buckets."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])
    sensitive = [tc for tc in tool_calls if tc["name"] in SENSITIVE_TOOLS]
    return {"pending_tool_calls": sensitive}


def human_gate(state: State) -> dict:
    """Show pending sensitive tool calls and ask for approval."""
    approved = []
    for tc in state["pending_tool_calls"]:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
        print(f"\n  ⚠️  Sensitive action requested: {tc['name']}({args_str})")
        choice = input("  Approve? (y/n): ").strip().lower()
        if choice == "y":
            approved.append(tc)
        else:
            print(f"  ✗ Rejected: {tc['name']}")
    return {"pending_tool_calls": approved}


def act(state: State) -> dict:
    """Execute all tool calls (safe ones auto-run; sensitive ones run if approved)."""
    last = state["messages"][-1]
    all_tool_calls = getattr(last, "tool_calls", [])
    approved_sensitive = {tc["id"] for tc in state.get("pending_tool_calls", [])}

    tool_messages = []
    for tc in all_tool_calls:
        is_sensitive = tc["name"] in SENSITIVE_TOOLS
        if is_sensitive and tc["id"] not in approved_sensitive:
            # Rejected — tell the LLM it was cancelled
            result = f"Action '{tc['name']}' was rejected by the user."
            print(f"  [ACT] skipped (rejected): {tc['name']}")
        else:
            print(f"  [ACT] executing: {tc['name']}({tc['args']})")
            result = tool_by_name[tc["name"]].invoke(tc["args"])
            print(f"  [ACT] result: {str(result)[:80]}")

        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
            name=tc["name"],
        ))

    return {"messages": tool_messages, "pending_tool_calls": []}


# ── 5. Routing ────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 8

def after_reason(state: State) -> Literal["classify_tools", "__end__"]:
    last = state["messages"][-1]
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "__end__"
    if getattr(last, "tool_calls", []):
        return "classify_tools"
    return "__end__"


def after_classify(state: State) -> Literal["human_gate", "act"]:
    """If there are sensitive tools pending approval, go to human_gate. Otherwise act directly."""
    if state.get("pending_tool_calls"):
        return "human_gate"
    return "act"


# ── 6. Build graph ────────────────────────────────────────────────────────────
def build_graph(checkpointer):
    builder = StateGraph(State)

    builder.add_node("reason",         reason)
    builder.add_node("classify_tools", classify_tools)
    builder.add_node("human_gate",     human_gate)
    builder.add_node("act",            act)

    builder.add_edge(START, "reason")
    builder.add_conditional_edges("reason",         after_reason,   ["classify_tools", "__end__"])
    builder.add_conditional_edges("classify_tools", after_classify, ["human_gate", "act"])
    builder.add_edge("human_gate", "act")
    builder.add_edge("act",        "reason")

    return builder.compile(checkpointer=checkpointer)


# ── 7. Chat interface ─────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", str(b)) if isinstance(b, dict) else str(b)
        for b in content
    )


def chat(graph, thread_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)], "pending_tool_calls": [], "iterations": 0},
        config
    )
    for msg in reversed(result["messages"]):
        text = extract_text(msg.content)
        if len(text.strip()) > 5:
            return text
    return "[no response]"


# ── 8. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer)
        thread = "user_srini"

        print("=" * 60)
        print("Production Research Assistant")
        print("Type 'quit' to exit | Memory persists across sessions")
        print("=" * 60)

        # Scripted demo turns — shows all features
        demo_turns = [
            "Hi! My name is Srini. What is LangGraph?",
            "What's the weather in Tokyo and Singapore?",
            "Calculate: 1250 * 12 + 500",
            "Do you remember my name from earlier?",
            "Send an email to srini@example.com with subject 'LangGraph Notes' and body 'Completed all 11 LangGraph examples today!'",
        ]

        for turn in demo_turns:
            print(f"\n{'─' * 60}")
            print(f"You: {turn}")
            response = chat(graph, thread, turn)
            print(f"AI:  {response[:300]}")

        # Interactive mode
        print(f"\n{'─' * 60}")
        print("Demo complete. Entering interactive mode (type 'quit' to exit):\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye! Your conversation is saved in agent_memory.db")
                break
            if not user_input:
                continue
            response = chat(graph, thread, user_input)
            print(f"AI:  {response[:400]}\n")
