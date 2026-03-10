"""
10_react_agent_from_scratch.py
------------------------------
ReAct = Reason + Act. It's the core loop inside every tool-calling agent.

In previous files we used ToolNode (a prebuilt) which hides the loop.
Here we build it manually so you understand exactly what's happening.

The ReAct loop:
  1. REASON  — LLM looks at messages and decides: call a tool OR answer directly
  2. ACT     — if a tool was chosen, execute it and add result to messages
  3. OBSERVE — LLM reads the tool result and reasons again
  4. REPEAT  — until the LLM decides no more tools are needed
  5. RESPOND — LLM produces the final answer

As a graph:
  START → reason → [tool chosen?] → act → reason → [done?] → END
                          ↑_______________|

The key insight: it's just a cycle. The routing function decides
whether to keep looping (more tools needed) or exit (final answer ready).

Concepts introduced:
  manual ReAct loop, ToolMessage, tool execution by name,
  reasoning trace (seeing every step), max_iterations guard
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
import json

load_dotenv()


# ── 1. Tools ──────────────────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    # Mocked results — simulates a real web search
    results = {
        "python popularity": "Python is ranked #1 in the TIOBE index as of 2024, used by 51% of developers.",
        "langgraph": "LangGraph is a framework by LangChain for building stateful, multi-actor LLM applications using graph-based orchestration.",
        "openai gpt-4": "GPT-4 is OpenAI's most capable model, supporting 128k context and multimodal inputs.",
    }
    for key in results:
        if key in query.lower():
            return results[key]
    return f"Search results for '{query}': No specific data found, but this is a popular topic."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '25 * 4 + 10'"""
    try:
        # Safe eval: only allow math operations
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only numeric expressions allowed"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    data = {
        "tokyo":    "Rainy, 18°C",
        "london":   "Cloudy, 12°C",
        "new york": "Sunny, 22°C",
        "sydney":   "Clear, 25°C",
    }
    return data.get(city.lower(), f"No weather data for {city}")


tools = [search_web, calculator, get_weather]

# Build a lookup dict so we can call tools by name during ACT step
tool_by_name = {t.name: t for t in tools}


# ── 2. State ──────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int     # tracks loop count — prevents infinite loops


# ── 3. LLM with tools bound ───────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(tools)


# ── 4. REASON node ────────────────────────────────────────────────────────────
# The LLM looks at all messages so far and decides:
#   - Call one or more tools (tool_calls will be populated)
#   - OR produce a final answer (no tool_calls)
def reason(state: State) -> dict:
    iteration = state.get("iterations", 0) + 1
    print(f"\n  [REASON #{iteration}] LLM thinking...")

    response = llm.invoke(state["messages"])

    # Show what the LLM decided
    content = response.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                print(f"  [REASON]  thought: {block['text'][:100]}")
    elif isinstance(content, str) and content.strip():
        print(f"  [REASON]  thought: {content[:100]}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f"  [REASON]  → will call: {tc['name']}({tc['args']})")

    return {"messages": [response], "iterations": iteration}


# ── 5. ACT node ───────────────────────────────────────────────────────────────
# Executes every tool the LLM requested, adds results as ToolMessages.
# ToolMessage is the standard way to return tool results back to the LLM.
def act(state: State) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id   = tool_call["id"]

        print(f"  [ACT]     executing: {tool_name}({tool_args})")

        if tool_name in tool_by_name:
            # Invoke the tool with its arguments
            result = tool_by_name[tool_name].invoke(tool_args)
        else:
            result = f"Error: tool '{tool_name}' not found"

        print(f"  [ACT]     result: {str(result)[:100]}")

        # Wrap result in ToolMessage — LLM needs this format to understand tool output
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id,    # must match the tool_call id from LLM
                name=tool_name,
            )
        )

    return {"messages": tool_messages}


# ── 6. Routing function ───────────────────────────────────────────────────────
MAX_ITERATIONS = 10   # safety limit — prevents runaway loops

def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    iterations   = state.get("iterations", 0)

    # Safety: stop if too many iterations
    if iterations >= MAX_ITERATIONS:
        print(f"  [ROUTE]   max iterations reached, forcing END")
        return "end"

    # If LLM produced tool calls → keep looping (go to act)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"  [ROUTE]   tools requested → continuing loop")
        return "act"

    # No tool calls → LLM has its final answer
    print(f"  [ROUTE]   no tools needed → END")
    return "end"


# ── 7. Build the graph ────────────────────────────────────────────────────────
graph_builder = StateGraph(State)

graph_builder.add_node("reason", reason)
graph_builder.add_node("act",    act)

graph_builder.add_edge(START, "reason")

# After reasoning: check if tools needed
graph_builder.add_conditional_edges(
    "reason",
    should_continue,
    {"act": "act", "end": END}
)

# After acting: always go back to reason (observe + reason again)
graph_builder.add_edge("act", "reason")

graph = graph_builder.compile()


# ── 8. Helper ─────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", str(b)) if isinstance(b, dict) else str(b)
        for b in content
    )


def run_query(query: str):
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print('=' * 60)

    result = graph.invoke({
        "messages": [HumanMessage(content=query)],
        "iterations": 0
    })

    final = ""
    for msg in reversed(result["messages"]):
        text = extract_text(msg.content)
        if len(text.strip()) > 10:
            final = text
            break

    print(f"\n  Final answer: {final}")


# ── 9. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Single tool call
    run_query("What's the weather like in Tokyo?")

    # Calculator — requires tool + direct answer
    run_query("What is 347 * 28 + 512?")

    # Multi-step: needs multiple tool calls in sequence
    run_query(
        "Search for information about LangGraph, then get the weather in London, "
        "and finally calculate 15 * 8. Give me a summary of all three."
    )
