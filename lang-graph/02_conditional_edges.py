"""
02_conditional_edges.py
-----------------------
Conditional edges let the graph decide which node to go to next
based on the current state — this is what makes LangGraph an agent framework.

Pattern: the LLM decides whether to call a tool or respond directly.
  - If the last message has tool_calls → route to "tools" node
  - Otherwise → route to END

Concepts introduced:
  tools, tool binding, ToolNode, conditional_edges, tool-calling loop
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

load_dotenv()


# ── 1. State (same pattern as before) ────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. Define Tools ──────────────────────────────────────────────────────────
# Tools are plain Python functions decorated with @tool.
# The docstring becomes the tool description the LLM uses to decide when to call it.

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Mocked for learning purposes
    weather_data = {
        "london": "Cloudy, 12°C",
        "new york": "Sunny, 22°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def get_population(city: str) -> str:
    """Get the population of a given city."""
    population_data = {
        "london": "~9 million",
        "new york": "~8 million",
        "tokyo": "~14 million",
    }
    return population_data.get(city.lower(), f"Population data not available for {city}")


tools = [get_weather, get_population]


# ── 3. LLM with tools bound ──────────────────────────────────────────────────
# bind_tools tells the LLM what tools are available so it can decide to call them
llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(tools)


# ── 4. Nodes ─────────────────────────────────────────────────────────────────
def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode handles tool execution automatically:
# it reads tool_calls from the last AI message and runs the matching functions
tool_node = ToolNode(tools)


# ── 5. Routing function (the "conditional" part) ─────────────────────────────
# Returns the name of the next node to go to.
def should_use_tools(state: State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── 6. Build the Graph ───────────────────────────────────────────────────────
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

# Conditional edge: after "chatbot", call should_use_tools() to decide next node
graph_builder.add_conditional_edges("chatbot", should_use_tools)

# After tools run, always go back to chatbot (so it can form a final response)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


# ── 7. Run it ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    queries = [
        "What's the weather like in Tokyo?",
        "Tell me the population and weather of London.",
        "What is the capital of France?",  # no tool needed — goes straight to END
    ]

    for query in queries:
        print(f"{'─' * 60}")
        print(f"User: {query}")
        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        print(f"AI:   {result['messages'][-1].content}")
        print()
