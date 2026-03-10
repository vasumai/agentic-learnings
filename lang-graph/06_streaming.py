"""
06_streaming.py
---------------
By default, graph.invoke() blocks until the entire response is ready.
Streaming lets you receive output incrementally — token by token or
node by node — so users see results immediately instead of waiting.

LangGraph offers three streaming modes:
  - "values"   → emits the full state after each node completes
  - "updates"  → emits only what changed in state after each node
  - "messages" → emits LLM tokens as they are generated (typewriter effect)

Concepts introduced:
  graph.stream(), stream_mode, astream_events, token-level streaming,
  streaming in multi-node graphs
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
import time

load_dotenv()


# ── 1. State ─────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. Tools ──────────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "london":   "Cloudy, 12°C",
        "new york": "Sunny, 22°C",
        "tokyo":    "Rainy, 18°C",
        "paris":    "Partly cloudy, 15°C",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


tools = [get_weather]
llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(tools)
tool_node = ToolNode(tools)


# ── 3. Nodes ──────────────────────────────────────────────────────────────────
def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def should_use_tools(state: State) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# ── 4. Graph ──────────────────────────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_use_tools)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()


# ── 5. Helpers ────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", str(block)) if isinstance(block, dict) else str(block)
        for block in content
    )

def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ── 6. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Mode 1: stream_mode="updates" ────────────────────────────────────────
    # Emits a dict of {node_name: state_changes} after each node completes.
    # Useful for knowing which node ran and what it produced.
    separator("MODE 1: stream updates (node-by-node)")

    query = "What's the weather in Tokyo and Paris?"
    print(f"User: {query}\n")

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="updates"
    ):
        for node_name, state_update in chunk.items():
            msgs = state_update.get("messages", [])
            for msg in msgs:
                text = extract_text(msg.content)
                if text.strip():
                    print(f"  [{node_name}] {text[:120]}{'...' if len(text) > 120 else ''}")


    # ── Mode 2: stream_mode="values" ─────────────────────────────────────────
    # Emits the FULL state after every node — useful for watching state evolve.
    separator("MODE 2: stream values (full state after each node)")

    query2 = "Tell me the weather in London."
    print(f"User: {query2}\n")

    for i, state_snapshot in enumerate(graph.stream(
        {"messages": [HumanMessage(content=query2)]},
        stream_mode="values"
    )):
        last_msg = state_snapshot["messages"][-1]
        text = extract_text(last_msg.content)
        role = type(last_msg).__name__.replace("Message", "")
        if text.strip():
            print(f"  [snapshot {i}] {role}: {text[:120]}{'...' if len(text) > 120 else ''}")


    # ── Mode 3: token-level streaming via astream_events ─────────────────────
    # This gives you individual tokens as the LLM generates them.
    # "on_chat_model_stream" fires for each token — perfect for typewriter UX.
    # Note: this uses asyncio since astream_events is an async generator.
    separator("MODE 3: token-level streaming (typewriter effect)")

    import asyncio

    async def stream_tokens(user_message: str):
        print(f"User: {user_message}")
        print("AI:  ", end="", flush=True)

        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=user_message)]},
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # chunk.content can be a string or list of blocks
                content = chunk.content
                if isinstance(content, str) and content:
                    print(content, end="", flush=True)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                print(text, end="", flush=True)
        print()  # newline after streaming finishes

    asyncio.run(stream_tokens("Explain in 3 sentences why the sky is blue."))
