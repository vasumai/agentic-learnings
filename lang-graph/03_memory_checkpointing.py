"""
03_memory_checkpointing.py
--------------------------
LangGraph graphs are stateless by default — each invoke() starts fresh.
Checkpointing adds persistence: the graph saves state after every step,
so you can resume a conversation across multiple invoke() calls.

Key concept: thread_id
  Every conversation gets a unique thread_id in the config dict.
  The checkpointer uses it to save/load the right state.
  Same thread_id = same conversation memory.
  Different thread_id = fresh conversation.

Concepts introduced:
  MemorySaver, thread_id, config, multi-turn conversation,
  get_state() to inspect saved memory
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import Literal

load_dotenv()


# ── 1. State ─────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. Tools (same as before) ─────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "london": "Cloudy, 12°C",
        "new york": "Sunny, 22°C",
        "tokyo": "Rainy, 18°C",
        "paris": "Partly cloudy, 15°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


tools = [get_weather]


# ── 3. LLM + nodes ───────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(tools)


def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_use_tools(state: State) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


# ── 4. Build graph WITH checkpointer ─────────────────────────────────────────
# MemorySaver stores state in-memory (perfect for learning).
# In production you'd swap this for SqliteSaver or PostgresSaver.
memory = MemorySaver()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_use_tools)
graph_builder.add_edge("tools", "chatbot")

# compile() now receives the checkpointer
graph = graph_builder.compile(checkpointer=memory)


# ── 5. Helper to chat ─────────────────────────────────────────────────────────
def chat(thread_id: str, user_message: str) -> str:
    """Send a message in a conversation identified by thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_message)]}, config)
    return result["messages"][-1].content


# ── 6. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Conversation A: multi-turn, the graph remembers context ──────────────
    print("=" * 60)
    print("CONVERSATION A  (thread: alice)")
    print("=" * 60)

    turns_a = [
        "Hi! My name is Alice.",
        "What's the weather in Tokyo?",
        "And what about London?",
        "Do you remember my name?",   # ← tests memory — no tool needed
    ]

    for msg in turns_a:
        print(f"\nUser: {msg}")
        reply = chat("alice", msg)
        print(f"AI:   {reply}")

    # ── Conversation B: fresh thread — no memory of Alice ────────────────────
    print("\n" + "=" * 60)
    print("CONVERSATION B  (thread: bob) — completely separate memory")
    print("=" * 60)

    turns_b = [
        "My name is Bob.",
        "Do you know anyone named Alice?",  # ← should NOT know about Alice
    ]

    for msg in turns_b:
        print(f"\nUser: {msg}")
        reply = chat("bob", msg)
        print(f"AI:   {reply}")

    # ── Inspect saved state for thread alice ─────────────────────────────────
    print("\n" + "=" * 60)
    print("INSPECTING saved state for thread: alice")
    print("=" * 60)
    config = {"configurable": {"thread_id": "alice"}}
    saved_state = graph.get_state(config)
    print(f"Total messages stored: {len(saved_state.values['messages'])}")
    for i, m in enumerate(saved_state.values["messages"]):
        role = m.__class__.__name__.replace("Message", "")
        # content can be a list (tool use) — show just text portions
        content = m.content if isinstance(m.content, str) else str(m.content)[:80]
        print(f"  [{i}] {role:12s}: {content[:70]}")
