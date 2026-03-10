"""
09_sqlite_persistence.py
------------------------
MemorySaver (from file 03) stores state in RAM — it vanishes when the
program exits. For real applications you need persistence that survives
restarts, deployments, and crashes.

SqliteSaver stores every checkpoint in a local SQLite database file.
The graph can resume any conversation from any point — even after restarting
the Python process — as long as it has the same thread_id.

Real-world relevance:
  - Chatbots that remember users across sessions
  - Long-running workflows that can be paused and resumed
  - Audit trails — every state transition is recorded

Concepts introduced:
  SqliteSaver, persistent checkpoints, resuming across process restarts,
  listing all threads, replaying conversation history
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
import os

load_dotenv()

DB_PATH = "conversations.db"   # SQLite file written to current directory

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ── 1. Graph (same structure as 03, different checkpointer) ───────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")


# ── 2. Helper ─────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", str(b)) if isinstance(b, dict) else str(b)
        for b in content
    )


def chat(graph, thread_id: str, user_message: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config
    )
    return extract_text(result["messages"][-1].content)


def print_history(graph, thread_id: str):
    """Print the full saved conversation for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.values.get("messages"):
        print("  (no history found)")
        return
    for msg in state.values["messages"]:
        role = type(msg).__name__.replace("Message", "")
        text = extract_text(msg.content)
        print(f"  {role:10s}: {text[:100]}")


# ── 3. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Clean up previous run so the demo always starts fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"(Removed previous {DB_PATH} for clean demo)\n")

    # SqliteSaver requires a sqlite3 connection — use as context manager
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = graph_builder.compile(checkpointer=checkpointer)

        # ── Session 1: start two conversations ───────────────────────────────
        print("=" * 60)
        print("SESSION 1 — starting conversations, then 'restarting'")
        print("=" * 60)

        print("\n[Thread: alice]")
        r = chat(graph, "alice", "Hi! I'm Alice. I love hiking and photography.")
        print(f"  AI: {r[:120]}")

        r = chat(graph, "alice", "What hobbies did I mention?")
        print(f"  AI: {r[:120]}")

        print("\n[Thread: bob]")
        r = chat(graph, "bob", "Hello, I'm Bob. I work as a software engineer.")
        print(f"  AI: {r[:120]}")

        print(f"\nCheckpoints saved to: {DB_PATH}")
        print(f"DB size: {os.path.getsize(DB_PATH):,} bytes")

    # ── Session 2: new process — reconnect and resume ─────────────────────────
    # This simulates a full application restart.
    # The graph is rebuilt from scratch, only the DB file persists.
    print("\n" + "=" * 60)
    print("SESSION 2 — simulating app restart, resuming from DB")
    print("=" * 60)

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer2:
        graph2 = graph_builder.compile(checkpointer=checkpointer2)

        # Alice's conversation resumes — the AI still knows her name and hobbies
        print("\n[Thread: alice — resuming after restart]")
        r = chat(graph2, "alice", "Do you still remember what I told you about myself?")
        print(f"  AI: {r[:200]}")

        # Bob's conversation also resumes
        print("\n[Thread: bob — resuming after restart]")
        r = chat(graph2, "bob", "What do you know about me so far?")
        print(f"  AI: {r[:200]}")

        # ── Inspect stored history ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Stored conversation history for thread: alice")
        print("=" * 60)
        print_history(graph2, "alice")

        print("\n" + "=" * 60)
        print("Stored conversation history for thread: bob")
        print("=" * 60)
        print_history(graph2, "bob")

        print(f"\nFinal DB size after session 2: {os.path.getsize(DB_PATH):,} bytes")
