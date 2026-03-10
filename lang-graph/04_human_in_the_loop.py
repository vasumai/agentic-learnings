"""
04_human_in_the_loop.py
-----------------------
Real agents shouldn't act autonomously on sensitive operations.
Human-in-the-loop lets the graph PAUSE before executing a tool,
show the user what it's about to do, and wait for approval.

How it works:
  - compile() receives interrupt_before=["tools"]
  - When the graph reaches the "tools" node it PAUSES and saves state
  - You inspect what tool it wants to call
  - You resume (approved) or stop (rejected)

Concepts introduced:
  interrupt_before, graph.get_state(), graph.invoke() with None to resume,
  Command(resume=...) for approval/modification
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

load_dotenv()


# ── 1. State ─────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. Tools — these are "sensitive" actions we want humans to approve ────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    # In reality this would call an email API
    print(f"\n  [EMAIL SENT] To: {to} | Subject: {subject}")
    return f"Email successfully sent to {to}."


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system."""
    # In reality this would delete a file
    print(f"\n  [FILE DELETED] {filename}")
    return f"File '{filename}' has been deleted."


tools = [send_email, delete_file]


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


# ── 4. Build graph with interrupt_before ─────────────────────────────────────
# interrupt_before=["tools"] tells LangGraph to PAUSE just before
# running the tools node — giving humans a chance to review.
memory = MemorySaver()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_use_tools)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]   # ← the key line
)


# ── 5. Helper: show pending tool calls ───────────────────────────────────────
def show_pending_tools(state_snapshot) -> list:
    """Print what tools the AI wants to call and return the tool_calls list."""
    last_msg = state_snapshot.values["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    print("\n  Pending tool calls:")
    for tc in tool_calls:
        args = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
        print(f"    → {tc['name']}({args})")
    return tool_calls


# ── 6. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo"}}

    # ── Scenario 1: User APPROVES the tool call ───────────────────────────────
    print("=" * 60)
    print("SCENARIO 1: Approve sending an email")
    print("=" * 60)

    msg = "Please send an email to alice@example.com with subject 'Hello' and body 'How are you?'"
    print(f"\nUser: {msg}")

    # First invoke — graph will pause before tools
    graph.invoke({"messages": [HumanMessage(content=msg)]}, config)

    # Inspect what the graph wants to do
    snapshot = graph.get_state(config)
    show_pending_tools(snapshot)

    # Simulate human approval
    user_input = input("\n  Approve? (y/n): ").strip().lower()
    if user_input == "y":
        # Resume by passing None as input — graph continues from checkpoint
        result = graph.invoke(None, config)
        print(f"\nAI: {result['messages'][-1].content}")
    else:
        print("\n  Action cancelled by user.")

    # ── Scenario 2: User REJECTS the tool call ────────────────────────────────
    print("\n" + "=" * 60)
    print("SCENARIO 2: Reject deleting a file")
    print("=" * 60)

    config2 = {"configurable": {"thread_id": "demo2"}}
    msg2 = "Delete the file named 'important_report.pdf'"
    print(f"\nUser: {msg2}")

    graph.invoke({"messages": [HumanMessage(content=msg2)]}, config2)

    snapshot2 = graph.get_state(config2)
    show_pending_tools(snapshot2)

    user_input2 = input("\n  Approve? (y/n): ").strip().lower()
    if user_input2 == "y":
        result2 = graph.invoke(None, config2)
        print(f"\nAI: {result2['messages'][-1].content}")
    else:
        print("\n  Action cancelled. File was NOT deleted.")
        print("  (Graph state preserved — could resume later if needed)")
