"""
01_simple_graph.py
------------------
The most basic LangGraph pattern:
  - Define a typed State
  - Add one node that calls an LLM
  - Compile and run the graph

Concepts introduced:
  StateGraph, TypedDict state, add_node, add_edge, compile, invoke
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages

load_dotenv()

# ── 1. Define the State ──────────────────────────────────────────────────────
# State is a TypedDict shared across all nodes.
# `add_messages` is a reducer: it appends new messages instead of overwriting.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. Create the LLM ────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ── 3. Define a Node ─────────────────────────────────────────────────────────
# A node is just a function: State in → dict (partial state update) out.
def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages appends this to the list


# ── 4. Build the Graph ───────────────────────────────────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

# START → chatbot → END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


# ── 5. Run it ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_input = "What is LangGraph and why would I use it over plain LangChain?"
    print(f"User: {user_input}\n")

    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    # result["messages"] is the full list; last message is the AI response
    print(f"AI: {result['messages'][-1].content}")
