"""
05_multi_agent.py
-----------------
Real-world tasks often need specialized agents working together.
This file demonstrates a Supervisor pattern:
  - Supervisor: receives the user request and decides which agent to call
  - Researcher: looks up facts/information
  - Writer: takes research and produces polished content

Flow:
  User → Supervisor → Researcher → Supervisor → Writer → Supervisor → END

Key fix: 'completed' set in State prevents the supervisor (which uses an LLM
and can be non-deterministic) from re-running agents or skipping steps.
The supervisor's LLM decision is used as a hint, but completed steps always
override — once researcher + writer are done, we always go to FINISH.

Concepts introduced:
  multi-agent architecture, supervisor pattern, agent handoff,
  routing with Literal return types, shared state across agents,
  guarding non-deterministic LLM routing with explicit step tracking
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

load_dotenv()


# ── 1. Shared State ───────────────────────────────────────────────────────────
# 'completed' tracks which agents have already run — prevents re-running.
# This guards against non-deterministic LLM supervisor decisions.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    completed: list[str]   # ← new: tracks which agents have finished


# ── 2. LLM ────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ── 3. Supervisor Node ────────────────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are a supervisor managing two agents:
- researcher: Finds and summarizes factual information on a topic
- writer: Takes research notes and writes polished, well-structured content

Given the conversation history, decide who should act next.
Reply with ONLY one word: researcher, writer, or FINISH.

Rules:
- Start with researcher if we need information
- Move to writer once the researcher has provided research findings
- Reply FINISH once the writer has produced the final content
"""

def supervisor(state: State) -> dict:
    completed = state.get("completed", [])

    # Hard rules override LLM — prevents loops and skipped steps
    if "researcher" not in completed:
        next_agent = "researcher"
    elif "writer" not in completed:
        next_agent = "writer"
    else:
        next_agent = "FINISH"

    print(f"\n  [Supervisor] → routing to: {next_agent}  (completed: {completed})")
    return {"next": next_agent}


# ── 4. Researcher Node ────────────────────────────────────────────────────────
RESEARCHER_PROMPT = """You are a research agent. Your job is to find and summarize
key facts about the topic requested. Be thorough but concise.
Present your findings as structured bullet points.
Label your response clearly as: RESEARCH FINDINGS:"""

def researcher(state: State) -> dict:
    messages = [SystemMessage(content=RESEARCHER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    print(f"\n  [Researcher] working...")
    completed = state.get("completed", [])
    return {"messages": [response], "completed": completed + ["researcher"]}


# ── 5. Writer Node ────────────────────────────────────────────────────────────
WRITER_PROMPT = """You are a professional content writer. Your job is to take
the research findings and turn them into a well-structured, engaging piece of content.
Use clear headings, smooth transitions, and a professional tone.
Label your response clearly as: FINAL CONTENT:"""

def writer(state: State) -> dict:
    messages = [SystemMessage(content=WRITER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    print(f"\n  [Writer] working...")
    completed = state.get("completed", [])
    return {"messages": [response], "completed": completed + ["writer"]}


# ── 6. Routing function ───────────────────────────────────────────────────────
def route_next(state: State) -> Literal["researcher", "writer", "__end__"]:
    next_agent = state.get("next", "researcher")
    if next_agent == "FINISH":
        return "__end__"
    return next_agent


# ── 7. Build the Graph ────────────────────────────────────────────────────────
graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("researcher", researcher)
graph_builder.add_node("writer", writer)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", route_next)
graph_builder.add_edge("researcher", "supervisor")
graph_builder.add_edge("writer", "supervisor")

graph = graph_builder.compile()


# ── 8. Helper ─────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    """Handle both string and list-of-blocks content from the LLM."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(block.get("text", str(block)))
        else:
            parts.append(str(block))
    return " ".join(parts)


# ── 9. Run it ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    topics = [
        "Write a short article about the benefits of solar energy",
        "Write a brief overview of how Large Language Models work",
    ]

    for topic in topics:
        print("\n" + "=" * 60)
        print(f"Task: {topic}")
        print("=" * 60)

        result = graph.invoke({
            "messages": [HumanMessage(content=topic)],
            "next": "",
            "completed": []
        })

        # Search backwards for the last substantive AI message (>200 chars)
        # Guards against the model returning a near-empty writer response
        # when the researcher already produced well-formatted content
        final = ""
        for msg in reversed(result["messages"]):
            text = extract_text(msg.content)
            if len(text.strip()) > 200:
                final = text
                break
        print("\n" + (final if final else "[no content returned]"))
