"""
08_parallel_execution.py
------------------------
By default, LangGraph runs nodes one at a time.
Parallel execution lets you fan out to multiple nodes simultaneously,
then fan back in to aggregate the results.

Two patterns covered:

Pattern A — Static parallel branches:
  Multiple independent nodes wired to run at the same time.
  Use when you know upfront exactly how many things to run in parallel.

Pattern B — Dynamic fan-out with Send API:
  One node spawns N parallel workers at runtime based on input data.
  Use when the number of parallel tasks depends on the data (e.g. N documents).

Concepts introduced:
  static parallel edges, Send API, fan-out, fan-in,
  operator.add reducer for merging parallel results
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import time

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ── Helper ────────────────────────────────────────────────────────────────────
def ask_llm(system: str, user: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    content = response.content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN A: Static parallel branches
# Three independent analysis nodes run simultaneously, then merge.
# ══════════════════════════════════════════════════════════════════════════════
class AnalysisState(TypedDict):
    text: str
    sentiment: str
    keywords: str
    readability: str
    final_report: str


def analyze_sentiment(state: AnalysisState) -> dict:
    print("  [sentiment]   started")
    result = ask_llm(
        "Analyze the sentiment of this text. Reply in one sentence starting with 'Sentiment:'",
        state["text"]
    )
    print("  [sentiment]   done")
    return {"sentiment": result}


def analyze_keywords(state: AnalysisState) -> dict:
    print("  [keywords]    started")
    result = ask_llm(
        "Extract the 5 most important keywords. Reply as: 'Keywords: word1, word2, word3, word4, word5'",
        state["text"]
    )
    print("  [keywords]    done")
    return {"keywords": result}


def analyze_readability(state: AnalysisState) -> dict:
    print("  [readability] started")
    result = ask_llm(
        "Assess the readability level of this text (e.g. beginner/intermediate/expert). "
        "Reply in one sentence starting with 'Readability:'",
        state["text"]
    )
    print("  [readability] done")
    return {"readability": result}


def merge_analysis(state: AnalysisState) -> dict:
    """Fan-in node: combines results from all three parallel branches."""
    print("  [merge]       combining results")
    report = "\n".join([
        state["sentiment"],
        state["keywords"],
        state["readability"],
    ])
    return {"final_report": report}


def build_static_graph() -> object:
    builder = StateGraph(AnalysisState)

    builder.add_node("sentiment",    analyze_sentiment)
    builder.add_node("keywords",     analyze_keywords)
    builder.add_node("readability",  analyze_readability)
    builder.add_node("merge",        merge_analysis)

    # Fan-out: START triggers all three nodes simultaneously
    builder.add_edge(START, "sentiment")
    builder.add_edge(START, "keywords")
    builder.add_edge(START, "readability")

    # Fan-in: all three must complete before merge runs
    builder.add_edge("sentiment",   "merge")
    builder.add_edge("keywords",    "merge")
    builder.add_edge("readability", "merge")

    builder.add_edge("merge", END)
    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN B: Dynamic fan-out with Send API
# The number of parallel workers is determined at runtime from the data.
# ══════════════════════════════════════════════════════════════════════════════

class DocSummaryState(TypedDict):
    """State for each individual parallel worker."""
    document: str       # the document this worker handles
    summary: str        # produced by this worker


class AggregatorState(TypedDict):
    """Parent state that holds all documents and collects summaries."""
    documents: list[str]
    # operator.add reducer merges lists from parallel workers into one list
    summaries: Annotated[list[str], operator.add]
    final_summary: str


def fan_out(state: AggregatorState) -> list[Send]:
    """
    Returns a list of Send objects — one per document.
    Each Send spawns an independent 'summarize_doc' node with its own state.
    This is how LangGraph knows to run them in parallel.
    """
    print(f"  [fan_out] spawning {len(state['documents'])} parallel workers")
    return [
        Send("summarize_doc", {"document": doc, "summary": ""})
        for doc in state["documents"]
    ]


def summarize_doc(state: DocSummaryState) -> dict:
    """Worker node — runs in parallel for each document."""
    short = state["document"][:60].strip()
    print(f"  [worker] summarizing: '{short}...'")
    result = ask_llm(
        "Summarize this text in exactly one sentence.",
        state["document"]
    )
    return {"summaries": [result]}   # list — operator.add merges these


def combine_summaries(state: AggregatorState) -> dict:
    """Fan-in node — runs once all workers are done."""
    print(f"  [combine] merging {len(state['summaries'])} summaries")
    combined = ask_llm(
        "Combine these individual summaries into one cohesive paragraph.",
        "\n\n".join(f"- {s}" for s in state["summaries"])
    )
    return {"final_summary": combined}


def build_dynamic_graph() -> object:
    builder = StateGraph(AggregatorState)

    # fan_out is a conditional edge source — it returns Send objects
    builder.add_node("summarize_doc",     summarize_doc)
    builder.add_node("combine_summaries", combine_summaries)

    # START → fan_out function → N parallel summarize_doc nodes
    builder.add_conditional_edges(START, fan_out, ["summarize_doc"])

    # All workers → combine
    builder.add_edge("summarize_doc", "combine_summaries")
    builder.add_edge("combine_summaries", END)

    return builder.compile()


# ── Run both patterns ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Pattern A: Static parallel branches ──────────────────────────────────
    print("=" * 60)
    print("PATTERN A: Static parallel branches")
    print("=" * 60)

    sample_text = """
    Artificial intelligence is rapidly transforming industries worldwide.
    From healthcare diagnostics to autonomous vehicles, machine learning models
    are achieving unprecedented accuracy. However, concerns about job displacement,
    data privacy, and algorithmic bias remain significant challenges that policymakers
    and technologists must address collaboratively.
    """

    print(f"\nAnalyzing text (3 nodes run simultaneously):\n{sample_text.strip()[:100]}...\n")

    t0 = time.time()
    static_graph = build_static_graph()
    result_a = static_graph.invoke({
        "text": sample_text.strip(),
        "sentiment": "", "keywords": "", "readability": "", "final_report": ""
    })
    elapsed = time.time() - t0

    print(f"\nResults (completed in {elapsed:.1f}s):")
    print(result_a["final_report"])


    # ── Pattern B: Dynamic fan-out ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PATTERN B: Dynamic fan-out with Send API")
    print("=" * 60)

    documents = [
        "The Amazon rainforest produces 20% of the world's oxygen and is home to 10% of all species on Earth.",
        "Quantum computers use qubits that can exist in superposition, enabling them to solve certain problems exponentially faster than classical computers.",
        "The Mediterranean diet, rich in olive oil, fish, and vegetables, is associated with reduced risk of heart disease and longer lifespan.",
        "The James Webb Space Telescope can observe galaxies formed just 300 million years after the Big Bang, offering a window into the early universe.",
    ]

    print(f"\nSummarizing {len(documents)} documents in parallel...\n")

    t0 = time.time()
    dynamic_graph = build_dynamic_graph()
    result_b = dynamic_graph.invoke({
        "documents": documents,
        "summaries": [],
        "final_summary": ""
    })
    elapsed = time.time() - t0

    print(f"\nIndividual summaries ({elapsed:.1f}s total):")
    for i, s in enumerate(result_b["summaries"], 1):
        print(f"  {i}. {s}")

    print(f"\nCombined summary:")
    print(result_b["final_summary"])
