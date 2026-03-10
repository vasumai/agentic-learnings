"""
07_subgraphs.py
---------------
As agents grow complex, a single flat graph becomes hard to manage.
Subgraphs let you build modular, reusable graph components and compose
them into a parent graph — like functions calling other functions.

Real-world use case:
  A document processing pipeline where each stage (extract, summarize,
  classify) is a self-contained subgraph that can be tested independently
  and reused across different parent workflows.

Architecture:
  Parent Graph
  ├── extract_subgraph    → pulls key info from raw text
  ├── summarize_subgraph  → generates a concise summary
  └── classify_subgraph   → assigns a category and priority

Key rule: subgraphs communicate with the parent ONLY through shared
state keys. Each subgraph can have private internal state keys too.

Concepts introduced:
  subgraph composition, StateGraph.compile() as a node,
  shared vs private state, modular agent design
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ── Helper ────────────────────────────────────────────────────────────────────
def ask_llm(system: str, user: str) -> str:
    """Simple single-turn LLM call, returns plain text."""
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    content = response.content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


# ══════════════════════════════════════════════════════════════════════════════
# SUBGRAPH 1: Extraction
# Shared state key: "raw_text" (input), "extracted" (output)
# Private state key: "extraction_prompt" (only used inside this subgraph)
# ══════════════════════════════════════════════════════════════════════════════
class ExtractState(TypedDict):
    raw_text: str       # shared with parent
    extracted: str      # shared with parent
    _format: str        # private — only used inside this subgraph


def build_extract_prompt(state: ExtractState) -> dict:
    """Private node: constructs the extraction prompt."""
    prompt = "Extract key facts as a numbered list. Be concise."
    return {"_format": prompt}


def run_extraction(state: ExtractState) -> dict:
    print("  [extract_subgraph] extracting key facts...")
    result = ask_llm(state["_format"], state["raw_text"])
    return {"extracted": result}


extract_builder = StateGraph(ExtractState)
extract_builder.add_node("build_prompt", build_extract_prompt)
extract_builder.add_node("extract", run_extraction)
extract_builder.add_edge(START, "build_prompt")
extract_builder.add_edge("build_prompt", "extract")
extract_builder.add_edge("extract", END)
extract_subgraph = extract_builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SUBGRAPH 2: Summarization
# Shared state keys: "extracted" (input), "summary" (output)
# ══════════════════════════════════════════════════════════════════════════════
class SummarizeState(TypedDict):
    extracted: str      # shared with parent
    summary: str        # shared with parent


def run_summarization(state: SummarizeState) -> dict:
    print("  [summarize_subgraph] writing summary...")
    result = ask_llm(
        "Write a clear 2-3 sentence summary based on the provided key facts.",
        state["extracted"]
    )
    return {"summary": result}


summarize_builder = StateGraph(SummarizeState)
summarize_builder.add_node("summarize", run_summarization)
summarize_builder.add_edge(START, "summarize")
summarize_builder.add_edge("summarize", END)
summarize_subgraph = summarize_builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SUBGRAPH 3: Classification
# Shared state keys: "summary" (input), "category" + "priority" (output)
# ══════════════════════════════════════════════════════════════════════════════
class ClassifyState(TypedDict):
    summary: str        # shared with parent
    category: str       # shared with parent
    priority: str       # shared with parent


def run_classification(state: ClassifyState) -> dict:
    print("  [classify_subgraph] classifying document...")
    result = ask_llm(
        """Classify this document. Reply in exactly this format (no extra text):
CATEGORY: <one of: Technology, Business, Science, Health, Other>
PRIORITY: <one of: High, Medium, Low>""",
        state["summary"]
    )
    # Parse the structured response
    category = "Other"
    priority = "Medium"
    for line in result.splitlines():
        if line.startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip()
        elif line.startswith("PRIORITY:"):
            priority = line.split(":", 1)[1].strip()
    return {"category": category, "priority": priority}


classify_builder = StateGraph(ClassifyState)
classify_builder.add_node("classify", run_classification)
classify_builder.add_edge(START, "classify")
classify_builder.add_edge("classify", END)
classify_subgraph = classify_builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PARENT GRAPH
# State must include ALL keys used by any subgraph
# ══════════════════════════════════════════════════════════════════════════════
class ParentState(TypedDict):
    raw_text: str       # input
    extracted: str      # produced by extract_subgraph
    summary: str        # produced by summarize_subgraph
    category: str       # produced by classify_subgraph
    priority: str       # produced by classify_subgraph
    _format: str        # required because ExtractState has it


parent_builder = StateGraph(ParentState)

# Each compiled subgraph becomes a node in the parent graph
parent_builder.add_node("extract",    extract_subgraph)
parent_builder.add_node("summarize",  summarize_subgraph)
parent_builder.add_node("classify",   classify_subgraph)

parent_builder.add_edge(START,      "extract")
parent_builder.add_edge("extract",  "summarize")
parent_builder.add_edge("summarize","classify")
parent_builder.add_edge("classify", END)

parent_graph = parent_builder.compile()


# ── Run it ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    documents = [
        """
        Apple Inc. announced today that it has achieved record quarterly revenue of $120 billion,
        driven primarily by strong iPhone 15 sales and rapid growth in its Services division.
        CEO Tim Cook noted that AI features in iOS 18 contributed significantly to higher upgrade
        rates among existing customers. The company also announced a $90 billion share buyback program.
        """,
        """
        Researchers at MIT have developed a new mRNA vaccine candidate that shows 94% efficacy
        against multiple strains of influenza in Phase 2 clinical trials. Unlike traditional flu shots,
        this vaccine targets a conserved region of the virus, potentially eliminating the need for
        annual reformulation. The study, published in Nature Medicine, involved 3,200 participants
        across 12 countries.
        """,
    ]

    for i, doc in enumerate(documents, 1):
        print(f"\n{'=' * 60}")
        print(f"Document {i}")
        print('=' * 60)
        print(f"Input: {doc.strip()[:100]}...")

        result = parent_graph.invoke({
            "raw_text": doc.strip(),
            "extracted": "",
            "summary": "",
            "category": "",
            "priority": "",
            "_format": "",
        })

        print(f"\n  Extracted facts:\n{result['extracted']}")
        print(f"\n  Summary: {result['summary']}")
        print(f"\n  Category: {result['category']}  |  Priority: {result['priority']}")
