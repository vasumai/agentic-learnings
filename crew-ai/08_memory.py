"""
08_memory.py — Agent Memory: Short-Term, Long-Term, and Entity

Concepts covered:
- memory=True: one flag that enables all memory types on a Crew
- Short-term memory: agents remember context WITHIN a single run
- Long-term memory: agents remember learnings ACROSS runs (persisted to disk)
- Entity memory: agents track specific people, places, companies mentioned
- How to see memory working: run this script TWICE and watch the difference

Real-world analogy:
  Short-term = what you remember during a meeting
  Long-term   = what you wrote in your notebook after the meeting
  Entity      = your contact card for "Anthropic" that gets richer over time

LangGraph comparison:
  LangGraph: you manually manage memory via MemorySaver / SqliteSaver
             tied to thread_id — you control exactly what's stored
  CrewAI:    memory=True and the framework handles storage automatically
             Uses embeddings + vector search under the hood (via ChromaDB)
"""

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Memory types in CrewAI:
#
# SHORT-TERM (RAGStorage):
#   - Lives within a single crew run
#   - Agents can recall earlier task outputs even if not in direct context
#   - Uses in-memory embeddings — gone when the script ends
#
# LONG-TERM (LTMSQLiteStorage):
#   - Persists across multiple crew.kickoff() calls
#   - Stored in a local SQLite DB (~/.local/share/crewai/<crew_name>/long_term_memory.db)
#   - Agents recall facts from PREVIOUS runs automatically
#
# ENTITY (RAGStorage):
#   - Tracks named entities: people, companies, tools, concepts mentioned
#   - Builds a knowledge graph of entities and their attributes
#   - Gets richer with every run
#
# CONTEXTUAL:
#   - Combines the above for retrieval during agent reasoning
#   - Automatically injected into agent context when relevant
# ══════════════════════════════════════════════════════════════════════════════

# ── Agents ────────────────────────────────────────────────────────────────────

researcher = Agent(
    role="AI Industry Researcher",
    goal="Track developments in the AI industry and build cumulative knowledge over time",
    backstory=(
        "You are a dedicated AI industry tracker who studies companies, models, "
        "and trends. You build on everything you've learned in previous research sessions. "
        "When you recall something from past research, you explicitly say so."
    ),
    verbose=True,
)

analyst = Agent(
    role="Trend Analyst",
    goal="Identify patterns and trends from accumulated AI industry research",
    backstory=(
        "You are a pattern-recognition specialist who connects dots across "
        "multiple research sessions. You explicitly reference past findings "
        "when they are relevant to current analysis."
    ),
    verbose=True,
)

# ── Tasks ─────────────────────────────────────────────────────────────────────

research_task = Task(
    description=(
        "Research the following AI company/topic: {topic}\n\n"
        "Cover:\n"
        "- What they do and their key products\n"
        "- Recent notable announcements or releases\n"
        "- Key people involved (founders, executives)\n"
        "- How they fit into the broader AI landscape\n\n"
        "IMPORTANT: If you recall any related information from previous research "
        "sessions, explicitly reference it and connect it to the current topic."
    ),
    expected_output=(
        "A research brief covering: overview, key products, recent news, "
        "key people, and industry positioning. "
        "Include any connections to previously researched topics if relevant."
    ),
    agent=researcher,
)

analysis_task = Task(
    description=(
        "Based on the research just completed about {topic}, and drawing on "
        "any previously accumulated knowledge:\n\n"
        "1. Identify 2-3 patterns or trends you notice\n"
        "2. Compare with anything researched in previous sessions\n"
        "3. Flag any emerging themes across multiple companies/topics\n"
        "4. Give a 'knowledge state' summary: what do we now collectively know?"
    ),
    expected_output=(
        "A trend analysis with: current patterns, cross-session comparisons "
        "(if applicable), emerging themes, and a cumulative knowledge summary."
    ),
    agent=analyst,
    context=[research_task],
)

# ══════════════════════════════════════════════════════════════════════════════
# The key: memory=True
# This single flag enables all three memory systems simultaneously.
# CrewAI handles storage — you just flip the switch.
# ══════════════════════════════════════════════════════════════════════════════

crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    memory=True,          # ← enables short-term + long-term + entity memory
    verbose=True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
# EXPERIMENT: Run this script multiple times with different topics.
# On the first run — agents research just the current topic.
# On the second run — agents RECALL the first topic and connect the dots.
# On the third run — even richer cross-topic analysis emerges.

import sys

# Allow passing topic as command-line arg for easy re-runs
# Usage: python 08_memory.py "OpenAI"
#        python 08_memory.py "Google DeepMind"
#        python 08_memory.py "Meta AI"

if len(sys.argv) > 1:
    topic = " ".join(sys.argv[1:])
else:
    topic = "Anthropic — the AI safety company behind Claude"

print("\n" + "=" * 60)
print("MEMORY-ENABLED CREW")
print(f"Topic this run: {topic}")
print("=" * 60)
print("\nTip: Run this script multiple times with different topics:")
print("  python 08_memory.py 'Anthropic'")
print("  python 08_memory.py 'OpenAI'")
print("  python 08_memory.py 'Google DeepMind'")
print("\nWatch how the analysis gets richer with each run!\n")

result = crew.kickoff(inputs={"topic": topic})

print("\n" + "=" * 60)
print("ANALYSIS OUTPUT")
print("=" * 60)
print(result)

print("\n" + "=" * 60)
print("MEMORY PERSISTED")
print("=" * 60)
print("Long-term memory saved to disk.")
print("Run again with a different topic to see cross-run recall in action.")
print("\nTry next:")
next_topics = {
    "Anthropic — the AI safety company behind Claude": "python 08_memory.py 'OpenAI'",
    "OpenAI": "python 08_memory.py 'Google DeepMind'",
    "Google DeepMind": "python 08_memory.py 'Meta AI'",
}
print(f"  {next_topics.get(topic, 'python 08_memory.py \"OpenAI\"')}")
