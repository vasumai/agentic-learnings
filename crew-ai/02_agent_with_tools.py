"""
02_agent_with_tools.py — Agent with Built-in Tools

Concepts covered:
- Giving an agent tools (SerperDevTool for web search)
- How the agent decides WHEN to use a tool (ReAct loop)
- FileWriterTool to save output to disk
- Using {placeholders} in task descriptions with crew.kickoff(inputs={})

Setup: You need a free Serper API key for web search.
  1. Sign up at https://serper.dev (free tier = 2500 searches)
  2. Add to .env:  SERPER_API_KEY=your_key_here

  Alternatively, if you don't have Serper, we use the fallback agent
  that uses only the LLM's built-in knowledge (no web search).
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool
from dotenv import load_dotenv

load_dotenv()

# ── Check if Serper key is available ────────────────────────────────────────
has_serper = bool(os.getenv("SERPER_API_KEY"))
tools = []

if has_serper:
    # SerperDevTool: gives the agent Google search capability
    # The agent autonomously decides when to call it based on the task
    search_tool = SerperDevTool()
    tools.append(search_tool)
    print("✓ Web search enabled (SerperDevTool)")
else:
    print("⚠ No SERPER_API_KEY found — agent will use built-in knowledge only")

# FileWriterTool: lets the agent write its output to a file
file_writer = FileWriterTool()
tools.append(file_writer)
print("✓ File writer enabled")

# ── 1. Agent with tools ──────────────────────────────────────────────────────
# Tools are passed as a list — the agent picks which tool to use and when.
# The LLM sees tool descriptions and decides autonomously (ReAct pattern).

researcher = Agent(
    role="Technology Analyst",
    goal="Research {topic} thoroughly and produce a well-structured report",
    backstory=(
        "You are a sharp technology analyst who always searches for the latest "
        "information before drawing conclusions. You never guess — you verify. "
        "Your reports are concise, factual, and easy to scan."
    ),
    tools=tools,
    verbose=True,
)

# ── 2. Task using {placeholders} ────────────────────────────────────────────
# {topic} is a placeholder — its value is injected at kickoff() time.
# This makes tasks reusable across different inputs.

research_task = Task(
    description=(
        "Research the topic: {topic}\n\n"
        "Steps:\n"
        "1. Search for recent information about {topic}\n"
        "2. Identify the top 3 key points\n"
        "3. Note any recent developments (last 6 months if possible)\n"
        "4. Write the report to a file named 'report.md'"
    ),
    expected_output=(
        "A markdown report saved to 'report.md' containing:\n"
        "- Title\n"
        "- 3 key points (each 2-3 sentences)\n"
        "- Recent developments section\n"
        "- A one-line conclusion"
    ),
    agent=researcher,
    output_file="report.md",   # CrewAI also auto-saves the final answer here
)

# ── 3. Crew ──────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True,
)

# ── 4. Kickoff with dynamic inputs ───────────────────────────────────────────
# inputs dict replaces {placeholders} in all task descriptions and agent goals
topic = "CrewAI framework — latest features and real-world use cases"

print("\n" + "=" * 60)
print(f"RESEARCHING: {topic}")
print("=" * 60 + "\n")

result = crew.kickoff(inputs={"topic": topic})

print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(result)
print("\n→ Check report.md for the saved file output")
