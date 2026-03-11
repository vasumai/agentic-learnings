"""
04_task_dependencies.py — Explicit Task Dependencies

Concepts covered:
- context=[]: explicitly wiring which task outputs feed into which task
- Three-agent pipeline: Researcher → Analyst → Editor
- Why explicit context matters vs implicit sequential handoff
- Skipping intermediate output when a task only needs specific upstream results

Real-world analogy:
  A news team: Reporter gathers facts, Analyst adds interpretation,
  Editor polishes the final piece — but the Editor only needs the
  Analyst's interpretation, not the raw reporter notes.
"""

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# ── 1. Three specialized agents ──────────────────────────────────────────────

reporter = Agent(
    role="Investigative Reporter",
    goal="Gather raw facts and data about {topic}",
    backstory=(
        "You are a fact-obsessed journalist. You collect raw data, quotes, "
        "statistics, and timelines. You do not editorialize — just facts."
    ),
    verbose=True,
)

analyst = Agent(
    role="Business Analyst",
    goal="Interpret research findings about {topic} and extract business implications",
    backstory=(
        "You are a strategic analyst who reads raw research and identifies "
        "what it means for businesses and practitioners. "
        "You spot patterns, risks, and opportunities others miss."
    ),
    verbose=True,
)

editor = Agent(
    role="Senior Editor",
    goal="Polish the analyst's insights into a crisp executive summary about {topic}",
    backstory=(
        "You are a no-nonsense editor who cuts fluff and sharpens arguments. "
        "You take dense analysis and make it scannable for busy executives. "
        "You never add new facts — you only refine what's given to you."
    ),
    verbose=True,
)

# ── 2. Three tasks with explicit context wiring ──────────────────────────────

# Task 1: Reporter gathers raw facts (no dependencies — it's the starting point)
fact_gathering = Task(
    description=(
        "Gather raw facts about: {topic}\n\n"
        "Collect:\n"
        "- Key statistics and numbers\n"
        "- Timeline of major events\n"
        "- Names of key players / companies involved\n"
        "- Any controversies or open questions\n\n"
        "Output format: raw bullet points, no interpretation."
    ),
    expected_output=(
        "A raw fact sheet with 4 sections: Statistics, Timeline, "
        "Key Players, and Open Questions. Bullet points only."
    ),
    agent=reporter,
)

# Task 2: Analyst interprets the reporter's facts
# context=[fact_gathering] → explicitly says: feed fact_gathering output into this task
analysis = Task(
    description=(
        "Analyze the raw facts provided about {topic}.\n\n"
        "Produce:\n"
        "- Top 3 business implications\n"
        "- Biggest risk to watch\n"
        "- Best opportunity to act on\n"
        "- Your overall verdict in one sentence"
    ),
    expected_output=(
        "A structured analysis with: Business Implications (3 points), "
        "Key Risk, Key Opportunity, and a one-sentence verdict."
    ),
    agent=analyst,
    context=[fact_gathering],   # ← explicit: only pull from fact_gathering
)

# Task 3: Editor only needs the ANALYSIS, not the raw facts
# This is the key difference from lesson 03 — we choose exactly what flows in.
# If we used pure sequential, the editor would get BOTH fact_gathering + analysis.
# With context=[], we control it precisely.
executive_summary = Task(
    description=(
        "Transform the analyst's output into a crisp executive summary about {topic}.\n\n"
        "Requirements:\n"
        "- Max 200 words\n"
        "- One opening sentence that frames the topic\n"
        "- 3 bullet points: what, so what, now what\n"
        "- One closing recommendation\n"
        "- Tone: confident, direct, no jargon"
    ),
    expected_output=(
        "An executive summary under 200 words in markdown. "
        "Sections: Opening, Three Bullets (What / So What / Now What), Recommendation."
    ),
    agent=editor,
    context=[analysis],         # ← explicit: only pull from analysis, skip raw facts
    output_file="executive_summary.md",
)

# ── 3. Crew ──────────────────────────────────────────────────────────────────
# Even though we define explicit context, we still use Process.sequential
# so tasks execute in list order. The context=[] just controls what each
# task *sees*, not when it runs.

crew = Crew(
    agents=[reporter, analyst, editor],
    tasks=[fact_gathering, analysis, executive_summary],
    process=Process.sequential,
    verbose=True,
)

# ── 4. Kickoff ───────────────────────────────────────────────────────────────
topic = "The rise of agentic AI systems in enterprise software (2024-2025)"

print("\n" + "=" * 60)
print(f"PIPELINE STARTING — Topic: {topic}")
print("=" * 60 + "\n")

result = crew.kickoff(inputs={"topic": topic})

print("\n" + "=" * 60)
print("EXECUTIVE SUMMARY")
print("=" * 60)
print(result)
print("\n→ Saved to executive_summary.md")
