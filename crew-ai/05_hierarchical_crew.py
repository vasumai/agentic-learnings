"""
05_hierarchical_crew.py — Hierarchical Crew with a Manager Agent

Concepts covered:
- Process.hierarchical: a Manager LLM delegates tasks to specialist agents
- You do NOT assign agents to tasks — the Manager decides who does what
- manager_llm: the LLM that drives delegation decisions
- When to use hierarchical vs sequential

Real-world analogy:
  You brief a Project Manager on the goal.
  The PM reads the tasks, assesses each specialist's role/skills,
  and assigns work — you don't micromanage who does what.

Key difference from sequential:
  Sequential (lessons 03, 04): YOU assign agent to each task explicitly
  Hierarchical (this lesson):  Manager Agent reads all agents + tasks
                               and decides assignments dynamically
"""

from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

load_dotenv()

# ── Manager LLM ───────────────────────────────────────────────────────────────
# CrewAI's built-in LLM wrapper — no need to import langchain_anthropic.
# The manager_llm is the "brain" of the crew — it reads the goal,
# reviews all available agents and tasks, and delegates intelligently.
# It does NOT need to be the same model as your worker agents.

manager_llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.3)

# ── 1. Specialist Agents — NO task assignment yet ─────────────────────────────
# Notice: we do NOT assign tasks to agents here.
# In hierarchical mode, the Manager reads these roles + backstories
# and decides who is best suited for each task.

market_researcher = Agent(
    role="Market Research Specialist",
    goal="Gather comprehensive market data, trends, and competitive landscape",
    backstory=(
        "You are a seasoned market researcher with expertise in identifying "
        "market trends, sizing opportunities, and profiling competitors. "
        "You produce structured, data-rich research briefs."
    ),
    verbose=True,
)

financial_analyst = Agent(
    role="Financial Analyst",
    goal="Analyze financial viability, cost structures, and revenue potential",
    backstory=(
        "You are a sharp financial analyst who evaluates business opportunities "
        "through the lens of unit economics, margins, and ROI. "
        "You translate complex financials into clear investment signals."
    ),
    verbose=True,
)

strategist = Agent(
    role="Business Strategist",
    goal="Synthesize research and financials into a concrete go-to-market strategy",
    backstory=(
        "You are an experienced business strategist who takes research and numbers "
        "and turns them into actionable plans. You think in terms of positioning, "
        "differentiation, and execution phases."
    ),
    verbose=True,
)

# ── 2. Tasks — described but NOT agent-assigned ───────────────────────────────
# In hierarchical mode, leave agent= unset (or the Manager will override it).
# The Manager reads each task description and assigns the best-fit agent.

market_research_task = Task(
    description=(
        "Conduct market research for: {business_idea}\n\n"
        "Cover:\n"
        "- Market size and growth rate\n"
        "- Top 3 competitors and their positioning\n"
        "- Target customer segments\n"
        "- Key market trends driving or threatening this space"
    ),
    expected_output=(
        "A structured market brief: Market Size, Competitors (3), "
        "Customer Segments, and Key Trends. Bullet points with numbers where possible."
    ),
)

financial_task = Task(
    description=(
        "Analyze the financial opportunity for: {business_idea}\n\n"
        "Estimate:\n"
        "- Likely revenue model (subscription, usage-based, license, etc.)\n"
        "- Key cost drivers (infrastructure, people, acquisition)\n"
        "- Rough unit economics (if applicable)\n"
        "- Break-even timeline estimate\n"
        "- Overall financial attractiveness (High / Medium / Low) with rationale"
    ),
    expected_output=(
        "A financial analysis covering: Revenue Model, Cost Drivers, "
        "Unit Economics, Break-even Estimate, and Attractiveness Rating with reasoning."
    ),
)

strategy_task = Task(
    description=(
        "Using the market research and financial analysis, develop a go-to-market "
        "strategy for: {business_idea}\n\n"
        "Include:\n"
        "- Recommended positioning (what makes this unique)\n"
        "- Primary target segment to win first\n"
        "- Top 3 go-to-market moves (Phase 1 actions)\n"
        "- Biggest risk and how to mitigate it\n"
        "- One-paragraph executive recommendation"
    ),
    expected_output=(
        "A go-to-market strategy document: Positioning, Target Segment, "
        "Top 3 GTM Moves, Key Risk + Mitigation, and Executive Recommendation."
    ),
    output_file="business_strategy.md",
)

# ── 3. Hierarchical Crew ──────────────────────────────────────────────────────
# Process.hierarchical activates the Manager agent.
# manager_llm: the LLM the Manager uses to reason about delegation.
# The Manager will:
#   1. Read all agent roles + backstories
#   2. Read all task descriptions
#   3. Assign tasks to the most suitable agents
#   4. Review outputs and ask for revisions if needed
#   5. Produce the final result

crew = Crew(
    agents=[market_researcher, financial_analyst, strategist],
    tasks=[market_research_task, financial_task, strategy_task],
    process=Process.hierarchical,
    manager_llm=manager_llm,
    verbose=True,
)

# ── 4. Kickoff ────────────────────────────────────────────────────────────────
business_idea = "An AI-powered code review tool for small engineering teams (2-10 devs)"

print("\n" + "=" * 60)
print("HIERARCHICAL CREW STARTING")
print(f"Business Idea: {business_idea}")
print("=" * 60)
print("\nWatch the Manager agent delegate tasks to specialists...\n")

result = crew.kickoff(inputs={"business_idea": business_idea})

print("\n" + "=" * 60)
print("FINAL STRATEGY OUTPUT")
print("=" * 60)
print(result)
print("\n→ Full strategy saved to business_strategy.md")
