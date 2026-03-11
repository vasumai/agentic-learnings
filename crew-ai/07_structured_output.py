"""
07_structured_output.py — Typed, Validated Output with Pydantic

Concepts covered:
- output_pydantic=: task returns a validated Pydantic model, not raw text
- output_json=: lighter alternative that returns a dict
- Why structured output matters for real applications
- Accessing individual fields from task results programmatically
- Chaining structured output: one task's model feeds the next

Real-world analogy:
  Lessons 01-06: agents returned essays (free text).
  This lesson: agents return structured data — like a form filled out correctly.
  Your downstream code can now do result.price, result.risk_level, result.tags
  instead of parsing paragraphs of text.
"""

from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Define your Pydantic output models
# These are the "forms" agents must fill out correctly.
# Field() lets you add descriptions — the LLM uses these as guidance.
# ══════════════════════════════════════════════════════════════════════════════

class TechCompany(BaseModel):
    """Structured profile of a technology company."""
    name: str = Field(description="Full company name")
    founded_year: int = Field(description="Year the company was founded")
    headquarters: str = Field(description="City and country of headquarters")
    core_product: str = Field(description="Primary product or service in one sentence")
    employee_count_estimate: str = Field(description="Rough employee count (e.g. '10,000-50,000')")
    key_competitors: List[str] = Field(description="List of 3 main competitors by name")
    strengths: List[str] = Field(description="List of 3 key competitive strengths")
    risks: List[str] = Field(description="List of 2 key business risks")
    analyst_rating: str = Field(description="Overall rating: Strong Buy / Buy / Hold / Sell")
    one_line_summary: str = Field(description="One punchy sentence summarizing the company")


class InvestmentThesis(BaseModel):
    """A structured investment thesis for a company."""
    company_name: str = Field(description="Name of the company being analyzed")
    investment_score: int = Field(description="Score from 1-10 (10 = strongest buy)")
    time_horizon: str = Field(description="Recommended holding period: Short / Medium / Long term")
    bull_case: str = Field(description="Best case scenario in 2 sentences")
    bear_case: str = Field(description="Worst case scenario in 2 sentences")
    key_metrics_to_watch: List[str] = Field(description="3 metrics investors should monitor")
    verdict: str = Field(description="Final verdict in one clear sentence")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Agents — same as before, nothing special needed on the agent
# ══════════════════════════════════════════════════════════════════════════════

research_agent = Agent(
    role="Technology Company Analyst",
    goal="Produce accurate, structured company profiles from research",
    backstory=(
        "You are a meticulous analyst who always fills out structured reports "
        "completely and accurately. You never skip fields or leave them vague. "
        "Every field in your report is specific and fact-based."
    ),
    verbose=True,
)

investment_agent = Agent(
    role="Investment Strategist",
    goal="Translate company research into structured investment theses",
    backstory=(
        "You are a sharp investment strategist who reads company profiles "
        "and produces clear, opinionated investment theses. "
        "You give concrete scores and verdicts — never wishy-washy answers."
    ),
    verbose=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Tasks with output_pydantic=
# The agent MUST return data matching the model schema.
# CrewAI validates the output and gives you a proper Python object.
# ══════════════════════════════════════════════════════════════════════════════

profile_task = Task(
    description=(
        "Research and produce a structured company profile for: {company}\n\n"
        "Fill out every field accurately. Be specific with numbers and names. "
        "For analyst_rating, use exactly one of: Strong Buy / Buy / Hold / Sell"
    ),
    expected_output=(
        "A complete, accurate company profile with all fields populated. "
        "key_competitors, strengths, and risks must each be proper lists."
    ),
    agent=research_agent,
    output_pydantic=TechCompany,   # ← magic: return a TechCompany object
)

thesis_task = Task(
    description=(
        "Using the company profile provided, write a structured investment thesis.\n\n"
        "investment_score must be an integer 1-10. "
        "time_horizon must be exactly: Short / Medium / Long term. "
        "Be opinionated — no fence-sitting."
    ),
    expected_output=(
        "A complete investment thesis with all fields populated. "
        "investment_score is a number, not a string."
    ),
    agent=investment_agent,
    context=[profile_task],
    output_pydantic=InvestmentThesis,  # ← returns an InvestmentThesis object
)

# ── Crew ──────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[research_agent, investment_agent],
    tasks=[profile_task, thesis_task],
    process=Process.sequential,
    verbose=True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
company = "Anthropic"


print("\n" + "=" * 60)
print(f"STRUCTURED ANALYSIS: {company}")
print("=" * 60 + "\n")

crew_result = crew.kickoff(inputs={"company": company})

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Access structured fields — this is why Pydantic output matters
# crew_result.pydantic gives you the last task's Pydantic object
# For individual tasks: profile_task.output.pydantic
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("COMPANY PROFILE (structured fields)")
print("=" * 60)

profile: TechCompany = profile_task.output.pydantic
print(f"Company:        {profile.name}")
print(f"Founded:        {profile.founded_year}")
print(f"HQ:             {profile.headquarters}")
print(f"Core Product:   {profile.core_product}")
print(f"Employees:      {profile.employee_count_estimate}")
print(f"Competitors:    {', '.join(profile.key_competitors)}")
print(f"Strengths:")
for s in profile.strengths:
    print(f"  • {s}")
print(f"Risks:")
for r in profile.risks:
    print(f"  • {r}")
print(f"Rating:         {profile.analyst_rating}")
print(f"Summary:        {profile.one_line_summary}")

print("\n" + "=" * 60)
print("INVESTMENT THESIS (structured fields)")
print("=" * 60)

thesis: InvestmentThesis = crew_result.pydantic  # last task's output
print(f"Company:        {thesis.company_name}")
print(f"Score:          {thesis.investment_score}/10")
print(f"Time Horizon:   {thesis.time_horizon}")
print(f"Bull Case:      {thesis.bull_case}")
print(f"Bear Case:      {thesis.bear_case}")
print(f"Watch Metrics:  {', '.join(thesis.key_metrics_to_watch)}")
print(f"Verdict:        {thesis.verdict}")

# Show that you can use this data programmatically
print("\n" + "=" * 60)
print("PROGRAMMATIC USE (what makes structured output powerful)")
print("=" * 60)
if thesis.investment_score >= 8:
    print(f"ACTION: Strong signal — {thesis.company_name} scores {thesis.investment_score}/10")
elif thesis.investment_score >= 5:
    print(f"ACTION: Monitor — {thesis.company_name} scores {thesis.investment_score}/10")
else:
    print(f"ACTION: Pass — {thesis.company_name} scores {thesis.investment_score}/10")
