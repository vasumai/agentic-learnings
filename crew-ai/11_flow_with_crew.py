"""
11_flow_with_crew.py — Flow Orchestrating Multiple Crews

Concepts covered:
- A Flow where each @listen step runs a dedicated Crew
- How to pass Flow state INTO a Crew via kickoff(inputs={})
- How to pull Crew output BACK into Flow state
- Conditional branching between Crews based on state
- The "Flow as conductor, Crew as orchestra section" pattern

This is the most powerful CrewAI pattern — and the closest to
what you'd use in a real production system.

Architecture:
  Flow manages the WHAT and WHEN (orchestration logic, state, branching)
  Crews manage the HOW (agents doing specialist work)

  Think of it like a software project:
    Flow = the project plan with milestones and gates
    Each Crew = a sprint team that handles one milestone

LangGraph equivalent:
  This pattern is similar to LangGraph subgraphs — composing multiple
  graphs into one larger workflow. But here the "subgraphs" are Crews
  (natural language teams) rather than compiled state machines.
"""

from crewai.flow.flow import Flow, listen, start, router
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Shared Flow State — flows through every stage
# ══════════════════════════════════════════════════════════════════════════════

class ProductLaunchState(BaseModel):
    # Input
    product_name: str = ""
    product_description: str = ""
    target_market: str = ""

    # Stage 1 outputs — Market Research Crew
    market_research: str = ""
    market_viable: bool = False       # gate: proceed or stop

    # Stage 2 outputs — Positioning Crew
    positioning_statement: str = ""
    key_messages: str = ""

    # Stage 3 outputs — Launch Plan Crew
    launch_plan: str = ""

    # Stage 4 outputs — Review Crew
    review_verdict: str = ""          # "approved" or "revise"
    review_notes: str = ""

    # Final
    final_deliverable: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build agents inline — keeps crew definitions compact
# ══════════════════════════════════════════════════════════════════════════════

def make_agent(role, goal, backstory):
    return Agent(role=role, goal=goal, backstory=backstory, verbose=False)

def make_task(description, expected_output, agent, context=None):
    kwargs = dict(description=description, expected_output=expected_output, agent=agent)
    if context:
        kwargs["context"] = context
    return Task(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# The Flow — each step runs a focused Crew, passes results into state
# ══════════════════════════════════════════════════════════════════════════════

class ProductLaunchFlow(Flow[ProductLaunchState]):

    # ── Stage 1: Market Research Crew ─────────────────────────────────────────
    @start()
    def run_market_research(self):
        print("\n[FLOW] Stage 1: Market Research Crew starting...")

        researcher = make_agent(
            role="Market Research Analyst",
            goal=f"Assess market opportunity for {self.state.product_name}",
            backstory="You are a sharp market analyst who spots real opportunities and dismisses hype.",
        )
        viability_agent = make_agent(
            role="Market Viability Assessor",
            goal="Give a clear go/no-go recommendation based on research",
            backstory="You make hard calls. Your job is to protect the team from launching into bad markets.",
        )

        research_task = make_task(
            description=(
                f"Research the market for: {self.state.product_name}\n"
                f"Description: {self.state.product_description}\n"
                f"Target market: {self.state.target_market}\n\n"
                "Assess: market size, key competitors, customer pain points, timing."
            ),
            expected_output="A concise market research brief: size, competitors, pain points, timing verdict.",
            agent=researcher,
        )
        viability_task = make_task(
            description=(
                "Based on the market research, give a GO or NO-GO verdict.\n"
                "Respond in EXACTLY this format:\n"
                "VERDICT: GO or NO-GO\n"
                "REASON: one sentence\n"
                "RISK: one sentence"
            ),
            expected_output="VERDICT: GO or NO-GO\nREASON: ...\nRISK: ...",
            agent=viability_agent,
            context=[research_task],
        )

        crew = Crew(
            agents=[researcher, viability_agent],
            tasks=[research_task, viability_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        # Pull both task outputs into state
        self.state.market_research = str(research_task.output.raw)
        viability_output = str(result)
        self.state.market_viable = "VERDICT: GO" in viability_output

        print(f"[FLOW] Market Research complete — Viable: {self.state.market_viable}")

    # ── Gate: route after market research ────────────────────────────────────
    @router(run_market_research)
    def check_market_viability(self):
        if self.state.market_viable:
            print("[FLOW] Market viable — proceeding to positioning")
            return "viable"
        else:
            print("[FLOW] Market NOT viable — stopping pipeline")
            return "not_viable"

    @listen("not_viable")
    def handle_no_go(self):
        print("\n[FLOW] Pipeline stopped: market not viable.")
        self.state.final_deliverable = (
            "PIPELINE STOPPED — Market research returned NO-GO.\n\n"
            f"Research findings:\n{self.state.market_research}"
        )

    # ── Stage 2: Positioning Crew ─────────────────────────────────────────────
    @listen("viable")
    def run_positioning(self):
        print("\n[FLOW] Stage 2: Positioning Crew starting...")

        strategist = make_agent(
            role="Brand Strategist",
            goal=f"Craft the positioning for {self.state.product_name}",
            backstory="You create razor-sharp positioning that makes products stand out in crowded markets.",
        )
        copywriter = make_agent(
            role="Messaging Copywriter",
            goal="Turn positioning strategy into concrete key messages",
            backstory="You translate strategy into words that land. Every message you write is memorable.",
        )

        positioning_task = make_task(
            description=(
                f"Create positioning for {self.state.product_name}.\n"
                f"Market research context:\n{self.state.market_research[:600]}\n\n"
                "Produce:\n"
                "- One-line positioning statement (who it's for, what it does, why it's different)\n"
                "- Three differentiators vs competitors"
            ),
            expected_output="Positioning statement + 3 differentiators.",
            agent=strategist,
        )
        messaging_task = make_task(
            description=(
                "Convert the positioning into 3 key messages for the launch.\n"
                "Each message: one punchy sentence targeted at the buyer's pain point."
            ),
            expected_output="3 key messages, one per line, each 1 sentence.",
            agent=copywriter,
            context=[positioning_task],
        )

        result = Crew(
            agents=[strategist, copywriter],
            tasks=[positioning_task, messaging_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        self.state.positioning_statement = str(positioning_task.output.raw)
        self.state.key_messages = str(result)
        print("[FLOW] Positioning complete")

    # ── Stage 3: Launch Plan Crew ─────────────────────────────────────────────
    @listen(run_positioning)
    def run_launch_planning(self):
        print("\n[FLOW] Stage 3: Launch Planning Crew starting...")

        planner = make_agent(
            role="Go-to-Market Planner",
            goal=f"Build a concrete launch plan for {self.state.product_name}",
            backstory="You build launch plans that are realistic, sequenced, and measurable.",
        )
        ops = make_agent(
            role="Launch Operations Specialist",
            goal="Add execution details and owners to the launch plan",
            backstory="You turn high-level plans into actionable checklists with clear owners and timelines.",
        )

        plan_task = make_task(
            description=(
                f"Build a 30-day launch plan for {self.state.product_name}.\n\n"
                f"Positioning: {self.state.positioning_statement[:400]}\n"
                f"Key messages: {self.state.key_messages[:300]}\n\n"
                "Structure: Week 1 (pre-launch), Week 2 (launch), Week 3-4 (post-launch).\n"
                "Include: 3-5 activities per week, each with a goal."
            ),
            expected_output="A 30-day launch plan organized by week with activities and goals.",
            agent=planner,
        )
        ops_task = make_task(
            description=(
                "Add execution details to each activity in the launch plan:\n"
                "- Owner role (e.g. Marketing, Engineering, Sales)\n"
                "- Success metric\n"
                "Keep it concise — one line per activity."
            ),
            expected_output="The launch plan with owner and metric added to each activity.",
            agent=ops,
            context=[plan_task],
        )

        result = Crew(
            agents=[planner, ops],
            tasks=[plan_task, ops_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        self.state.launch_plan = str(result)
        print("[FLOW] Launch plan complete")

    # ── Stage 4: Final Review Crew ────────────────────────────────────────────
    @listen(run_launch_planning)
    def run_final_review(self):
        print("\n[FLOW] Stage 4: Final Review Crew starting...")

        reviewer = make_agent(
            role="Launch Director",
            goal="Review the complete launch package and approve or request revisions",
            backstory=(
                "You are a seasoned launch director who has shipped 50+ products. "
                "You approve only when everything is tight, realistic, and coherent."
            ),
        )

        review_task = make_task(
            description=(
                f"Review the complete launch package for {self.state.product_name}.\n\n"
                f"POSITIONING:\n{self.state.positioning_statement[:400]}\n\n"
                f"KEY MESSAGES:\n{self.state.key_messages[:300]}\n\n"
                f"LAUNCH PLAN (excerpt):\n{self.state.launch_plan[:500]}\n\n"
                "Evaluate: coherence, realism, completeness.\n"
                "Respond in EXACTLY this format:\n"
                "VERDICT: APPROVED or REVISE\n"
                "NOTES: two sentences max"
            ),
            expected_output="VERDICT: APPROVED or REVISE\nNOTES: ...",
            agent=reviewer,
        )

        result = Crew(
            agents=[reviewer],
            tasks=[review_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        output = str(result)
        self.state.review_verdict = "approved" if "VERDICT: APPROVED" in output else "revise"
        self.state.review_notes = output
        print(f"[FLOW] Review verdict: {self.state.review_verdict.upper()}")

    # ── Final gate ────────────────────────────────────────────────────────────
    @router(run_final_review)
    def route_on_review(self):
        return self.state.review_verdict  # "approved" or "revise"

    @listen("approved")
    def compile_final_deliverable(self):
        print("\n[FLOW] Compiling final launch package...")
        self.state.final_deliverable = (
            f"# {self.state.product_name} — Launch Package\n\n"
            f"## Positioning\n{self.state.positioning_statement}\n\n"
            f"## Key Messages\n{self.state.key_messages}\n\n"
            f"## 30-Day Launch Plan\n{self.state.launch_plan}\n\n"
            f"---\n*Approved by Launch Director*\n{self.state.review_notes}"
        )
        print("[FLOW] Final deliverable compiled — launch package ready!")

    @listen("revise")
    def handle_revision_needed(self):
        print("\n[FLOW] Revision requested by Launch Director.")
        self.state.final_deliverable = (
            f"# {self.state.product_name} — REVISION NEEDED\n\n"
            f"## Review Notes\n{self.state.review_notes}\n\n"
            f"## Current Launch Plan (needs revision)\n{self.state.launch_plan}"
        )


# ── Run the Flow ──────────────────────────────────────────────────────────────

flow = ProductLaunchFlow()

print("\n" + "=" * 60)
print("PRODUCT LAUNCH FLOW — 4 CREWS, 1 FLOW")
print("=" * 60)
print("\nStages:")
print("  1. Market Research Crew  → viability gate")
print("  2. Positioning Crew      → (if viable)")
print("  3. Launch Planning Crew  → (after positioning)")
print("  4. Final Review Crew     → approve or revise")
print()

result = flow.kickoff(inputs={
    "product_name": "CodePilot",
    "product_description": "An AI pair programmer that reviews PRs, suggests refactors, and explains legacy code",
    "target_market": "Mid-size engineering teams (20-200 devs) at B2B SaaS companies",
})

# Final state summary
print("\n" + "=" * 60)
print("FLOW COMPLETE — FINAL STATE SUMMARY")
print("=" * 60)
print(f"Product:        {flow.state.product_name}")
print(f"Market Viable:  {flow.state.market_viable}")
print(f"Review Verdict: {flow.state.review_verdict.upper() if flow.state.review_verdict else 'N/A'}")
print("\n--- FINAL DELIVERABLE ---")
print(flow.state.final_deliverable[:2000])

# Save to file
with open("launch_package.md", "w") as f:
    f.write(flow.state.final_deliverable)
print("\n→ Full launch package saved to launch_package.md")
