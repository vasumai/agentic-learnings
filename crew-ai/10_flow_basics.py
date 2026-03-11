"""
10_flow_basics.py — CrewAI Flows: Event-Driven Orchestration

Concepts covered:
- Flow: a class-based orchestrator with typed shared state
- @start: the entry point method of a Flow
- @listen: a method that triggers when another method completes
- FlowState (BaseModel): structured state shared across all Flow methods
- @router: conditional branching based on state values
- flow.kickoff(): runs the flow from @start through all listeners

The conceptual leap from Crew to Flow:
  Crew  = "hire agents, assign tasks, run them"  → good for linear pipelines
  Flow  = "define steps, react to completions, branch on conditions"
        → good for multi-stage workflows with logic between stages

Real-world analogy:
  A Crew is a team meeting where tasks are handed out.
  A Flow is a business process: step A finishes → triggers step B or C
  depending on the outcome. Like a flowchart that actually runs.

LangGraph comparison:
  LangGraph: StateGraph + nodes + edges — you draw the graph explicitly
  CrewAI Flow: @start/@listen decorators — you declare reactions to events
  Both achieve event-driven orchestration; syntax and mental model differ.
"""

from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel
from dotenv import load_dotenv
import time

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Define FlowState — the shared state across ALL methods in the Flow
# This is analogous to StateGraph's TypedDict state in LangGraph.
# Every method in the Flow can read AND write to this state.
# ══════════════════════════════════════════════════════════════════════════════

class ContentPipelineState(BaseModel):
    # Inputs
    topic: str = ""
    target_audience: str = ""

    # Intermediate outputs (populated as flow progresses)
    outline: str = ""
    draft: str = ""
    seo_keywords: list[str] = []
    quality_score: int = 0
    quality_verdict: str = ""   # "pass" or "revise"

    # Final output
    final_article: str = ""
    publish_ready: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Define the Flow class
# Each method decorated with @start or @listen is a "step" in the flow.
# Methods can be sync or async — CrewAI handles both.
# ══════════════════════════════════════════════════════════════════════════════

class ContentCreationFlow(Flow[ContentPipelineState]):
    """
    A content creation pipeline as a Flow.
    Steps: plan → draft (+ SEO in parallel) → quality check → route → finalize or revise
    """

    # ── Step 1: @start — the entry point ─────────────────────────────────────
    # @start marks this as the first method to run when flow.kickoff() is called.
    # There can be only ONE @start method.
    # It receives no arguments (state is already initialized from kickoff inputs).

    @start()
    def plan_content(self):
        print("\n[FLOW] Step 1: Planning content outline...")

        # In a real flow, this would call an Agent/Crew.
        # Here we simulate the logic clearly so you see the Flow structure.
        from crewai import Agent, Task, Crew, Process

        planner = Agent(
            role="Content Strategist",
            goal=f"Create a detailed outline for an article about: {self.state.topic}",
            backstory="You are an expert content strategist who creates clear, logical article outlines.",
            verbose=False,
        )

        plan_task = Task(
            description=(
                f"Create a structured article outline for the topic: '{self.state.topic}'\n"
                f"Target audience: {self.state.target_audience}\n\n"
                "Produce:\n"
                "- Article title\n"
                "- 4-5 section headings with one-line descriptions\n"
                "- Key angle/hook for the introduction"
            ),
            expected_output="A structured article outline with title, sections, and intro hook.",
            agent=planner,
        )

        result = Crew(agents=[planner], tasks=[plan_task], verbose=False).kickoff()
        self.state.outline = str(result)
        print(f"[FLOW] Outline complete ({len(self.state.outline)} chars)")

    # ── Step 2: @listen — reacts to plan_content completing ──────────────────
    # @listen(plan_content) means: "run this method as soon as plan_content finishes"
    # The method receives the return value of plan_content (None here, since we
    # write to self.state directly — which is the preferred pattern).

    @listen(plan_content)
    def write_draft(self):
        print("\n[FLOW] Step 2: Writing article draft...")

        from crewai import Agent, Task, Crew

        writer = Agent(
            role="Content Writer",
            goal="Write a compelling article based on the provided outline",
            backstory="You write engaging, well-structured articles that educate and inspire.",
            verbose=False,
        )

        write_task = Task(
            description=(
                f"Write a full article based on this outline:\n\n{self.state.outline}\n\n"
                f"Topic: {self.state.topic}\n"
                f"Audience: {self.state.target_audience}\n"
                "Requirements: 400-600 words, conversational tone, concrete examples."
            ),
            expected_output="A complete article in markdown with title, intro, body sections, and conclusion.",
            agent=writer,
        )

        result = Crew(agents=[writer], tasks=[write_task], verbose=False).kickoff()
        self.state.draft = str(result)
        print(f"[FLOW] Draft complete ({len(self.state.draft)} chars)")

    # ── Step 3: @listen — ALSO reacts to plan_content ────────────────────────
    # Multiple methods can listen to the same upstream step.
    # write_draft AND generate_seo both listen to plan_content —
    # they would run in parallel in async mode. In sync mode, sequential.
    # This is Fan-Out: one event → multiple listeners triggered.

    @listen(plan_content)
    def generate_seo_keywords(self):
        print("\n[FLOW] Step 2b: Generating SEO keywords (parallel with draft)...")

        from crewai import Agent, Task, Crew

        seo_agent = Agent(
            role="SEO Specialist",
            goal="Identify high-value SEO keywords for the article",
            backstory="You identify search keywords that maximize article discoverability.",
            verbose=False,
        )

        seo_task = Task(
            description=(
                f"Based on this outline: {self.state.outline[:500]}...\n\n"
                f"Identify 5-7 SEO keywords for the topic: '{self.state.topic}'\n"
                "Return ONLY a comma-separated list of keywords, nothing else."
            ),
            expected_output="A comma-separated list of 5-7 SEO keywords.",
            agent=seo_agent,
        )

        result = Crew(agents=[seo_agent], tasks=[seo_task], verbose=False).kickoff()
        keywords_text = str(result).strip()
        self.state.seo_keywords = [k.strip() for k in keywords_text.split(",")]
        print(f"[FLOW] SEO keywords: {self.state.seo_keywords}")

    # ── Step 4: Quality check — listens to write_draft ───────────────────────
    # This only runs AFTER write_draft completes (not after plan_content).
    # The flow wires these dependencies automatically from your decorators.

    @listen(write_draft)
    def quality_check(self):
        print("\n[FLOW] Step 3: Quality checking the draft...")

        from crewai import Agent, Task, Crew

        reviewer = Agent(
            role="Quality Editor",
            goal="Score article quality and decide if it needs revision",
            backstory="You are a tough editor who scores articles 1-10 and gives clear verdicts.",
            verbose=False,
        )

        review_task = Task(
            description=(
                f"Review this article draft:\n\n{self.state.draft}\n\n"
                "Score it 1-10 on: clarity, engagement, structure, and depth.\n"
                "Respond in EXACTLY this format (no other text):\n"
                "SCORE: <number>\n"
                "VERDICT: <pass or revise>\n"
                "REASON: <one sentence>"
            ),
            expected_output="SCORE: N\nVERDICT: pass or revise\nREASON: one sentence",
            agent=reviewer,
        )

        result = Crew(agents=[reviewer], tasks=[review_task], verbose=False).kickoff()
        output = str(result)

        # Parse the structured response
        for line in output.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    self.state.quality_score = int(line.split(":")[1].strip().split()[0])
                except:
                    self.state.quality_score = 7
            elif line.startswith("VERDICT:"):
                verdict = line.split(":")[1].strip().lower()
                self.state.quality_verdict = "pass" if "pass" in verdict else "revise"

        print(f"[FLOW] Quality score: {self.state.quality_score}/10 — Verdict: {self.state.quality_verdict}")

    # ── Step 5: @router — conditional branching ───────────────────────────────
    # @router reads state and returns a STRING route label.
    # IMPORTANT: route labels must be DIFFERENT from method names —
    # if label == method name the method re-triggers itself (infinite loop).
    # Convention: use short descriptive labels like "approved" / "rejected".

    @router(quality_check)
    def route_on_quality(self):
        print(f"\n[FLOW] Routing based on quality verdict: '{self.state.quality_verdict}'")
        if self.state.quality_score >= 7 and self.state.quality_verdict == "pass":
            return "approved"       # ← label, NOT the method name
        else:
            return "needs_revision" # ← label, NOT the method name

    # ── Step 6a: Finalize (happy path) ───────────────────────────────────────
    @listen("approved")             # ← listens for the route label "approved"
    def finalize_article(self):
        print("\n[FLOW] Step 4a: Finalizing article for publication...")
        keywords_str = ", ".join(self.state.seo_keywords)
        self.state.final_article = (
            f"{self.state.draft}\n\n"
            f"---\n*SEO Keywords: {keywords_str}*"
        )
        self.state.publish_ready = True
        print("[FLOW] Article finalized and marked publish-ready!")

    # ── Step 6b: Revise (revision path) ──────────────────────────────────────
    @listen("needs_revision")       # ← listens for the route label "needs_revision"
    def revise_draft(self):
        print("\n[FLOW] Step 4b: Draft needs revision — improving...")

        from crewai import Agent, Task, Crew

        reviser = Agent(
            role="Senior Editor",
            goal="Improve the article draft to meet quality standards",
            backstory="You take good drafts and make them great through targeted improvements.",
            verbose=False,
        )

        revise_task = Task(
            description=(
                f"Improve this article draft (current score: {self.state.quality_score}/10):\n\n"
                f"{self.state.draft}\n\n"
                "Focus on: stronger opening, clearer examples, better flow.\n"
                "Return the complete revised article in markdown."
            ),
            expected_output="A revised, improved version of the full article in markdown.",
            agent=reviser,
        )

        result = Crew(agents=[reviser], tasks=[revise_task], verbose=False).kickoff()
        keywords_str = ", ".join(self.state.seo_keywords)
        self.state.final_article = (
            f"{str(result)}\n\n"
            f"---\n*SEO Keywords: {keywords_str}*"
        )
        self.state.publish_ready = True
        print("[FLOW] Revision complete — article ready!")


# ── Run the Flow ──────────────────────────────────────────────────────────────

flow = ContentCreationFlow()

print("\n" + "=" * 60)
print("CREWAI FLOW — CONTENT CREATION PIPELINE")
print("=" * 60)
print("\nFlow steps:")
print("  plan_content (@start)")
print("  ├── write_draft (@listen)")
print("  │   └── quality_check (@listen)")
print("  │       └── route_on_quality (@router)")
print("  │           ├── 'approved' → finalize_article  (score >= 7)")
print("  │           └── 'needs_revision' → revise_draft  (score < 7)")
print("  └── generate_seo_keywords (@listen) ← parallel")
print()

result = flow.kickoff(inputs={
    "topic": "How AI agents are transforming software development workflows",
    "target_audience": "Senior software engineers and engineering managers",
})

print("\n" + "=" * 60)
print("FLOW COMPLETE — FINAL STATE")
print("=" * 60)
print(f"Quality Score:  {flow.state.quality_score}/10")
print(f"Verdict:        {flow.state.quality_verdict}")
print(f"SEO Keywords:   {', '.join(flow.state.seo_keywords)}")
print(f"Publish Ready:  {flow.state.publish_ready}")
print("\n--- FINAL ARTICLE ---")
print(flow.state.final_article[:1000] + "..." if len(flow.state.final_article) > 1000 else flow.state.final_article)
