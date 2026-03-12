"""
Lesson 08 — ParallelAgent
==========================
Concepts covered:
  - ParallelAgent: runs all sub-agents concurrently (true async fan-out)
  - Each sub-agent runs in an isolated branch context — no shared history
  - All sub-agents share the same session state (for reading inputs)
  - Each sub-agent writes to its own output_key — no conflicts
  - Combining ParallelAgent + SequentialAgent: fan-out then merge

Key insight:
  ParallelAgent does NOT have a model — it's pure orchestration.
  It's always combined with a merger agent in a SequentialAgent wrapper:

    SequentialAgent([
        ParallelAgent([agent_a, agent_b, agent_c]),  ← all run at once
        merger_agent,                                 ← reads all three outputs
    ])

  Use it when sub-tasks are independent and you want to save wall-clock time.

Pipeline in this lesson (job posting analysis):
  ParallelAgent runs simultaneously:
    ├── TechAgent    → analyses tech stack requirements → state["tech_analysis"]
    ├── CultureAgent → analyses work culture signals    → state["culture_analysis"]
    └── SalaryAgent  → analyses compensation signals    → state["salary_analysis"]
  Then:
    MergeAgent → reads all three, writes final recommendation → state["recommendation"]
"""

import asyncio
import time
from dotenv import load_dotenv
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_pipeline(runner: Runner, session_id: str, message: str) -> None:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        pass  # we read results from session state after completion


# ── Sample job posting ─────────────────────────────────────────────────────────

JOB_POSTING = """
Senior Backend Engineer — CloudScale AI (Series B, San Francisco / Remote)

We're looking for a senior engineer to join our platform team. You'll be building
and scaling distributed systems that process billions of events daily.

Tech stack: Python, Go, Kubernetes, Kafka, PostgreSQL, Redis, AWS (ECS, RDS, S3).
Experience with ML pipelines and vector databases (Pinecone, Weaviate) is a big plus.

Culture: We move fast — most features ship within two-week sprints. On-call rotation
(roughly once every 6 weeks). Engineers own their services end-to-end from design
to production. We value autonomy over process. Remote-first with quarterly offsites.

Compensation: $180,000–$230,000 base + equity (0.05–0.15%). Full benefits.
401k with 4% match. $2,000/year learning budget. Unlimited PTO (average ~18 days taken).
"""

# ══════════════════════════════════════════════════════════════════════════════
# Sub-agents (run in parallel — each focuses on ONE dimension)
# ══════════════════════════════════════════════════════════════════════════════

tech_agent = Agent(
    name="tech_agent",
    model="gemini-2.5-flash",
    description="Analyses the technology stack and engineering requirements in job postings.",
    instruction=(
        "You are a senior engineer. Analyse ONLY the technical requirements and stack "
        "from the job posting in the session context. In 2-3 sentences cover: "
        "1) How modern/relevant the stack is, 2) Estimated complexity/scope, "
        "3) Any red flags or exciting tech. Be direct and opinionated."
    ),
    output_key="tech_analysis",
)

culture_agent = Agent(
    name="culture_agent",
    model="gemini-2.5-flash",
    description="Analyses work culture, team dynamics, and work-life balance signals.",
    instruction=(
        "You are an experienced engineering manager. Analyse ONLY the culture, "
        "work style, and work-life balance signals from the job posting in the "
        "session context. In 2-3 sentences cover: 1) Work pace and autonomy, "
        "2) On-call burden, 3) Red or green flags for sustainable work. Be honest."
    ),
    output_key="culture_analysis",
)

salary_agent = Agent(
    name="salary_agent",
    model="gemini-2.5-flash",
    description="Analyses compensation, equity, and total compensation package.",
    instruction=(
        "You are a compensation expert. Analyse ONLY the salary, equity, and benefits "
        "from the job posting in the session context. In 2-3 sentences cover: "
        "1) How the base compares to market for a senior engineer in SF, "
        "2) Whether the equity range is meaningful, 3) Standout benefits or gaps. Be specific."
    ),
    output_key="salary_analysis",
)

# ══════════════════════════════════════════════════════════════════════════════
# ParallelAgent — fans out to all three simultaneously
# ══════════════════════════════════════════════════════════════════════════════

analyst_panel = ParallelAgent(
    name="analyst_panel",
    description="Three specialist agents analyse the job posting in parallel.",
    sub_agents=[tech_agent, culture_agent, salary_agent],
)

# ══════════════════════════════════════════════════════════════════════════════
# Merger agent — reads all three outputs, writes final recommendation
# ══════════════════════════════════════════════════════════════════════════════

merger_agent = Agent(
    name="merger_agent",
    model="gemini-2.5-flash",
    description="Synthesises specialist analyses into a final hiring recommendation.",
    instruction=(
        "You are a senior career advisor. Three specialists have analysed a job posting "
        "and their findings are in the session state:\n"
        "  - 'tech_analysis'    : technology stack assessment\n"
        "  - 'culture_analysis' : work culture assessment\n"
        "  - 'salary_analysis'  : compensation assessment\n\n"
        "Synthesise all three into a balanced final recommendation. Structure your "
        "response as:\n"
        "VERDICT: [Apply / Apply with caution / Pass]\n"
        "REASON: 2-3 sentences integrating all three perspectives.\n"
        "TOP PRO: One standout positive.\n"
        "TOP CON: One standout concern."
    ),
    output_key="recommendation",
)

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: ParallelAgent → MergeAgent (wrapped in SequentialAgent)
# ══════════════════════════════════════════════════════════════════════════════

pipeline = SequentialAgent(
    name="job_analysis_pipeline",
    description="Parallel job posting analysis followed by synthesis.",
    sub_agents=[analyst_panel, merger_agent],
)

# ── Runner ─────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
runner = Runner(
    agent=pipeline,
    app_name="parallel_demo",
    session_service=session_service,
)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 08 — ParallelAgent ===\n")
    print("Pipeline: [TechAgent ║ CultureAgent ║ SalaryAgent] → MergeAgent\n")
    print("Job posting (excerpt):")
    print(f"{JOB_POSTING[:200].strip()}...\n")

    session = await session_service.create_session(
        app_name="parallel_demo", user_id="learner"
    )

    start = time.time()
    await run_pipeline(runner, session.id, JOB_POSTING)
    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.1f}s\n")

    # Read session state to show each agent's output
    stored = await session_service.get_session(
        app_name="parallel_demo", user_id="learner", session_id=session.id
    )
    state = stored.state

    print("── Parallel outputs (all written concurrently) ──\n")

    print("[tech_agent]")
    print(f"  {state.get('tech_analysis', 'N/A')}\n")

    print("[culture_agent]")
    print(f"  {state.get('culture_analysis', 'N/A')}\n")

    print("[salary_agent]")
    print(f"  {state.get('salary_analysis', 'N/A')}\n")

    print("── Merged recommendation ──\n")
    print(state.get("recommendation", "N/A"))


if __name__ == "__main__":
    asyncio.run(main())
