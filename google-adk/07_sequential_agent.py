"""
Lesson 07 — SequentialAgent
============================
Concepts covered:
  - SequentialAgent: runs sub-agents one after another in order
  - Data passing between agents via session state (output_key → state key)
  - Combining structured output (lesson 05) + tool agents in a pipeline
  - Each sub-agent reads from and writes to shared session state

Key insight:
  SequentialAgent is ADK's pipeline primitive. It doesn't have a model itself —
  it just orchestrates other agents in sequence. The agents share session state,
  so agent N can read what agent N-1 wrote via output_key.

  This is also the workaround for the built-in tool mixing restriction (lesson 03):
  - Agent A: uses google_search (built-in only)
  - Agent B: uses custom Python tools
  - SequentialAgent runs A then B — no mixing within a single agent.

Pipeline in this lesson:
  [1] ResearchAgent   — fetches facts about a topic (uses google_search)
                        writes → state["research"]
  [2] AnalysisAgent   — reads state["research"], extracts key points
                        writes → state["analysis"] (structured Pydantic output)
  [3] SummaryAgent    — reads state["analysis"], writes a final human summary
                        writes → state["final_summary"]
"""

import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

load_dotenv()

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_pipeline(runner: Runner, session_id: str, message: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    reply = ""
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    reply = part.text
    return reply


# ── Pydantic schema for the analysis step ─────────────────────────────────────

class TopicAnalysis(BaseModel):
    topic: str = Field(description="The topic being analysed")
    key_facts: list[str] = Field(description="3-5 most important facts")
    significance: str = Field(description="Why this topic matters in one sentence")
    open_questions: Optional[list[str]] = Field(
        default=None, description="Interesting unanswered questions, if any"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 1 — ResearchAgent
# Uses google_search (built-in tool — no custom tools allowed alongside it)
# Writes its findings to state["research"] via output_key
# ══════════════════════════════════════════════════════════════════════════════

research_agent = Agent(
    name="research_agent",
    model="gemini-2.5-flash",
    description="Researches a topic using Google Search and summarises findings.",
    instruction=(
        "You are a research assistant. Use Google Search to find current, accurate "
        "information about the topic provided. Write a concise factual summary "
        "(3-5 sentences) covering the most important recent facts."
    ),
    tools=[google_search],
    output_key="research",      # saves reply → session.state["research"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 2 — AnalysisAgent
# Reads state["research"] (injected via instruction template)
# Uses output_schema for structured extraction
# Writes structured result to state["analysis"] via output_key
#
# NOTE: output_schema disables tools — that's fine here, this agent only reads
# ══════════════════════════════════════════════════════════════════════════════

analysis_agent = Agent(
    name="analysis_agent",
    model="gemini-2.5-flash",
    description="Analyses research findings and extracts structured key points.",
    instruction=(
        "You are an analytical assistant. Analyse the research provided in the "
        "session context and extract structured information.\n\n"
        "The research findings are available in session state under 'research'."
    ),
    output_schema=TopicAnalysis,
    output_key="analysis",      # saves structured reply → session.state["analysis"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent 3 — SummaryAgent
# Reads state["analysis"] and writes a friendly final summary
# Plain text output, no schema needed
# ══════════════════════════════════════════════════════════════════════════════

summary_agent = Agent(
    name="summary_agent",
    model="gemini-2.5-flash",
    description="Writes a clear, engaging summary from structured analysis.",
    instruction=(
        "You are a skilled science communicator. Using the structured analysis "
        "available in the session context (state key: 'analysis'), write a "
        "clear, engaging 2-3 paragraph summary suitable for a general audience. "
        "Highlight why the topic matters and end with an interesting open question."
    ),
    output_key="final_summary",
)

# ══════════════════════════════════════════════════════════════════════════════
# SequentialAgent — wires all three together
# ══════════════════════════════════════════════════════════════════════════════

pipeline = SequentialAgent(
    name="research_pipeline",
    description="Full research pipeline: search → analyse → summarise",
    sub_agents=[research_agent, analysis_agent, summary_agent],
)

# ── Runner ─────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
runner = Runner(
    agent=pipeline,
    app_name="sequential_demo",
    session_service=session_service,
)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 07 — SequentialAgent ===\n")
    print("Pipeline: ResearchAgent → AnalysisAgent → SummaryAgent\n")

    session = await session_service.create_session(
        app_name="sequential_demo", user_id="learner"
    )

    topic = "Quantum computing breakthroughs in 2025"
    print(f"Topic: {topic}\n")
    print("Running pipeline...\n")

    await run_pipeline(runner, session.id, f"Research and summarise: {topic}")

    # Read the final state to show what each agent produced
    stored = await session_service.get_session(
        app_name="sequential_demo", user_id="learner", session_id=session.id
    )

    print("── State after pipeline ──\n")

    research = stored.state.get("research", "")
    print(f"[research_agent → state['research']]")
    print(f"{research[:300]}...\n" if len(research) > 300 else f"{research}\n")

    analysis_raw = stored.state.get("analysis", "")
    if analysis_raw:
        try:
            analysis = TopicAnalysis.model_validate_json(analysis_raw) \
                if isinstance(analysis_raw, str) \
                else TopicAnalysis.model_validate(analysis_raw)
            print(f"[analysis_agent → state['analysis']] (structured)")
            print(f"  topic      : {analysis.topic}")
            print(f"  key facts  :")
            for fact in analysis.key_facts:
                print(f"    • {fact}")
            print(f"  significance: {analysis.significance}")
        except Exception:
            print(f"[analysis] {str(analysis_raw)[:200]}")
    print()

    final = stored.state.get("final_summary", "")
    print(f"[summary_agent → state['final_summary']]")
    print(final)


if __name__ == "__main__":
    asyncio.run(main())
