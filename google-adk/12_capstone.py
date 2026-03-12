"""
Lesson 12 — Capstone: Content Review Pipeline
================================================
This capstone wires together every major concept from lessons 01–11
into a single realistic pipeline: a blog-post editorial system.

Concepts combined:
  Lesson 01-02  Agent + tools
  Lesson 04     Multi-turn session state
  Lesson 05     Structured output (Pydantic output_schema)
  Lesson 06     Callbacks (before_agent_callback, before_tool_callback)
  Lesson 07     SequentialAgent — pipeline stages
  Lesson 08     ParallelAgent — concurrent specialist analysis
  Lesson 10     Human-in-the-Loop approval gate
  Lesson 11     MCP integration (McpToolset → 11_mcp_server.py)

Pipeline architecture:
  [1] StatsAgent      uses MCP to get word/sentence stats   → state["post_stats"]
  [2] ParallelAgent   three analysts run simultaneously:
        TopicAgent    → state["topic_analysis"]
        StyleAgent    → state["style_analysis"]
        SEOAgent      → state["seo_keywords"]
  [3] ReviewAgent     synthesises all analyses, Pydantic output → state["decision"]
  [4] PublishAgent    calls publish_post() — HITL gate intercepts for approval

Key insight:
  Real pipelines combine ALL these patterns. The trick is clean hand-offs:
  every agent writes ONE thing (output_key), the next agent reads it from state.
  Callbacks and HITL sit orthogonally — they intercept without changing agent logic.
"""

import asyncio
import sys
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mcp import StdioServerParameters
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()

MCP_SERVER_SCRIPT = str(Path(__file__).parent / "11_mcp_server.py")

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_pipeline(runner: Runner, session_id: str, message: str) -> None:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    async for _ in runner.run_async(
        user_id="editor", session_id=session_id, new_message=content
    ):
        pass  # results are read from session state after completion


# ── Sample blog post ───────────────────────────────────────────────────────────

BLOG_POST = """
Title: Why Every Developer Should Understand Agentic AI

Agentic AI is no longer a research concept — it's shipping in production.
Frameworks like Google ADK, LangGraph, and CrewAI let developers build
systems where an LLM doesn't just answer questions; it takes multi-step
actions, uses tools, coordinates with other agents, and even pauses for
human approval before doing something irreversible.

The shift matters for three reasons. First, the developer surface area has
exploded. You now design agent workflows, not just prompts. Second, correctness
and safety requirements are stricter — a hallucinated SQL query can delete
rows. Third, debugging is harder: failures emerge from agent interactions,
not single model calls.

Understanding how LoopAgent, SequentialAgent, and ParallelAgent compose,
how session state flows between steps, and how callbacks intercept tool calls
will be as foundational as knowing REST in 2015. Developers who get ahead of
this curve now will build the systems everyone else relies on.

Start small: add one tool to one agent, run it, trace the session events.
Then add a second agent and wire them with output_key. The patterns click
fast once you have working code in front of you.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Structured output schema  (Lesson 05)
# ══════════════════════════════════════════════════════════════════════════════

class EditorialDecision(BaseModel):
    verdict: Literal["APPROVE", "REVISE", "REJECT"] = Field(
        description="Final editorial decision"
    )
    score: int = Field(description="Overall quality score 0-100")
    strengths: list[str] = Field(description="2-3 specific strengths of the post")
    improvements: list[str] = Field(description="2-3 concrete improvement suggestions")
    summary: str = Field(description="One-sentence editorial summary")


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks  (Lesson 06)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_entry_log(callback_context: CallbackContext) -> types.Content | None:
    """Log each agent invocation — pure observability, never blocks."""
    print(f"  → [{callback_context.agent_name}] starting")
    return None  # always proceed


PUBLISH_APPROVED = True   # flip to False to simulate rejection


def publish_gate(
    tool: BaseTool, args: dict, tool_context: ToolContext
) -> dict | None:
    """HITL gate: intercept publish_post and require human sign-off.  (Lesson 10)"""
    if tool.name != "publish_post":
        return None  # auto-approve any other tools

    print(f"\n  ┌─ PUBLISH APPROVAL REQUIRED ─────────────────────────")
    print(f"  │ Title : {args.get('title', '')}")
    print(f"  │ Length: {len(args.get('content', '').split())} words")
    print(f"  └──────────────────────────────────────────────────────")

    if PUBLISH_APPROVED:
        print("  [HITL] APPROVED — post will be published\n")
        return None
    else:
        print("  [HITL] REJECTED — post sent back for revision\n")
        return {
            "error": "Human editor rejected the publish request. "
                     "Post requires revision before it can go live."
        }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — StatsAgent  (uses MCP toolset from Lesson 11)
#
# Calls count_words and get_text_stats from our MCP server.
# Writes raw stats to state["post_stats"] for downstream agents.
# ══════════════════════════════════════════════════════════════════════════════

def build_stats_agent(toolset: McpToolset) -> Agent:
    return Agent(
        name="stats_agent",
        model="gemini-2.5-flash",
        description="Measures text metrics using MCP tools.",
        instruction=(
            "You are a text metrics agent. Use count_words AND get_text_stats "
            "on the blog post provided. Report the results as a plain summary: "
            "e.g. '74 words, 4 sentences, 2 paragraphs, avg word length 5.4'. "
            "Do not add any commentary."
        ),
        tools=[toolset],
        output_key="post_stats",
        before_agent_callback=pipeline_entry_log,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — ParallelAgent  (Lesson 08)
#
# Three specialist agents run concurrently, each writing to its own state key.
# No tools needed — pure LLM analysis of the post.
# ══════════════════════════════════════════════════════════════════════════════

topic_agent = Agent(
    name="topic_agent",
    model="gemini-2.5-flash",
    description="Assesses technical depth and topic coverage of a blog post.",
    instruction=(
        "You are a technical editor. Analyse the blog post in the session context. "
        "In 2-3 sentences assess: 1) topic clarity, 2) technical depth, "
        "3) whether the argument is well-structured. Be concise and direct."
    ),
    output_key="topic_analysis",
    before_agent_callback=pipeline_entry_log,
)

style_agent = Agent(
    name="style_agent",
    model="gemini-2.5-flash",
    description="Evaluates writing style, tone, and readability.",
    instruction=(
        "You are a writing coach. Analyse the blog post in the session context. "
        "In 2-3 sentences assess: 1) readability for a developer audience, "
        "2) tone (too formal / too casual / just right), "
        "3) one specific phrasing improvement. Be specific."
    ),
    output_key="style_analysis",
    before_agent_callback=pipeline_entry_log,
)

seo_agent = Agent(
    name="seo_agent",
    model="gemini-2.5-flash",
    description="Identifies SEO keywords and search visibility opportunities.",
    instruction=(
        "You are an SEO specialist. Analyse the blog post in the session context. "
        "List 5 high-value keywords or phrases a developer would search for "
        "that are either present or should be added. One line each."
    ),
    output_key="seo_keywords",
    before_agent_callback=pipeline_entry_log,
)

analyst_panel = ParallelAgent(
    name="analyst_panel",
    description="Topic, style, and SEO specialists analyse the post in parallel.",
    sub_agents=[topic_agent, style_agent, seo_agent],
)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — ReviewAgent  (structured output, Lesson 05)
#
# Reads all four state keys, produces a structured EditorialDecision.
# output_schema disables tools — that's fine, this agent only reads + reasons.
# ══════════════════════════════════════════════════════════════════════════════

review_agent = Agent(
    name="review_agent",
    model="gemini-2.5-flash",
    description="Synthesises all analyses into a structured editorial decision.",
    instruction=(
        "You are the chief editor. Four specialists have analysed a blog post. "
        "Their findings are in session state:\\n"
        "  post_stats    : text metrics (from MCP)\\n"
        "  topic_analysis: technical depth assessment\\n"
        "  style_analysis: writing style assessment\\n"
        "  seo_keywords  : SEO keyword suggestions\\n\\n"
        "Synthesise all four into a balanced EditorialDecision. "
        "Score the post 0-100. Verdict must be APPROVE (≥75), REVISE (50-74), "
        "or REJECT (<50)."
    ),
    output_schema=EditorialDecision,
    output_key="decision",
    before_agent_callback=pipeline_entry_log,
)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — PublishAgent  (HITL via before_tool_callback, Lesson 10)
#
# Reads state["decision"]. If verdict is APPROVE, calls publish_post().
# The HITL gate intercepts that call for final human sign-off.
# ══════════════════════════════════════════════════════════════════════════════

def publish_post(title: str, content: str) -> dict:
    """Publishes a blog post to the content management system.

    Args:
        title: The post title.
        content: The full post content to publish.
    """
    word_count = len(content.split())
    print(f"  [CMS] Publishing '{title}' ({word_count} words)")
    return {"published": True, "title": title, "word_count": word_count, "url": f"/blog/{title.lower().replace(' ', '-')}"}


publish_agent = Agent(
    name="publish_agent",
    model="gemini-2.5-flash",
    description="Publishes approved posts to the CMS after HITL sign-off.",
    instruction=(
        "You are a publishing assistant. Check the editorial decision in session "
        "state key 'decision'. "
        "If the verdict is APPROVE, call publish_post with the blog post title "
        "and content. "
        "If the verdict is REVISE or REJECT, report that the post was not "
        "published and explain why based on the decision."
    ),
    tools=[publish_post],
    before_tool_callback=publish_gate,
    before_agent_callback=pipeline_entry_log,
)


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline  (Lesson 07 — SequentialAgent)
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(toolset: McpToolset) -> SequentialAgent:
    return SequentialAgent(
        name="editorial_pipeline",
        description="Full blog post review: stats → analysis → decision → publish.",
        sub_agents=[
            build_stats_agent(toolset),
            analyst_panel,
            review_agent,
            publish_agent,
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 12 — Capstone: Content Review Pipeline ===\n")
    print("Architecture:")
    print("  StatsAgent (MCP)")
    print("  → [TopicAgent ║ StyleAgent ║ SEOAgent]  (parallel)")
    print("  → ReviewAgent (structured output)")
    print("  → PublishAgent (HITL gate)")
    print()
    print(f"Blog post title: '{BLOG_POST.strip().splitlines()[0]}'\n")

    # Create MCP toolset — manages the subprocess lifecycle
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=[MCP_SERVER_SCRIPT],
            ),
            timeout=30.0,
        ),
    )

    pipeline = build_pipeline(toolset)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        app_name="12_capstone",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="12_capstone", user_id="editor"
    )

    try:
        print("Running pipeline...\n")
        await run_pipeline(runner, session.id, BLOG_POST)

        # Read and display each state key
        stored = await session_service.get_session(
            app_name="12_capstone", user_id="editor", session_id=session.id
        )
        state = stored.state

        print("\n── Pipeline results (session state) ──\n")

        print("[stats_agent → post_stats]  (via MCP)")
        print(f"  {state.get('post_stats', 'N/A')}\n")

        print("[topic_agent → topic_analysis]")
        print(f"  {state.get('topic_analysis', 'N/A')}\n")

        print("[style_agent → style_analysis]")
        print(f"  {state.get('style_analysis', 'N/A')}\n")

        print("[seo_agent → seo_keywords]")
        print(f"  {state.get('seo_keywords', 'N/A')}\n")

        print("[review_agent → decision]  (structured Pydantic output)")
        raw_decision = state.get("decision")
        if raw_decision:
            try:
                decision = EditorialDecision.model_validate_json(raw_decision) \
                    if isinstance(raw_decision, str) \
                    else EditorialDecision.model_validate(raw_decision)
                print(f"  Verdict    : {decision.verdict}")
                print(f"  Score      : {decision.score}/100")
                print(f"  Summary    : {decision.summary}")
                print(f"  Strengths  :")
                for s in decision.strengths:
                    print(f"    + {s}")
                print(f"  Improvements:")
                for i in decision.improvements:
                    print(f"    → {i}")
            except Exception:
                print(f"  {str(raw_decision)[:300]}")
        print()

    finally:
        await toolset.close()
        print("[MCP] Toolset closed — server subprocess terminated")


if __name__ == "__main__":
    asyncio.run(main())
