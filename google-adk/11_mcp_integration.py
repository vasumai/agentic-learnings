"""
Lesson 11 — MCP Integration
=============================
Concepts covered:
  - McpToolset: connects an ADK agent to any MCP server natively
  - StdioConnectionParams: launches a local MCP server as a subprocess (stdio)
  - The MCP protocol bridge: MCP tools become ADK tools automatically
  - tool_filter: expose only a subset of the MCP server's tools
  - Lifecycle: always close the toolset when done

Key insight:
  MCP (Model Context Protocol) is an open standard for exposing tools to LLMs.
  ADK's McpToolset bridges the gap — it launches an MCP server (or connects to
  a remote one), fetches its tool catalogue, and wraps each tool as a native
  ADK BaseTool. The agent doesn't know (or care) that the tools came from MCP.

  Connection types supported:
    StdioConnectionParams   — spawn a local process, talk over stdin/stdout
    SseConnectionParams     — connect to a remote SSE endpoint
    StreamableHTTPConnectionParams — connect via Streamable HTTP

  In this lesson we use stdio: our MCP server is 11_mcp_server.py, which
  exposes three text-analysis tools. ADK spawns it as a subprocess and
  communicates via stdin/stdout using the MCP JSON-RPC protocol.

Two demos:
  1. Full toolset — agent sees all three tools from the MCP server
  2. Filtered toolset — agent sees only count_words (tool_filter demo)
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from mcp import StdioServerParameters
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.genai import types

load_dotenv()

# Path to our companion MCP server script
MCP_SERVER_SCRIPT = str(Path(__file__).parent / "11_mcp_server.py")

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_turn(runner: Runner, session_id: str, message: str) -> str:
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


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 1 — Full MCP toolset
#
# McpToolset launches 11_mcp_server.py as a subprocess.
# The MCP protocol handshake happens automatically:
#   1. ADK sends "initialize" → server sends back capabilities
#   2. ADK sends "tools/list" → server returns [count_words, get_text_stats,
#      find_keywords]
#   3. Each MCP tool is wrapped as an ADK BaseTool
#   4. Agent sees them as normal ADK tools — it can call any of them
#
# StdioConnectionParams (preferred over StdioServerParameters):
#   - Adds timeout support for slow-starting servers
#   - Wraps StdioServerParameters internally
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_TEXT = """
The Model Context Protocol (MCP) is an open standard that enables
developers to build secure, two-way connections between their data
sources and AI-powered tools. Organisations can now give AI systems
direct, secure access to the tools and data they need.

MCP was created by Anthropic and has been adopted widely. Google ADK
supports MCP natively through McpToolset, making it straightforward
to plug any MCP-compatible server into an ADK agent without writing
any glue code.
"""


async def demo_full_toolset():
    print("── Demo 1: Full MCP toolset (all three tools) ──\n")

    # McpToolset launches the MCP server subprocess and holds the connection
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,   # same Python interpreter as this script
                args=[MCP_SERVER_SCRIPT],
            ),
            timeout=30.0,  # seconds to wait for server to start
        ),
    )

    agent = Agent(
        name="text_analyst",
        model="gemini-2.5-flash",
        instruction=(
            "You are a text analysis assistant. Use the available tools to "
            "analyse text and report detailed statistics. Always use multiple "
            "tools to give a complete picture."
        ),
        tools=[toolset],
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="11_demo1", session_service=session_service)
    session = await session_service.create_session(app_name="11_demo1", user_id="learner")

    try:
        print(f"Sample text (excerpt): {SAMPLE_TEXT.strip()[:80]}...\n")

        print("Turn 1: Full text analysis")
        reply = await run_turn(
            runner, session.id,
            f"Analyse this text completely — word count, statistics, and keywords:\n{SAMPLE_TEXT}"
        )
        print(f"Agent: {reply}\n")

        print("Turn 2: Follow-up — keywords only")
        reply2 = await run_turn(
            runner, session.id,
            "Now just find me the keywords (words with 7+ characters)."
        )
        print(f"Agent: {reply2}\n")

    finally:
        # Always close the toolset — this terminates the MCP server subprocess
        await toolset.close()
        print("  [MCP] Toolset closed — server subprocess terminated\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 2 — Tool filtering
#
# The MCP server still exposes all three tools, but tool_filter restricts
# which ones ADK surfaces to the agent. The agent only sees count_words.
#
# Use cases for tool_filter:
#   - Least-privilege: don't give agents tools they don't need
#   - Role separation: different agents get different tool subsets
#   - Cost control: block expensive tools on low-priority agents
# ══════════════════════════════════════════════════════════════════════════════

async def demo_tool_filter():
    print("── Demo 2: tool_filter — expose only count_words ──\n")

    # Same MCP server, but we only expose one tool to the agent
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=[MCP_SERVER_SCRIPT],
            ),
            timeout=30.0,
        ),
        tool_filter=["count_words"],  # ← only this tool is visible to the agent
    )

    agent = Agent(
        name="word_counter",
        model="gemini-2.5-flash",
        instruction=(
            "You are a word-counting assistant. Count words in any text the "
            "user provides and report the results clearly."
        ),
        tools=[toolset],
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="11_demo2", session_service=session_service)
    session = await session_service.create_session(app_name="11_demo2", user_id="learner")

    try:
        print("Turn 1: Count words (agent has count_words tool only)")
        reply = await run_turn(
            runner, session.id,
            f"How many words are in this passage?\n{SAMPLE_TEXT}"
        )
        print(f"Agent: {reply}\n")

        print("Turn 2: Ask for keywords — agent cannot, tool not available")
        reply2 = await run_turn(
            runner, session.id,
            "Can you also find the keywords in that passage?"
        )
        print(f"Agent: {reply2}\n")

    finally:
        await toolset.close()
        print("  [MCP] Toolset closed — server subprocess terminated\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 11 — MCP Integration ===\n")
    print(f"MCP server script: {MCP_SERVER_SCRIPT}\n")

    await demo_full_toolset()
    await demo_tool_filter()


if __name__ == "__main__":
    asyncio.run(main())
