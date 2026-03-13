"""
Lesson 11 — MCP Integration
=============================
Covers:
  - MCPServerStdio — connect to an MCP server over stdio
  - MCPServerStdioParams — configure the server process (command, args, cwd)
  - connect() / cleanup() lifecycle — must bracket every use
  - mcp_servers=[...] on Agent — tools auto-discovered from the server
  - cache_tools_list=True — avoid re-fetching tool list every turn
  - Using async context manager (recommended pattern)
  - Combining MCP tools with local @function_tools on the same agent

Key concept:
  MCP (Model Context Protocol) is an open standard for connecting agents
  to external tool servers. Instead of writing tools inline, you run a
  separate process that speaks MCP and the SDK discovers its tools
  automatically via list_tools().

  The companion server (11_mcp_server.py) exposes 4 tools:
    get_weather, list_notes, add_note, get_note

  MCPServerStdio launches it as a subprocess — no separate terminal needed.
  You MUST call connect() before use and cleanup() after, or use the
  async context manager which handles both.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerStdio, MCPServerStdioParams, ToolFilterStatic

load_dotenv()

# Path to our companion MCP server script
SERVER_SCRIPT = str(Path(__file__).parent / "11_mcp_server.py")
PYTHON = sys.executable   # same Python/venv that's running this file


# ---------------------------------------------------------------------------
# 1. Basic MCP connection — connect, use, cleanup
# ---------------------------------------------------------------------------

async def basic_mcp_demo():
    print("=" * 50)
    print("PART 1: Basic MCP — connect, discover tools, use them")
    print("=" * 50)

    server = MCPServerStdio(
        params=MCPServerStdioParams(
            command=PYTHON,
            args=[SERVER_SCRIPT],
        ),
        cache_tools_list=True,   # fetch tool list once, cache it
        name="LessonServer",
    )

    await server.connect()
    try:
        # Inspect what tools the server exposes
        tools = await server.list_tools()
        print(f"\nTools discovered from MCP server:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        agent = Agent(
            name="WeatherAgent",
            instructions="You are a helpful assistant. Use tools to answer questions.",
            model="gpt-4o-mini",
            mcp_servers=[server],
        )

        queries = [
            "What's the weather like in Tokyo and Mumbai?",
            "List all saved notes, then retrieve the welcome note.",
        ]

        for q in queries:
            print(f"\nUser: {q}")
            result = await Runner.run(agent, input=q)
            print(f"Agent: {result.final_output}")
    finally:
        await server.cleanup()


# ---------------------------------------------------------------------------
# 2. Async context manager — cleaner lifecycle management
# ---------------------------------------------------------------------------

async def context_manager_demo():
    print("\n" + "=" * 50)
    print("PART 2: Async context manager — recommended pattern")
    print("=" * 50)

    async with MCPServerStdio(
        params=MCPServerStdioParams(command=PYTHON, args=[SERVER_SCRIPT]),
        cache_tools_list=True,
        name="LessonServer",
    ) as server:
        agent = Agent(
            name="NoteAgent",
            instructions=(
                "You are a note-taking assistant. "
                "Use tools to save and retrieve notes. Be concise."
            ),
            model="gpt-4o-mini",
            mcp_servers=[server],
        )

        steps = [
            "Save a note titled 'meeting' with content: 'Discuss Q2 roadmap with team on Friday.'",
            "Save a note titled 'todo' with content: 'Buy groceries, call dentist, review PR #42.'",
            "List all notes.",
            "Get the meeting note.",
        ]

        result = None
        for step in steps:
            print(f"\nUser: {step}")
            history = result.to_input_list() if result else []
            result = await Runner.run(
                agent,
                input=history + [{"role": "user", "content": step}],
            )
            print(f"Agent: {result.final_output}")


# ---------------------------------------------------------------------------
# 3. MCP tools + local @function_tools on the same agent
# ---------------------------------------------------------------------------

@function_tool
def convert_temp(celsius: float) -> str:
    """Convert a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius.
    """
    f = celsius * 9 / 5 + 32
    return f"{celsius}°C = {f:.1f}°F"


async def mixed_tools_demo():
    print("\n" + "=" * 50)
    print("PART 3: MCP tools + local tools on the same agent")
    print("=" * 50)

    async with MCPServerStdio(
        params=MCPServerStdioParams(command=PYTHON, args=[SERVER_SCRIPT]),
        cache_tools_list=True,
        name="LessonServer",
    ) as server:
        agent = Agent(
            name="HybridAgent",
            instructions=(
                "You are a helpful assistant with weather and note-taking abilities. "
                "You can also convert temperatures. Use all available tools."
            ),
            model="gpt-4o-mini",
            tools=[convert_temp],         # local tool
            mcp_servers=[server],         # MCP tools (get_weather, notes)
        )

        query = (
            "Get the weather in London, convert that temperature to Fahrenheit, "
            "then save a note titled 'london-weather' with the full weather summary including both C and F."
        )
        print(f"\nUser: {query}")
        result = await Runner.run(agent, input=query)
        print(f"Agent: {result.final_output}")

        # Verify the note was saved
        verify = "Show me the london-weather note."
        print(f"\nUser: {verify}")
        result2 = await Runner.run(
            agent,
            input=result.to_input_list() + [{"role": "user", "content": verify}],
        )
        print(f"Agent: {result2.final_output}")


# ---------------------------------------------------------------------------
# 4. tool_filter — expose only a subset of MCP tools to the agent
# ---------------------------------------------------------------------------

async def tool_filter_demo():
    print("\n" + "=" * 50)
    print("PART 4: tool_filter — expose only specific MCP tools")
    print("=" * 50)

    # Only expose weather tool — hide note tools from this agent
    # tool_filter must be a ToolFilterStatic dict, not a plain list
    weather_only_filter: ToolFilterStatic = {"allowed_tool_names": ["get_weather"]}

    async with MCPServerStdio(
        params=MCPServerStdioParams(command=PYTHON, args=[SERVER_SCRIPT]),
        cache_tools_list=True,
        name="LessonServer",
        tool_filter=weather_only_filter,
    ) as server:
        agent = Agent(
            name="WeatherOnlyAgent",
            instructions="You only answer weather questions.",
            model="gpt-4o-mini",
            mcp_servers=[server],
        )

        # list_tools() with a static filter requires agent context — skip it here.
        # The filter takes effect when Runner.run() calls the agent.
        result = await Runner.run(agent, input="What's the weather in Sydney?")
        print(f"\nAgent: {result.final_output}")
        print("(Note: only get_weather was available — add_note/list_notes/get_note were filtered out)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await basic_mcp_demo()
    await context_manager_demo()
    await mixed_tools_demo()
    await tool_filter_demo()


if __name__ == "__main__":
    asyncio.run(main())
