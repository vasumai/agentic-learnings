"""
Lesson 11 — MCP Integration (Model Context Protocol)
======================================================
MCP (Model Context Protocol) is an open standard by Anthropic that lets
AI models communicate with external tools, data sources, and services
through a uniform protocol — like a USB-C port for AI integrations.

Instead of every app writing custom plugin code, MCP servers expose their
capabilities once and any MCP-compatible client (SK, Claude Desktop, etc.)
can use them.

Architecture:
  MCP Server  — exposes Tools, Resources, and Prompts over stdio or SSE
  MCP Client  — discovers and calls those capabilities (SK is the client here)

SK 1.40 MCP support:
  MCPStdioPlugin  — connects to an MCP server over stdio (subprocess)
  MCPSsePlugin    — connects to an MCP server over HTTP/SSE (remote)
  Once added as a plugin, MCP tools work IDENTICALLY to native SK plugins.

Comparison to what you know:
  LangGraph  → ToolNode with custom tool definitions per project
  CrewAI     → @tool or BaseTool defined in each project
  SK + MCP   → plug in ANY MCP server; tools appear as @kernel_function
               Zero custom plugin code needed for 3rd-party integrations

This lesson covers:
  1. What MCP looks like — create a minimal MCP server with FastMCP
  2. SK as MCP client — connect via MCPStdioPlugin, use tools naturally
  3. Resources — MCP servers can also expose read-only data (files, DB rows)
  4. Real-world MCP servers — filesystem, GitHub, databases (reference)
  5. SK as MCP server — expose your SK plugins to other MCP clients

SETUP: requires the 'mcp' package
  pip install mcp
"""

import asyncio
import os
import sys
import tempfile
import textwrap
from dotenv import load_dotenv

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

load_dotenv()

print("=" * 62)
print("Lesson 11 — MCP Integration")
print("=" * 62)

# ---------------------------------------------------------------------------
# Check MCP availability
# ---------------------------------------------------------------------------
try:
    from semantic_kernel.connectors.mcp import MCPStdioPlugin
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("\n[WARNING] MCPStdioPlugin not found in this SK version.")
    print("  Try: pip install semantic-kernel[mcp]  or  pip install mcp")
    print("  Showing code patterns only — set MCP_AVAILABLE=True once installed.\n")


# ---------------------------------------------------------------------------
# Embedded MCP server code
# ---------------------------------------------------------------------------
# We write a tiny FastMCP server to a temp file and launch it as a subprocess.
# In production you'd point MCPStdioPlugin at an installed MCP server package
# (e.g. npx @modelcontextprotocol/server-filesystem /path/to/dir).

MCP_SERVER_CODE = textwrap.dedent("""\
    \"\"\"Minimal MCP server for SK Lesson 11.\"\"\"
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("sk-lesson-11-server")

    # --- Tools (callable functions) ---

    @mcp.tool()
    def add(a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    @mcp.tool()
    def multiply(a: float, b: float) -> float:
        \"\"\"Multiply two numbers.\"\"\"
        return a * b

    @mcp.tool()
    def get_weather(city: str) -> str:
        \"\"\"Get current weather for a city (simulated).\"\"\"
        data = {
            "london":   "Cloudy, 12°C, light drizzle",
            "tokyo":    "Sunny, 22°C, gentle breeze",
            "new york": "Partly cloudy, 18°C, humid",
        }
        return data.get(city.lower(), f"No weather data for '{city}'")

    @mcp.tool()
    def summarise(text: str, max_words: int = 20) -> str:
        \"\"\"Return the first max_words words of a text as a summary.\"\"\"
        words = text.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

    # --- Resources (read-only data the client can fetch) ---

    @mcp.resource("config://app-settings")
    def app_settings() -> str:
        \"\"\"Return the application settings as a JSON string.\"\"\"
        import json
        return json.dumps({
            "version": "1.0",
            "environment": "development",
            "max_retries": 3,
            "timeout_seconds": 30,
        }, indent=2)

    if __name__ == "__main__":
        mcp.run()
""")


def write_server_file() -> str:
    """Write the MCP server code to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="sk_mcp_server_"
    ) as f:
        f.write(MCP_SERVER_CODE)
        return f.name


def make_kernel() -> tuple[sk.Kernel, AnthropicChatCompletion]:
    kernel = sk.Kernel()
    service = AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
    kernel.add_service(service)
    return kernel, service


async def run_prompt(service, kernel, prompt: str, max_tokens: int = 300) -> str:
    history = ChatHistory()
    history.add_user_message(prompt)
    response = await service.get_chat_message_content(
        chat_history=history,
        settings=PromptExecutionSettings(
            max_tokens=max_tokens,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        ),
        kernel=kernel,
    )
    return str(response)


# ---------------------------------------------------------------------------
# 1. MCP Server overview (always runs — no MCP package needed)
# ---------------------------------------------------------------------------
def mcp_overview():
    print("\n--- 1. MCP Overview ---")
    print("""
  MCP Server exposes three primitive types:
  ┌──────────────────────────────────────────────────────┐
  │  Tools      → callable functions  (like @kernel_function) │
  │  Resources  → read-only data      (files, DB rows, APIs) │
  │  Prompts    → reusable templates  (named prompt strings)  │
  └──────────────────────────────────────────────────────┘

  Transport options:
    stdio  — server is a subprocess; client talks via stdin/stdout
    SSE    — server runs as HTTP; client connects to /sse endpoint

  SK MCPStdioPlugin wraps an MCP server into a regular SK plugin.
  Once added with kernel.add_plugin(), MCP tools = native @kernel_function.
  The LLM never knows the difference.

  Real-world MCP servers you can drop in:
    npx @modelcontextprotocol/server-filesystem /path
    npx @modelcontextprotocol/server-github
    npx @modelcontextprotocol/server-postgres postgresql://...
    npx @modelcontextprotocol/server-brave-search
    (see: https://github.com/modelcontextprotocol/servers)
    """)


# ---------------------------------------------------------------------------
# 2. SK as MCP client — MCPStdioPlugin
# ---------------------------------------------------------------------------
async def mcp_client_example():
    print("\n--- 2. SK as MCP Client (MCPStdioPlugin) ---")

    if not MCP_AVAILABLE:
        print("""  [PATTERN — install mcp to run]

  server_path = write_server_file()   # write FastMCP server to temp file

  async with MCPStdioPlugin(
      name="demo",
      command=sys.executable,         # python interpreter
      args=[server_path],
  ) as mcp_plugin:
      kernel.add_plugin(mcp_plugin)   # MCP tools now work like native plugins
      result = await run_prompt(...)  # agent auto-calls MCP tools
        """)
        return

    server_path = write_server_file()
    kernel, service = make_kernel()

    print(f"  Starting MCP server: {os.path.basename(server_path)}")

    async with MCPStdioPlugin(
        name="demo",
        command=sys.executable,   # use the same Python interpreter
        args=[server_path],
    ) as mcp_plugin:
        kernel.add_plugin(mcp_plugin)

        # List available MCP tools (they are now regular KernelFunctions)
        plugin = kernel.get_plugin("demo")
        tool_names = list(plugin.functions.keys())
        print(f"  MCP tools registered as SK functions: {tool_names}\n")

        # Use them exactly like native plugins — agent picks the right tool
        prompts = [
            "What is 47 multiplied by 23?",
            "What's the weather in Tokyo?",
            "Add 15 and 27, then multiply the result by 4.",
        ]
        for prompt in prompts:
            print(f"  User: {prompt}")
            result = await run_prompt(service, kernel, prompt)
            print(f"  Agent: {result}\n")

    # Clean up temp file
    os.unlink(server_path)


# ---------------------------------------------------------------------------
# 3. MCP Resources — read-only data
# ---------------------------------------------------------------------------
async def mcp_resources_example():
    print("\n--- 3. MCP Resources (read-only data) ---")

    if not MCP_AVAILABLE:
        print("""  [PATTERN — install mcp to run]

  async with MCPStdioPlugin(name="demo", ...) as plugin:
      # List resources exposed by the server
      resources = await plugin.list_resources()
      for r in resources:
          print(r.uri, r.description)

      # Read a specific resource
      content = await plugin.read_resource("config://app-settings")
      print(content)
        """)
        return

    # MCPStdioPlugin in SK 1.40 exposes tools as KernelFunctions but does not
    # yet surface resource listing/reading as Python methods. Resources are part
    # of the MCP protocol spec and accessible via the raw MCP client session.
    # The pattern below shows how to access them via the underlying session:
    print("""
  Resources are available via the underlying MCP client session:

  from mcp import ClientSession
  from mcp.client.stdio import stdio_client, StdioServerParameters

  params = StdioServerParameters(command=sys.executable, args=[server_path])
  async with stdio_client(params) as (read, write):
      async with ClientSession(read, write) as session:
          await session.initialize()
          resources = await session.list_resources()
          for r in resources.resources:
              print(r.uri, r.name)
          content = await session.read_resource("config://app-settings")
          print(content.contents[0].text)

  In production, MCPSsePlugin (SSE transport) supports richer resource access
  depending on the server implementation.
    """)


# ---------------------------------------------------------------------------
# 4. Real-world MCP servers — reference (not run)
# ---------------------------------------------------------------------------
def real_world_servers_reference():
    print("\n--- 4. Real-World MCP Servers (reference) ---")
    print("""
  Any MCP server can be plugged into SK with MCPStdioPlugin.
  No custom plugin code needed — just point at the server.

  # Filesystem (read/write local files)
  async with MCPStdioPlugin(
      name="filesystem",
      command="npx",
      args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
  ) as fs_plugin:
      kernel.add_plugin(fs_plugin)
      # Agent can now read_file, write_file, list_directory, etc.

  # GitHub (repos, issues, PRs)
  async with MCPStdioPlugin(
      name="github",
      command="npx",
      args=["-y", "@modelcontextprotocol/server-github"],
      env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_TOKEN"]},
  ) as gh_plugin:
      kernel.add_plugin(gh_plugin)
      # Agent can now search_repos, get_issue, create_pr, etc.

  # Remote MCP server over SSE
  from semantic_kernel.connectors.mcp import MCPSsePlugin
  async with MCPSsePlugin(
      name="remote_tools",
      url="http://my-mcp-server.internal/sse",
  ) as remote_plugin:
      kernel.add_plugin(remote_plugin)
    """)


# ---------------------------------------------------------------------------
# 5. SK as MCP Server — expose your plugins (reference)
# ---------------------------------------------------------------------------
def sk_as_mcp_server_reference():
    print("\n--- 5. SK as MCP Server (reference) ---")
    print("""
  You can expose your SK plugins AS an MCP server so other MCP clients
  (Claude Desktop, other SK apps, LangChain, etc.) can use them.

  from mcp.server.fastmcp import FastMCP
  from semantic_kernel.functions import kernel_function

  # 1. Build your SK plugin normally
  class MyPlugin:
      @kernel_function(name="greet", description="Greet a user.")
      def greet(self, name: str) -> str:
          return f"Hello, {name}!"

  # 2. Wrap it in FastMCP and re-export each function as an MCP tool
  mcp = FastMCP("my-sk-server")

  plugin = MyPlugin()

  @mcp.tool()
  def greet(name: str) -> str:
      "Greet a user by name."
      return plugin.greet(name=name)

  # 3. Run as stdio server (Claude Desktop / other clients can connect)
  if __name__ == "__main__":
      mcp.run()   # listens on stdio

  # Use case: build SK plugins once, share them across Claude Desktop,
  # other agent frameworks, and internal tools — all via MCP.
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    mcp_overview()
    await mcp_client_example()
    await mcp_resources_example()
    real_world_servers_reference()
    sk_as_mcp_server_reference()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • MCP = universal plug-in standard (tools / resources / prompts)")
    print("  • MCPStdioPlugin: wrap any MCP server as an SK plugin")
    print("  • MCPSsePlugin:   connect to remote MCP servers over HTTP/SSE")
    print("  • MCP tools behave identically to native @kernel_function")
    print("  • Resources: read-only data (files, DB, config) via URI")
    print("  • Real servers: filesystem, GitHub, postgres, brave-search, ...")
    print("  • SK as MCP server: expose your plugins to other MCP clients")
    print("  • Next: Lesson 12 — Capstone (real-world pipeline)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
