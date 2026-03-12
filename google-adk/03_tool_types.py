"""
Lesson 03 — Tool Types
======================
Concepts covered:
  - FunctionTool     : explicit wrapper around a Python callable (ADK does this
                       automatically in lesson 02, but you can be explicit)
  - LongRunningFunctionTool : for operations that stream progress updates while
                       the agent keeps running (e.g. file processing, polling)
  - Built-in tools   : google_search and url_context — powered by Gemini natively
  - The mixing rule  : built-in tools CANNOT be combined with custom Python tools
                       in the same agent — this is a Gemini API limitation (not ADK)

Workaround for mixing:
  Use two separate agents — one with built-in tools, one with custom tools —
  then wire them together with SequentialAgent or sub-agents (lesson 07+).
"""

import asyncio
import time
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, LongRunningFunctionTool, google_search
from google.genai import types

load_dotenv()

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_turn(runner: Runner, session_id: str, user_message: str) -> str:
    content = types.Content(
        role="user", parts=[types.Part(text=user_message)]
    )
    final_reply = ""
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_reply = event.content.parts[0].text
    return final_reply


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — FunctionTool (explicit wrapping)
# ══════════════════════════════════════════════════════════════════════════════
#
# In lesson 02 we passed plain functions directly to `tools=[]`.
# ADK silently wrapped each one in a FunctionTool for us.
# Here we do it explicitly — same result, but you can see the wrapper.

def square(number: int) -> int:
    """Returns the square of a given integer.

    Args:
        number: The integer to square.
    """
    return number * number


# Explicit wrap — identical to passing `square` directly
square_tool = FunctionTool(func=square)


async def demo_function_tool():
    print("── Part 1: FunctionTool (explicit) ──\n")

    agent = Agent(
        name="function_tool_agent",
        model="gemini-2.5-flash",
        instruction="You are a math assistant. Use tools for calculations.",
        tools=[square_tool],           # explicit FunctionTool
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="03_part1", session_service=session_service)
    session = await session_service.create_session(app_name="03_part1", user_id="learner")

    reply = await run_turn(runner, session.id, "What is 12 squared?")
    print(f"You  : What is 12 squared?")
    print(f"Agent: {reply}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — LongRunningFunctionTool
# ══════════════════════════════════════════════════════════════════════════════
#
# LongRunningFunctionTool is a FunctionTool with is_long_running=True.
# ADK injects an instruction into the tool description:
#   "Do not call this tool again if it has already returned some status."
#
# This is ideal for polling/job-check patterns where the tool might return
# a "pending" status before eventually returning the final result.
# The model will NOT keep calling it repeatedly once a status is returned.
#
# The function itself is a plain function (NOT a generator).

def process_dataset(filename: str, rows: int):
    """Processes a dataset file and returns a summary. Takes time for large files.

    Args:
        filename: Name of the dataset file to process.
        rows: Number of rows in the dataset.
    """
    time.sleep(1)  # simulate long work
    return {
        "status": "complete",
        "filename": filename,
        "rows_processed": rows,
        "summary": f"Found {rows // 10} anomalies, avg value: {rows * 1.5:.1f}",
    }


long_running_tool = LongRunningFunctionTool(func=process_dataset)


async def demo_long_running_tool():
    print("── Part 2: LongRunningFunctionTool ──\n")

    agent = Agent(
        name="long_running_agent",
        model="gemini-2.5-flash",
        instruction="You process datasets and summarise the results.",
        tools=[long_running_tool],
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="03_part2", session_service=session_service)
    session = await session_service.create_session(app_name="03_part2", user_id="learner")

    question = "Please process the file sales_data.csv which has 500 rows."
    print(f"You  : {question}")
    print(f"  (tool will take ~1s — model won't re-call it while it runs)")
    reply = await run_turn(runner, session.id, question)
    print(f"Agent: {reply}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Built-in tool: google_search  (ALONE — cannot mix with custom tools)
# ══════════════════════════════════════════════════════════════════════════════
#
# google_search is a GoogleSearchTool — it's handled entirely inside Gemini,
# not in Python. That's why Gemini's API refuses to mix it with Python callables:
# the two tool types go through different execution paths in the model backend.
#
# Rule: one agent = either built-in tools OR custom Python tools, not both.

async def demo_google_search():
    print("── Part 3: Built-in google_search ──\n")
    print("NOTE: google_search runs ALONE — cannot be combined with Python tools.\n")

    agent = Agent(
        name="search_agent",
        model="gemini-2.5-flash",
        instruction="Answer questions using Google Search. Be concise.",
        tools=[google_search],         # built-in only — no custom tools here
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="03_part3", session_service=session_service)
    session = await session_service.create_session(app_name="03_part3", user_id="learner")

    question = "What is the latest stable version of Python as of today?"
    print(f"You  : {question}")
    reply = await run_turn(runner, session.id, question)
    print(f"Agent: {reply}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Show the mixing error clearly
# ══════════════════════════════════════════════════════════════════════════════

async def demo_mixing_error():
    print("── Part 4: Mixing built-in + custom → expected error ──\n")

    def my_tool(x: str) -> str:
        """A custom tool."""
        return x.upper()

    agent = Agent(
        name="mixed_agent",
        model="gemini-2.5-flash",
        instruction="Be helpful.",
        tools=[google_search, my_tool],    # ← this combination is rejected by Gemini
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="03_part4", session_service=session_service)
    session = await session_service.create_session(app_name="03_part4", user_id="learner")

    try:
        await run_turn(runner, session.id, "Hello")
    except Exception as e:
        # Extract just the key message
        msg = str(e)
        if "Built-in tools" in msg:
            start = msg.find("Built-in tools")
            print(f"  ✗ API error: {msg[start:start+90]}")
        else:
            print(f"  ✗ Error: {msg[:120]}")
    print()
    print("Workaround: separate agents — one for built-in tools, one for custom tools.")
    print("Wire them together using SequentialAgent or sub-agents (see lesson 07+).\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 03 — Tool Types ===\n")
    await demo_function_tool()
    await demo_long_running_tool()
    await demo_google_search()
    await demo_mixing_error()


if __name__ == "__main__":
    asyncio.run(main())
