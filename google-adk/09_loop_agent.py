"""
Lesson 09 — LoopAgent
======================
Concepts covered:
  - LoopAgent: runs sub-agents in a loop until a stop condition is met
  - Two termination mechanisms:
      1. max_iterations — hard cap (safety net, always set this)
      2. exit_loop tool — agent calls this tool when done (clean termination)
  - How to pass state between iterations via session state
  - Combining LoopAgent with SequentialAgent for "refine until good enough"

Key insight:
  LoopAgent is ADK's while-loop primitive. The loop continues until either:
    a) The agent calls the `exit_loop` tool (sets actions.escalate = True)
    b) max_iterations is reached (safety net)
  Always set max_iterations — without it, a buggy agent loops forever.

  Pattern 1 — single agent loop: one agent runs repeatedly, improving output
  Pattern 2 — multi-agent loop: checker → worker → checker → worker...

Demos:
  1. Code review loop: a reviewer iterates on code until quality passes
  2. Data cleaning loop: an agent cleans rows until no errors remain
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import exit_loop
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


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 1 — Code review loop
#
# Pattern: LoopAgent wraps a single "writer+reviewer" agent.
# Each iteration: agent rewrites the code AND self-evaluates quality.
# When quality meets the bar → agent calls exit_loop → loop stops.
# Safety net: max_iterations=4
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_CODE = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    avg = total / len(numbers)
    return avg
"""

# Writer: improves the code one step and saves it (no exit_loop — output_key works)
code_writer = Agent(
    name="code_writer",
    model="gemini-2.5-flash",
    description="Applies ONE improvement to the Python code each iteration.",
    instruction="""You are a Python refactoring specialist.

Look at the CURRENT version of the code in session state key 'current_code'
(or the original code if first iteration). Apply exactly ONE improvement:
  - Add type hints, OR
  - Add a docstring, OR
  - Add error handling (e.g. empty list, division by zero), OR
  - Replace verbose loops with sum() or similar builtins.

Output ONLY the complete improved function (no explanation).
""",
    output_key="current_code",   # saves output → state["current_code"]
)

# Checker: evaluates quality and calls exit_loop when done
# Does NOT use output_key — exit_loop sets skip_summarization so no text stored
code_checker = Agent(
    name="code_checker",
    model="gemini-2.5-flash",
    description="Checks code quality and exits the loop when standards are met.",
    instruction="""You are a Python code quality checker.

Review the code in session state key 'current_code'.
Check whether it has ALL of:
  ✓ Type hints on parameters and return value
  ✓ A docstring
  ✓ Error handling for edge cases (empty list, zero division, etc.)
  ✓ Pythonic style (no verbose manual loops if a builtin works)

If ALL four are present → call exit_loop (you are done).
If anything is missing → respond with a brief note on what's still needed.
Do NOT call exit_loop if anything is missing.
""",
    tools=[exit_loop],
)

code_loop = LoopAgent(
    name="code_review_loop",
    description="Writer improves code, checker evaluates — loop until all criteria met.",
    sub_agents=[code_writer, code_checker],
    max_iterations=4,   # safety net
)


async def demo_code_review_loop():
    print("── Demo 1: Code review loop ──\n")
    print(f"Original code:\n{INITIAL_CODE}")

    session_service = InMemorySessionService()
    runner = Runner(agent=code_loop, app_name="09_demo1", session_service=session_service)
    session = await session_service.create_session(app_name="09_demo1", user_id="learner")

    await run_pipeline(
        runner, session.id,
        f"Please improve this Python function iteratively:\n{INITIAL_CODE}"
    )

    stored = await session_service.get_session(
        app_name="09_demo1", user_id="learner", session_id=session.id
    )

    # Count how many iterations ran (events from the improver agent)
    iterations = sum(
        1 for e in stored.events
        if e.author == "code_checker" and e.content and e.content.role == "model"
    )
    print(f"Loop ran {iterations} iteration(s)\n")
    print(f"Final code (from state['current_code']):\n")
    print(stored.state.get("current_code", "N/A"))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 2 — Two-agent checker/fixer loop
#
# Pattern: SequentialAgent wrapping a LoopAgent that has TWO sub-agents:
#   [FixerAgent → CheckerAgent] × N
#
# CheckerAgent validates the data. If OK → calls exit_loop. If not → continues.
# FixerAgent reads the checker's feedback and corrects the data.
# ══════════════════════════════════════════════════════════════════════════════

DIRTY_DATA = """
Name,Age,Email,Score
Alice,25,alice@example.com,87
Bob,-5,not-an-email,92
Charlie,150,charlie@example.com,105
Diana,30,diana@example.com,78
Eve,22,,88
"""

fixer_agent = Agent(
    name="fixer_agent",
    model="gemini-2.5-flash",
    description="Fixes data quality issues in CSV data.",
    instruction="""You are a data cleaning specialist.

The CSV data to clean is in session state key 'data_to_clean' (or the original input).
The checker's latest feedback is in session state key 'checker_feedback' (if present).

Fix ALL the issues mentioned in the feedback:
  - Age must be between 0 and 120
  - Email must be a valid format (contains @ and .)
  - Score must be between 0 and 100
  - Missing values should be replaced with sensible defaults

Output ONLY the corrected CSV (with header). No explanation.
""",
    output_key="data_to_clean",
)

checker_agent = Agent(
    name="checker_agent",
    model="gemini-2.5-flash",
    description="Validates CSV data quality and signals done when clean.",
    instruction="""You are a data quality validator.

Check the CSV data in session state key 'data_to_clean' for:
  - Age: must be 0-120 (invalid if negative or > 120)
  - Email: must contain @ and a dot after @ (invalid if missing or malformed)
  - Score: must be 0-100 (invalid if > 100 or < 0)
  - No missing values

If ALL rows pass ALL checks:
  → Say "PASSED: data is clean." then call exit_loop.

If any issues remain:
  → List each issue clearly (row, field, problem). Do NOT call exit_loop.
  → Be specific so the fixer can correct them.
""",
    tools=[exit_loop],
    output_key="checker_feedback",
)

data_cleaning_loop = LoopAgent(
    name="data_cleaning_loop",
    description="Alternates between fixer and checker until data is clean.",
    sub_agents=[fixer_agent, checker_agent],
    max_iterations=3,   # each iteration = one fixer + one checker pass
)


async def demo_data_cleaning_loop():
    print("── Demo 2: Fixer/Checker loop ──\n")
    print(f"Dirty data:\n{DIRTY_DATA}")

    session_service = InMemorySessionService()
    runner = Runner(agent=data_cleaning_loop, app_name="09_demo2", session_service=session_service)
    session = await session_service.create_session(app_name="09_demo2", user_id="learner")

    await run_pipeline(runner, session.id, f"Clean this CSV:\n{DIRTY_DATA}")

    stored = await session_service.get_session(
        app_name="09_demo2", user_id="learner", session_id=session.id
    )

    iterations = sum(
        1 for e in stored.events
        if e.author == "checker_agent" and e.content and e.content.role == "model"
    )
    print(f"Loop ran {iterations} checker pass(es)\n")

    print("Final clean data (from state['data_to_clean']):\n")
    print(stored.state.get("data_to_clean", "N/A"))

    print("\nChecker's final verdict (from state['checker_feedback']):\n")
    feedback = stored.state.get("checker_feedback", "N/A")
    print(feedback[:300] if len(str(feedback)) > 300 else feedback)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 09 — LoopAgent ===\n")
    await demo_code_review_loop()
    await demo_data_cleaning_loop()


if __name__ == "__main__":
    asyncio.run(main())
