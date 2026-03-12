"""
Lesson 02 — Tools
=================
Concepts covered:
  - Defining tools as plain Python functions (no decorator needed)
  - Docstrings are the tool description — ADK reads them automatically
  - Type hints tell ADK the parameter schema
  - Automatic tool calling — ADK decides when to call a tool and feeds
    the result back into the conversation without any manual wiring

Key insight:
  Unlike LangChain (@tool decorator) or SK (kernel.add_function),
  ADK tools are just regular Python functions. The docstring IS the
  description the model uses to decide whether to call the tool.
"""

import asyncio
import random
from datetime import datetime, timezone as tz
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── 1. Define tools as plain Python functions ──────────────────────────────────
#
# Rules:
#   - Must have a docstring (this is the tool description for the model)
#   - Must have type-annotated parameters (ADK builds the schema from these)
#   - Must return a string or dict (dict gets JSON-serialised)

def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current date and time.

    Args:
        timezone: The timezone name (e.g. 'UTC', 'US/Eastern'). Defaults to UTC.
    """
    now = datetime.now(tz.utc)
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


def get_weather(city: str) -> dict:
    """Returns the current weather for a given city.

    Args:
        city: The name of the city to get weather for (e.g. 'London', 'Tokyo').
    """
    # Simulated weather data — in a real app you'd call a weather API here
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly cloudy", "Windy"]
    temp_c = random.randint(5, 35)
    condition = random.choice(conditions)
    return {
        "city": city,
        "temperature_celsius": temp_c,
        "temperature_fahrenheit": round(temp_c * 9 / 5 + 32, 1),
        "condition": condition,
        "humidity_percent": random.randint(30, 90),
    }


def calculate(expression: str) -> str:
    """Evaluates a simple mathematical expression and returns the result.

    Args:
        expression: A mathematical expression as a string (e.g. '2 + 2', '10 * 3.5').
    """
    try:
        # Restrict to safe math operations only
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: only basic arithmetic is supported (+, -, *, /, parentheses)"
        result = eval(expression)  # noqa: S307 — safe: restricted character set above
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ── 2. Create the agent — pass tools as a list ─────────────────────────────────
#
# ADK inspects each function's signature and docstring to build
# the tool spec sent to the model. No wrapper classes needed.

agent = Agent(
    name="tool_agent",
    model="gemini-2.5-flash",
    description="An assistant that can check the time, weather, and do math.",
    instruction=(
        "You are a helpful assistant with access to tools. "
        "Use tools when you need real data — don't make up answers."
    ),
    tools=[get_current_time, get_weather, calculate],
)

# ── 3. Runner + session (same pattern as lesson 01) ────────────────────────────

session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="tools_demo",
    session_service=session_service,
)


async def run_turn(session_id: str, user_message: str) -> str:
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )
    final_reply = ""
    async for event in runner.run_async(
        user_id="learner",
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_reply = event.content.parts[0].text
    return final_reply


async def main():
    session = await session_service.create_session(
        app_name="tools_demo",
        user_id="learner",
    )
    session_id = session.id

    print("=== Lesson 02 — Tools ===\n")

    questions = [
        "What's the weather like in Tokyo?",
        "What is 123 * 456?",
    ]

    for i, question in enumerate(questions):
        if i > 0:
            await asyncio.sleep(20)  # respect free-tier 5 RPM limit
        print(f"You  : {question}")
        reply = await run_turn(session_id, question)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
