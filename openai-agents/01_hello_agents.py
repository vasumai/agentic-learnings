"""
Lesson 01 — Hello Agents
========================
Covers:
  - Creating an Agent (name, instructions, model)
  - Runner.run() — the async entry point for every run
  - Single-turn Q&A
  - Multi-turn conversation using to_input_list()

Key concept:
  The OpenAI Agents SDK is built around two primitives:
    Agent  — a configured LLM persona (instructions + model + tools)
    Runner — executes the agent against a list of messages

  `Runner.run()` returns a `RunResult`. To continue a conversation,
  call `result.to_input_list()` which packages the full history
  (original input + all agent messages) ready for the next turn.
"""

import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()


# ---------------------------------------------------------------------------
# 1. A simple single-turn agent
# ---------------------------------------------------------------------------

async def single_turn():
    print("=" * 50)
    print("PART 1: Single-turn Q&A")
    print("=" * 50)

    agent = Agent(
        name="Tutor",
        instructions=(
            "You are a friendly Python tutor. "
            "Give concise, clear answers with a short code example when helpful."
        ),
        model="gpt-4o-mini",
    )

    result = await Runner.run(agent, input="What is a list comprehension in Python?")

    print(f"Agent: {result.final_output}")


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation using to_input_list()
# ---------------------------------------------------------------------------

async def multi_turn():
    print("\n" + "=" * 50)
    print("PART 2: Multi-turn conversation")
    print("=" * 50)

    agent = Agent(
        name="Tutor",
        instructions=(
            "You are a friendly Python tutor. "
            "Remember what the student asked before and build on it."
        ),
        model="gpt-4o-mini",
    )

    # Turn 1
    user_msg_1 = "What is a decorator in Python?"
    print(f"\nUser: {user_msg_1}")
    result1 = await Runner.run(agent, input=user_msg_1)
    print(f"Agent: {result1.final_output}")

    # Turn 2 — extend history with to_input_list()
    user_msg_2 = "Can you show me a simple real-world example of one?"
    print(f"\nUser: {user_msg_2}")
    result2 = await Runner.run(
        agent,
        input=result1.to_input_list() + [{"role": "user", "content": user_msg_2}],
    )
    print(f"Agent: {result2.final_output}")

    # Turn 3 — keep building the history
    user_msg_3 = "How is that different from a context manager?"
    print(f"\nUser: {user_msg_3}")
    result3 = await Runner.run(
        agent,
        input=result2.to_input_list() + [{"role": "user", "content": user_msg_3}],
    )
    print(f"Agent: {result3.final_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await single_turn()
    await multi_turn()


if __name__ == "__main__":
    asyncio.run(main())
