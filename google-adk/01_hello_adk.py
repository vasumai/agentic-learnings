"""
Lesson 01 — Hello ADK
=====================
Concepts covered:
  - The three core ADK objects: Agent, Runner, InMemorySessionService
  - Creating a minimal agent with a system instruction
  - Running a single turn and printing the response

Key insight:
  ADK separates WHAT the agent is (Agent) from HOW it runs (Runner)
  and WHERE state lives (SessionService). You always need all three.
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── 1. Define the agent ────────────────────────────────────────────────────────
#
# Agent = the "who" — model, personality, and instructions.
# No tools yet; we'll add those in lesson 02.

agent = Agent(
    name="hello_agent",
    model="gemini-2.5-flash",          # ADK default model via GOOGLE_API_KEY
    description="A friendly assistant that answers questions concisely.",
    instruction="You are a helpful assistant. Keep your answers brief and clear.",
)

# ── 2. Session service ─────────────────────────────────────────────────────────
#
# SessionService stores conversation history (turns).
# InMemorySessionService = ephemeral, great for dev/learning.
# Swap for DatabaseSessionService in production.

session_service = InMemorySessionService()

# ── 3. Runner ──────────────────────────────────────────────────────────────────
#
# Runner = the "how" — wires the agent to a session service and handles
# the run loop: sending messages, receiving events, calling tools.

runner = Runner(
    agent=agent,
    app_name="hello_adk",              # logical app name (used in session IDs)
    session_service=session_service,
)


# ── 4. Send a message and collect the reply ────────────────────────────────────

async def run_turn(session_id: str, user_message: str) -> str:
    """Send one user message and return the agent's text reply."""
    # ADK uses google.genai Content/Part for messages
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )

    final_reply = ""
    # runner.run_async streams events; we collect the last text response
    async for event in runner.run_async(
        user_id="learner",
        session_id=session_id,
        new_message=content,
    ):
        # is_final_response() marks the agent's finished output
        if event.is_final_response():
            if event.content and event.content.parts:
                final_reply = event.content.parts[0].text

    return final_reply


async def main():
    # Create a fresh session (think of it as a conversation thread)
    session = await session_service.create_session(
        app_name="hello_adk",
        user_id="learner",
    )
    session_id = session.id

    print("=== Lesson 01 — Hello ADK ===\n")

    questions = [
        "What is Google ADK in one sentence?",
        "Why would I use ADK instead of calling the Gemini API directly?",
    ]

    for question in questions:
        print(f"You : {question}")
        reply = await run_turn(session_id, question)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
