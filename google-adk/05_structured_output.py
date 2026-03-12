"""
Lesson 05 — Structured Output
==============================
Concepts covered:
  - output_schema  : force the agent to reply with a Pydantic model (JSON)
  - output_key     : automatically save the structured reply into session state
  - Reading state  : access stored values via session.state[key]
  - The constraint : output_schema disables tools — the agent can only reply

Key insight:
  Structured output is how you turn a conversational agent into a reliable
  data-extraction pipeline. The model is constrained to produce valid JSON
  matching your schema — no free-form prose, no tool calls.
  Think of it as the ADK equivalent of OpenAI's "response_format=JSON".
"""

import asyncio
import json
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_turn(runner: Runner, session_id: str, user_message: str):
    """Send a message and return the final event (for structured output we need
    the raw event to access the parsed Pydantic object)."""
    content = types.Content(
        role="user", parts=[types.Part(text=user_message)]
    )
    final_event = None
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            final_event = event
    return final_event


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Basic structured output: movie review extraction
# ══════════════════════════════════════════════════════════════════════════════

class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre (e.g. Drama, Sci-Fi, Comedy)")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One-sentence summary of the review")
    recommended: bool = Field(description="Whether the reviewer recommends the movie")


async def demo_basic_structured_output():
    print("── Part 1: Structured output — MovieReview ──\n")

    # output_schema forces the agent to reply with MovieReview JSON
    # NOTE: tools are not allowed when output_schema is set
    agent = Agent(
        name="review_extractor",
        model="gemini-2.5-flash",
        instruction=(
            "You extract structured movie review information from unstructured text. "
            "Fill all fields accurately based on the review provided."
        ),
        output_schema=MovieReview,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="05_part1", session_service=session_service)
    session = await session_service.create_session(app_name="05_part1", user_id="learner")

    review_text = """
    I just watched Interstellar (2014) last night. As a science-fiction fan,
    I was blown away. The visual effects are stunning, the score by Hans Zimmer
    is haunting, and the story about time and love is deeply moving.
    A few scenes dragged on a bit, but overall it's a masterpiece.
    I'd give it a solid 9 out of 10 — absolutely worth watching.
    """

    print(f"Input review:\n{review_text.strip()}\n")

    event = await run_turn(runner, session.id, review_text)

    # The structured reply is in event.content.parts[0].text as JSON
    if event and event.content and event.content.parts:
        raw_json = event.content.parts[0].text
        review = MovieReview.model_validate_json(raw_json)
        print(f"Extracted MovieReview:")
        print(f"  title      : {review.title}")
        print(f"  year       : {review.year}")
        print(f"  genre      : {review.genre}")
        print(f"  rating     : {review.rating}/10")
        print(f"  recommended: {review.recommended}")
        print(f"  summary    : {review.summary}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — output_key: auto-save to session state
# ══════════════════════════════════════════════════════════════════════════════
#
# output_key tells ADK to write the agent's reply into session.state[key].
# This is how you pass data between agents in a pipeline (lesson 07+).

class TaskList(BaseModel):
    tasks: list[str] = Field(description="List of action items extracted from the text")
    priority: str = Field(description="Overall priority: low, medium, or high")
    deadline: Optional[str] = Field(default=None, description="Deadline if mentioned, else null")


async def demo_output_key():
    print("── Part 2: output_key — auto-save to session state ──\n")

    agent = Agent(
        name="task_extractor",
        model="gemini-2.5-flash",
        instruction="Extract a structured task list from the text provided.",
        output_schema=TaskList,
        output_key="extracted_tasks",   # ← saved to session.state["extracted_tasks"]
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="05_part2", session_service=session_service)
    session = await session_service.create_session(app_name="05_part2", user_id="learner")

    message = """
    Meeting notes from today: We urgently need to fix the login bug before Friday.
    Also, update the README, write unit tests for the payment module,
    and schedule a code review with the team next week.
    """

    print(f"Input:\n{message.strip()}\n")
    await run_turn(runner, session.id, message)

    # Retrieve the session and read the saved state
    stored = await session_service.get_session(
        app_name="05_part2", user_id="learner", session_id=session.id
    )

    raw = stored.state.get("extracted_tasks")
    if raw:
        task_list = TaskList.model_validate_json(raw) if isinstance(raw, str) else TaskList.model_validate(raw)
        print(f"Session state['extracted_tasks']:")
        print(f"  priority : {task_list.priority}")
        print(f"  deadline : {task_list.deadline}")
        print(f"  tasks    :")
        for t in task_list.tasks:
            print(f"    - {t}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Nested Pydantic models
# ══════════════════════════════════════════════════════════════════════════════

class Address(BaseModel):
    street: str
    city: str
    country: str

class ContactCard(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None
    tags: list[str] = Field(default_factory=list, description="Labels like 'client', 'vendor'")


async def demo_nested_schema():
    print("── Part 3: Nested Pydantic schema ──\n")

    agent = Agent(
        name="contact_extractor",
        model="gemini-2.5-flash",
        instruction="Extract a structured contact card from the text.",
        output_schema=ContactCard,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="05_part3", session_service=session_service)
    session = await session_service.create_session(app_name="05_part3", user_id="learner")

    text = """
    Please add Sarah Chen to our vendor list. She's the main contact at TechSupply Co.
    Her email is sarah.chen@techsupply.com, phone +1-415-555-0192.
    They're based at 200 Market Street, San Francisco, USA.
    """

    print(f"Input:\n{text.strip()}\n")
    event = await run_turn(runner, session.id, text)

    if event and event.content and event.content.parts:
        raw_json = event.content.parts[0].text
        contact = ContactCard.model_validate_json(raw_json)
        print(f"Extracted ContactCard:")
        print(f"  name   : {contact.name}")
        print(f"  email  : {contact.email}")
        print(f"  phone  : {contact.phone}")
        print(f"  tags   : {contact.tags}")
        if contact.address:
            print(f"  address: {contact.address.street}, {contact.address.city}, {contact.address.country}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 05 — Structured Output ===\n")
    await demo_basic_structured_output()
    await demo_output_key()
    await demo_nested_schema()


if __name__ == "__main__":
    asyncio.run(main())
