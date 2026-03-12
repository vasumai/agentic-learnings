"""
Lesson 04 — Multi-turn Conversation
=====================================
Concepts covered:
  - Sessions as conversation threads — each session holds its own history
  - Multiple turns within one session — the agent remembers prior messages
  - Multiple sessions in parallel — completely isolated from each other
  - Inspecting session history via session_service.get_session()

Key insight:
  ADK's SessionService is what gives the agent memory within a conversation.
  Each session is identified by (app_name, user_id, session_id).
  Two sessions with different IDs have zero shared context — even for the
  same user_id. This is how you run isolated conversations in parallel.
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── Agent (shared across all demos) ───────────────────────────────────────────

agent = Agent(
    name="chat_agent",
    model="gemini-2.5-flash",
    description="A conversational assistant that remembers what was said earlier.",
    instruction=(
        "You are a helpful assistant. "
        "Refer to earlier parts of the conversation when relevant. "
        "Keep answers concise."
    ),
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="multi_turn", session_service=session_service)


# ── Helper ─────────────────────────────────────────────────────────────────────

async def send(session_id: str, user_message: str) -> str:
    """Send one message in a session and return the agent's reply."""
    content = types.Content(
        role="user", parts=[types.Part(text=user_message)]
    )
    reply = ""
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = event.content.parts[0].text
    return reply


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Multi-turn memory: agent recalls earlier messages
# ══════════════════════════════════════════════════════════════════════════════

async def demo_multi_turn_memory():
    print("── Part 1: Multi-turn memory ──\n")

    session = await session_service.create_session(
        app_name="multi_turn", user_id="learner"
    )
    sid = session.id

    # Turn 1 — introduce a preference
    q1 = "My favourite programming language is Python."
    print(f"You  : {q1}")
    print(f"Agent: {await send(sid, q1)}\n")

    # Turn 2 — ask something unrelated
    q2 = "What are the benefits of using virtual environments?"
    print(f"You  : {q2}")
    print(f"Agent: {await send(sid, q2)}\n")

    # Turn 3 — refer back to turn 1 implicitly
    q3 = "Given what I told you about my favourite language, what framework would you recommend for building web APIs?"
    print(f"You  : {q3}")
    print(f"Agent: {await send(sid, q3)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Two isolated sessions: no shared context
# ══════════════════════════════════════════════════════════════════════════════

async def demo_isolated_sessions():
    print("── Part 2: Isolated sessions ──\n")

    # Session A — user prefers Python
    session_a = await session_service.create_session(
        app_name="multi_turn", user_id="learner"
    )
    # Session B — user prefers Go
    session_b = await session_service.create_session(
        app_name="multi_turn", user_id="learner"
    )

    # Seed each session with a different preference
    await send(session_a.id, "I work primarily with Python.")
    await send(session_b.id, "I work primarily with Go.")

    # Ask the same follow-up in both — responses should differ
    question = "What testing framework would you suggest for my work?"

    print(f"You (session A): {question}")
    reply_a = await send(session_a.id, question)
    print(f"Agent (session A): {reply_a}\n")

    print(f"You (session B): {question}")
    reply_b = await send(session_b.id, question)
    print(f"Agent (session B): {reply_b}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Inspect session history
# ══════════════════════════════════════════════════════════════════════════════

async def demo_inspect_history():
    print("── Part 3: Inspect session history ──\n")

    session = await session_service.create_session(
        app_name="multi_turn", user_id="learner"
    )
    sid = session.id

    await send(sid, "My name is Alex.")
    await send(sid, "What is 7 times 8?")

    # Retrieve the session and print its stored events
    stored = await session_service.get_session(
        app_name="multi_turn", user_id="learner", session_id=sid
    )
    print(f"Session ID  : {stored.id}")
    print(f"Total events: {len(stored.events)}\n")

    for i, event in enumerate(stored.events):
        role = event.content.role if event.content else "system"
        text = ""
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    text = part.text[:80]   # truncate for readability
                    break
        if text:
            print(f"  [{i}] {role:10s}: {text}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 04 — Multi-turn Conversation ===\n")
    await demo_multi_turn_memory()
    await demo_isolated_sessions()
    await demo_inspect_history()


if __name__ == "__main__":
    asyncio.run(main())
