"""
Lesson 06 — Callbacks
======================
Concepts covered:
  - before_agent_callback  : runs before the agent starts — can short-circuit it
  - after_agent_callback   : runs after the agent finishes
  - before_model_callback  : runs before the LLM is called — can modify the prompt
                             or block the call entirely
  - before_tool_callback   : runs before each tool call — can block or override it
  - after_tool_callback    : runs after each tool call — can modify the result

Key insight:
  Callbacks are ADK's interception layer — equivalent to middleware in web
  frameworks, SK filters, or LangGraph interrupt/resume. They give you hooks
  to log, validate, block, or transform at every stage without changing agent logic.

  Return None  → proceed normally
  Return a value → short-circuit (skip the normal step, use this instead)
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool
from google.genai import types

load_dotenv()

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_turn(runner: Runner, session_id: str, message: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    reply = ""
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = event.content.parts[0].text
    return reply


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — before_agent_callback + after_agent_callback
#
# before_agent_callback: (callback_context) → types.Content | None
#   Return None  → agent runs normally
#   Return Content → skip the agent entirely, use this as the reply
#
# after_agent_callback: (callback_context) → types.Content | None
#   Called after the agent finishes — return value is ignored in current ADK
# ══════════════════════════════════════════════════════════════════════════════

def before_agent(callback_context: CallbackContext) -> types.Content | None:
    """Log that the agent is starting. Optionally block based on session state."""
    user_msg = ""
    if callback_context.user_content and callback_context.user_content.parts:
        user_msg = callback_context.user_content.parts[0].text[:60]
    print(f"  [before_agent] agent='{callback_context.agent_name}' message='{user_msg}...'")

    # Example guard: block if session state has a 'banned' flag
    if callback_context.state.get("banned"):
        print("  [before_agent] BLOCKED — user is banned")
        return types.Content(
            role="model",
            parts=[types.Part(text="Sorry, your account has been suspended.")],
        )
    return None  # proceed normally


def after_agent(callback_context: CallbackContext) -> types.Content | None:
    print(f"  [after_agent]  agent='{callback_context.agent_name}' finished")
    return None


async def demo_agent_callbacks():
    print("── Part 1: before_agent_callback / after_agent_callback ──\n")

    agent = Agent(
        name="guarded_agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant.",
        before_agent_callback=before_agent,
        after_agent_callback=after_agent,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="06_part1", session_service=session_service)

    # Normal turn
    session = await session_service.create_session(app_name="06_part1", user_id="learner")
    reply = await run_turn(runner, session.id, "What is the capital of France?")
    print(f"  Agent: {reply}\n")

    # Turn with 'banned' flag in session state
    session2 = await session_service.create_session(
        app_name="06_part1", user_id="learner", state={"banned": True}
    )
    reply2 = await run_turn(runner, session2.id, "Tell me something interesting.")
    print(f"  Agent (banned session): {reply2}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — before_model_callback
#
# Signature: (callback_context, llm_request) → LlmResponse | None
#   Return None       → LLM call proceeds with (possibly modified) llm_request
#   Return LlmResponse → skip the LLM call, use this response instead
#
# Use cases: prompt injection guard, request logging, adding system context
# ══════════════════════════════════════════════════════════════════════════════

BLOCKED_WORDS = {"password", "secret", "creditcard"}
BLOCKED_PHRASES = ["credit card", "social security", "bank account"]

def before_model(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    """Block requests containing sensitive keywords."""
    last_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                last_text = content.parts[0].text.lower()
                break

    flagged = (BLOCKED_WORDS & set(last_text.split())) or \
              [p for p in BLOCKED_PHRASES if p in last_text]
    if flagged:
        print(f"  [before_model] BLOCKED — sensitive words: {flagged}")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="I can't process requests containing sensitive information.")],
            )
        )

    print(f"  [before_model] request approved — forwarding to LLM")
    return None  # proceed


async def demo_model_callback():
    print("── Part 2: before_model_callback — prompt guard ──\n")

    agent = Agent(
        name="safe_agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant.",
        before_model_callback=before_model,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="06_part2", session_service=session_service)
    session = await session_service.create_session(app_name="06_part2", user_id="learner")

    # Safe message
    reply = await run_turn(runner, session.id, "What is the tallest mountain?")
    print(f"  Agent: {reply}\n")

    # Blocked message
    reply2 = await run_turn(runner, session.id, "What is my credit card number?")
    print(f"  Agent: {reply2}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — before_tool_callback + after_tool_callback
#
# before_tool_callback: (tool, args, tool_context) → dict | None
#   Return None → tool runs normally with original args
#   Return dict → skip tool execution, use this dict as the tool result
#
# after_tool_callback: (tool, args, tool_context, result) → dict | None
#   Return None → original result is used
#   Return dict → replace the tool result with this
# ══════════════════════════════════════════════════════════════════════════════

def get_price(product: str) -> dict:
    """Returns the current price of a product.

    Args:
        product: The product name to look up (e.g. 'laptop', 'phone').
    """
    prices = {"laptop": 999.99, "phone": 699.99, "tablet": 499.99}
    price = prices.get(product.lower(), 0)
    return {"product": product, "price_usd": price, "in_stock": price > 0}


def before_tool(
    tool: BaseTool, args: dict, tool_context: ToolContext
) -> dict | None:
    """Log tool calls and inject a cache hit for known products."""
    print(f"  [before_tool] '{tool.name}' called with args={args}")

    # Example: serve from cache without hitting the tool
    if tool.name == "get_price" and args.get("product", "").lower() == "laptop":
        print(f"  [before_tool] cache hit — returning cached result")
        return {"product": "laptop", "price_usd": 899.99, "in_stock": True, "source": "cache"}

    return None  # proceed normally


def after_tool(
    tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict
) -> dict | None:
    """Add a timestamp to every tool result."""
    from datetime import datetime, timezone as tz
    tool_response["fetched_at"] = datetime.now(tz.utc).strftime("%H:%M:%S UTC")
    print(f"  [after_tool]  '{tool.name}' result augmented with timestamp")
    return tool_response  # return modified result


async def demo_tool_callbacks():
    print("── Part 3: before_tool_callback / after_tool_callback ──\n")

    agent = Agent(
        name="shop_agent",
        model="gemini-2.5-flash",
        instruction="You are a shopping assistant. Use the get_price tool to look up prices.",
        tools=[get_price],
        before_tool_callback=before_tool,
        after_tool_callback=after_tool,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="06_part3", session_service=session_service)
    session = await session_service.create_session(app_name="06_part3", user_id="learner")

    # laptop → cache hit (before_tool returns early)
    reply = await run_turn(runner, session.id, "How much does a laptop cost?")
    print(f"  Agent: {reply}\n")

    # phone → normal tool call + after_tool augmentation
    reply2 = await run_turn(runner, session.id, "What about a phone?")
    print(f"  Agent: {reply2}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 06 — Callbacks ===\n")
    await demo_agent_callbacks()
    await demo_model_callback()
    await demo_tool_callbacks()


if __name__ == "__main__":
    asyncio.run(main())
