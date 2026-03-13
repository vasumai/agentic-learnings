"""
Lesson 04 — Context
===================
Covers:
  - RunContextWrapper[T] in depth — typed shared state for an entire run
  - Passing context through tools AND handoffs
  - Context is NOT sent to the LLM — it's your app-side state
  - Using context for dependency injection (e.g. a "database" or service)
  - Reading context in on_handoff callbacks

Key concept:
  Context is the SDK's answer to "how do agents share state?"
  You pass ONE context object into Runner.run(context=...).
  Every tool and every on_handoff callback in that run receives it
  via RunContextWrapper[T] as their first argument.

  Important: the LLM never sees the context object.
  It only sees tool return values. Context is purely app-side.
  Use it for: user sessions, DB connections, accumulators, config.
"""

import asyncio
from dataclasses import dataclass, field
from dotenv import load_dotenv
from agents import Agent, Runner, RunContextWrapper, function_tool, handoff

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Context as a user session — tools read and write it
# ---------------------------------------------------------------------------

@dataclass
class UserSession:
    user_id: str
    name: str
    language: str = "English"
    actions_taken: list[str] = field(default_factory=list)


@function_tool
def get_user_profile(ctx: RunContextWrapper[UserSession]) -> str:
    """Return the current user's profile information."""
    u = ctx.context
    return f"User: {u.name} (id={u.user_id}), preferred language: {u.language}"


@function_tool
def set_language(ctx: RunContextWrapper[UserSession], language: str) -> str:
    """Update the user's preferred language.

    Args:
        language: The new preferred language (e.g. 'Spanish', 'French').
    """
    ctx.context.language = language
    ctx.context.actions_taken.append(f"language changed to {language}")
    return f"Preferred language updated to {language}."


@function_tool
def log_action(ctx: RunContextWrapper[UserSession], action: str) -> str:
    """Log an action taken during this session.

    Args:
        action: A short description of what the user did.
    """
    ctx.context.actions_taken.append(action)
    return f"Logged: '{action}'"


async def session_context_demo():
    print("=" * 50)
    print("PART 1: Context as a user session")
    print("=" * 50)

    session = UserSession(user_id="u-42", name="Srini")

    agent = Agent(
        name="ProfileAgent",
        instructions=(
            "You are a helpful account assistant. "
            "Use tools to look up and update the user's profile. "
            "Be concise."
        ),
        model="gpt-4o-mini",
        tools=[get_user_profile, set_language, log_action],
    )

    questions = [
        "What does my profile look like?",
        "Change my language to Spanish.",
        "Log that I reviewed my account settings.",
    ]

    result = None
    for q in questions:
        print(f"\nUser: {q}")
        history = result.to_input_list() if result else []
        result = await Runner.run(
            agent,
            input=history + [{"role": "user", "content": q}],
            context=session,
        )
        print(f"Agent: {result.final_output}")

    print(f"\n[Session state after run]")
    print(f"  language     : {session.language}")
    print(f"  actions_taken: {session.actions_taken}")


# ---------------------------------------------------------------------------
# 2. Context flows through handoffs — specialist agents see the same object
# ---------------------------------------------------------------------------

@dataclass
class OrderContext:
    order_id: str
    customer: str
    items: list[str]
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


@function_tool
def get_order_details(ctx: RunContextWrapper[OrderContext]) -> str:
    """Return the full details of the current order."""
    o = ctx.context
    return (
        f"Order #{o.order_id} for {o.customer}\n"
        f"Items: {', '.join(o.items)}\n"
        f"Status: {o.status}"
    )


@function_tool
def update_order_status(ctx: RunContextWrapper[OrderContext], new_status: str) -> str:
    """Update the order status.

    Args:
        new_status: The new status (e.g. 'shipped', 'cancelled', 'refunded').
    """
    old = ctx.context.status
    ctx.context.status = new_status
    ctx.context.notes.append(f"status changed: {old} → {new_status}")
    return f"Order status updated to '{new_status}'."


@function_tool
def add_order_note(ctx: RunContextWrapper[OrderContext], note: str) -> str:
    """Add an internal note to the order.

    Args:
        note: The note text to attach to the order.
    """
    ctx.context.notes.append(note)
    return f"Note added: '{note}'"


def on_handoff_to_shipping(ctx: RunContextWrapper[OrderContext]):
    print(f"  [handoff → ShippingAgent] order #{ctx.context.order_id}")

def on_handoff_to_refunds(ctx: RunContextWrapper[OrderContext]):
    print(f"  [handoff → RefundsAgent] order #{ctx.context.order_id}")


shipping_agent = Agent(
    name="ShippingAgent",
    instructions=(
        "You handle shipping and delivery questions. "
        "Use tools to check and update order details. Be concise."
    ),
    model="gpt-4o-mini",
    tools=[get_order_details, update_order_status, add_order_note],
)

refunds_agent = Agent(
    name="RefundsAgent",
    instructions=(
        "You handle refund and cancellation requests. "
        "Use tools to update order status and add notes. Be concise."
    ),
    model="gpt-4o-mini",
    tools=[get_order_details, update_order_status, add_order_note],
)

order_triage_agent = Agent(
    name="OrderTriageAgent",
    instructions=(
        "You are an order support triage agent. Route to the right specialist.\n"
        "- Shipping / delivery questions → ShippingAgent\n"
        "- Refunds / cancellations → RefundsAgent\n"
        "Do not answer yourself — always hand off."
    ),
    model="gpt-4o-mini",
    handoffs=[
        handoff(shipping_agent, on_handoff=on_handoff_to_shipping),
        handoff(refunds_agent,  on_handoff=on_handoff_to_refunds),
    ],
)


async def context_through_handoffs_demo():
    print("\n" + "=" * 50)
    print("PART 2: Context flows through handoffs")
    print("=" * 50)

    order = OrderContext(
        order_id="ORD-999",
        customer="Srini",
        items=["Laptop Stand", "USB Hub", "Mechanical Keyboard"],
    )

    queries = [
        ("shipping", "Where is my order? Can you mark it as shipped?"),
        ("refund",   "I want to cancel and get a refund for order ORD-999."),
    ]

    for label, q in queries:
        print(f"\nUser ({label}): {q}")
        result = await Runner.run(order_triage_agent, input=q, context=order)
        print(f"  Answered by : {result.last_agent.name}")
        print(f"  Answer      : {result.final_output}")

    print(f"\n[Order context after both runs]")
    print(f"  status : {order.status}")
    print(f"  notes  : {order.notes}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await session_context_demo()
    await context_through_handoffs_demo()


if __name__ == "__main__":
    asyncio.run(main())
