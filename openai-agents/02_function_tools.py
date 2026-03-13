"""
Lesson 02 — Function Tools
==========================
Covers:
  - @function_tool decorator — turns a Python function into an agent tool
  - Type hints → JSON schema (the SDK reads them automatically)
  - Docstring → tool description (what the LLM sees)
  - RunContextWrapper[T] — how tools access shared run context
  - Returning structured data from tools

Key concept:
  Tools give agents the ability to DO things, not just say things.
  The SDK introspects your function's type hints and docstring to build
  the JSON schema it sends to the model — no manual schema writing needed.

  RunContextWrapper wraps any context object you pass into Runner.run().
  Tools receive it as their first argument (typed, so you get autocomplete).
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, RunContextWrapper, function_tool

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Basic tools — type hints + docstring = schema
# ---------------------------------------------------------------------------

@function_tool
def get_current_time(timezone: str) -> str:
    """Return the current time for a given timezone label (e.g. 'UTC', 'EST').
    This is a simplified demo — returns local time with the label appended."""
    now = datetime.now().strftime("%H:%M:%S")
    return f"{now} ({timezone})"


@function_tool
def roll_dice(sides: int, count: int = 1) -> str:
    """Roll one or more dice with the given number of sides.

    Args:
        sides: Number of sides on each die (e.g. 6, 20).
        count: How many dice to roll (default 1).
    """
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    return f"Rolled {count}d{sides}: {rolls} (total: {total})"


@function_tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


async def basic_tools_demo():
    print("=" * 50)
    print("PART 1: Basic function tools")
    print("=" * 50)

    agent = Agent(
        name="Assistant",
        instructions=(
            "You are a helpful assistant. Use your tools when the user asks "
            "for something you can compute. Be concise."
        ),
        model="gpt-4o-mini",
        tools=[get_current_time, roll_dice, add_numbers],
    )

    questions = [
        "What time is it in EST right now?",
        "Roll 3 six-sided dice for me.",
        "What is 42.5 plus 17.3?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        result = await Runner.run(agent, input=q)
        print(f"Agent: {result.final_output}")


# ---------------------------------------------------------------------------
# 2. RunContextWrapper — shared context across tools and the run
# ---------------------------------------------------------------------------

@dataclass
class ShoppingCart:
    """Shared context passed through the entire run."""
    items: list[str] = field(default_factory=list)
    discount_pct: float = 0.0


@function_tool
def add_to_cart(ctx: RunContextWrapper[ShoppingCart], item: str) -> str:
    """Add an item to the shopping cart.

    Args:
        item: The name of the item to add.
    """
    ctx.context.items.append(item)
    return f"Added '{item}' to cart. Cart now has {len(ctx.context.items)} item(s)."


@function_tool
def view_cart(ctx: RunContextWrapper[ShoppingCart]) -> str:
    """Show everything currently in the shopping cart."""
    cart = ctx.context
    if not cart.items:
        return "Cart is empty."
    lines = "\n".join(f"  - {item}" for item in cart.items)
    discount_note = f"\n  Discount applied: {cart.discount_pct:.0f}%" if cart.discount_pct else ""
    return f"Cart contents:\n{lines}{discount_note}"


@function_tool
def apply_discount(ctx: RunContextWrapper[ShoppingCart], percent: float) -> str:
    """Apply a percentage discount to the cart.

    Args:
        percent: Discount percentage to apply (e.g. 10 for 10% off).
    """
    ctx.context.discount_pct = percent
    return f"Applied {percent:.0f}% discount to cart."


async def context_tools_demo():
    print("\n" + "=" * 50)
    print("PART 2: RunContextWrapper — shared context")
    print("=" * 50)

    cart = ShoppingCart()

    agent = Agent(
        name="ShopBot",
        instructions=(
            "You are a shopping assistant. Help the user manage their cart. "
            "Use tools to add items, apply discounts, and show the cart. "
            "Be friendly and concise."
        ),
        model="gpt-4o-mini",
        tools=[add_to_cart, view_cart, apply_discount],
    )

    conversation = [
        "Add apples, bananas, and a jar of peanut butter to my cart.",
        "Apply a 15% discount.",
        "What's in my cart now?",
    ]

    result = None
    for msg in conversation:
        print(f"\nUser: {msg}")
        history = result.to_input_list() if result else []
        result = await Runner.run(
            agent,
            input=history + [{"role": "user", "content": msg}],
            context=cart,   # <-- passed here, available in every tool via ctx.context
        )
        print(f"Agent: {result.final_output}")

    # Inspect the context object directly — tools mutated it in place
    print(f"\n[Cart object after run] items={cart.items}, discount={cart.discount_pct}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await basic_tools_demo()
    await context_tools_demo()


if __name__ == "__main__":
    asyncio.run(main())
