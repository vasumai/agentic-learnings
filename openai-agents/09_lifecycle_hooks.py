"""
Lesson 09 — Lifecycle Hooks
============================
Covers:
  - RunHooks  — run-level callbacks (fire for everything in the run)
  - AgentHooks — agent-level callbacks (fire only for that specific agent)
  - on_agent_start / on_agent_end
  - on_tool_start / on_tool_end
  - on_handoff
  - Practical uses: logging, timing, cost tracking, audit trails

Key concept:
  Hooks give you observability without touching agent or tool code.
  RunHooks attach to Runner.run() and see every event in the run.
  AgentHooks attach to a specific Agent and only fire for that agent.

  Both are subclasses — override only the methods you need.
  Hooks receive RunContextWrapper so they can read/write shared context,
  making them ideal for building audit trails and metrics accumulators.
"""

import asyncio
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    RunHooks,
    AgentHooks,
    function_tool,
    handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()


# ---------------------------------------------------------------------------
# 1. RunHooks — observe everything in a run
# ---------------------------------------------------------------------------

class LoggingRunHooks(RunHooks):
    """Prints every lifecycle event for the entire run."""

    async def on_agent_start(self, context, agent: Agent) -> None:
        print(f"  [run hook] agent_start  → {agent.name}")

    async def on_agent_end(self, context, agent: Agent, output) -> None:
        print(f"  [run hook] agent_end    → {agent.name} | output: {str(output)[:60]}")

    async def on_tool_start(self, context, agent: Agent, tool) -> None:
        print(f"  [run hook] tool_start   → {agent.name} calling {tool.name}")

    async def on_tool_end(self, context, agent: Agent, tool, result: str) -> None:
        print(f"  [run hook] tool_end     → {tool.name} returned: {result[:60]}")

    async def on_handoff(self, context, from_agent: Agent, to_agent: Agent) -> None:
        print(f"  [run hook] handoff      → {from_agent.name} ⟶ {to_agent.name}")


@function_tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
    """
    prices = {"AAPL": "$189.45", "MSFT": "$415.20", "GOOGL": "$172.30", "NVDA": "$875.00"}
    return prices.get(ticker.upper(), f"Ticker {ticker} not found.")


async def run_hooks_demo():
    print("=" * 50)
    print("PART 1: RunHooks — log every lifecycle event")
    print("=" * 50)

    agent = Agent(
        name="StockAdvisor",
        instructions="You are a stock price assistant. Use tools to look up prices. Be concise.",
        model="gpt-4o-mini",
        tools=[get_stock_price],
    )

    print("\nUser: What are the prices of AAPL and NVDA?")
    result = await Runner.run(
        agent,
        input="What are the prices of AAPL and NVDA?",
        hooks=LoggingRunHooks(),
    )
    print(f"\nFinal: {result.final_output}")


# ---------------------------------------------------------------------------
# 2. AgentHooks — per-agent observability
# ---------------------------------------------------------------------------

class TimingAgentHooks(AgentHooks):
    """Measures time spent in tools for a specific agent."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._tool_start_times: dict[str, float] = {}

    async def on_tool_start(self, context, agent: Agent, tool) -> None:
        self._tool_start_times[tool.name] = time.perf_counter()
        print(f"  [{self.agent_name} hook] ⏱ starting {tool.name}")

    async def on_tool_end(self, context, agent: Agent, tool, result: str) -> None:
        elapsed = time.perf_counter() - self._tool_start_times.get(tool.name, 0)
        print(f"  [{self.agent_name} hook] ✓ {tool.name} took {elapsed*1000:.1f}ms")


@function_tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g. USD).
        to_currency: Target currency code (e.g. EUR).
    """
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "USD_GBP": 0.79, "GBP_USD": 1.27}
    key = f"{from_currency.upper()}_{to_currency.upper()}"
    rate = rates.get(key)
    if not rate:
        return f"No rate found for {from_currency} → {to_currency}."
    converted = amount * rate
    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"


async def agent_hooks_demo():
    print("\n" + "=" * 50)
    print("PART 2: AgentHooks — per-agent timing")
    print("=" * 50)

    agent = Agent(
        name="CurrencyAgent",
        instructions="You are a currency conversion assistant. Use tools to convert amounts.",
        model="gpt-4o-mini",
        tools=[convert_currency],
        hooks=TimingAgentHooks("CurrencyAgent"),   # attached directly to the agent
    )

    print("\nUser: Convert 500 USD to EUR and 1000 GBP to USD.")
    result = await Runner.run(agent, input="Convert 500 USD to EUR and 1000 GBP to USD.")
    print(f"\nFinal: {result.final_output}")


# ---------------------------------------------------------------------------
# 3. Context-accumulating hooks — build an audit trail + metrics
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    tool_calls: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    agents_involved: list[str] = field(default_factory=list)
    total_tool_calls: int = 0


class MetricsRunHooks(RunHooks):
    """Accumulates run metrics into shared context."""

    async def on_agent_start(self, context: RunContextWrapper[RunMetrics], agent: Agent) -> None:
        if agent.name not in context.context.agents_involved:
            context.context.agents_involved.append(agent.name)

    async def on_tool_start(self, context: RunContextWrapper[RunMetrics], agent: Agent, tool) -> None:
        context.context.tool_calls.append(f"{agent.name}.{tool.name}")
        context.context.total_tool_calls += 1

    async def on_handoff(self, context: RunContextWrapper[RunMetrics], from_agent: Agent, to_agent: Agent) -> None:
        context.context.handoffs.append(f"{from_agent.name} → {to_agent.name}")


@function_tool
def lookup_order(ctx: RunContextWrapper[RunMetrics], order_id: str) -> str:
    """Look up an order by ID.

    Args:
        order_id: The order identifier.
    """
    orders = {
        "ORD-1": {"status": "shipped", "total": "$149.99"},
        "ORD-2": {"status": "processing", "total": "$89.50"},
    }
    order = orders.get(order_id, None)
    if not order:
        return f"Order {order_id} not found."
    return f"Order {order_id}: {order['status']}, total {order['total']}"


@function_tool
def request_refund(ctx: RunContextWrapper[RunMetrics], order_id: str) -> str:
    """Request a refund for an order.

    Args:
        order_id: The order to refund.
    """
    return f"Refund requested for {order_id}. Processing in 3–5 business days."


fulfillment_agent = Agent(
    name="FulfillmentAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You handle order lookups. For refund requests, hand off to RefundAgent."
    ),
    model="gpt-4o-mini",
    tools=[lookup_order],
)

refund_agent = Agent(
    name="RefundAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You handle refund requests. Use the request_refund tool."
    ),
    model="gpt-4o-mini",
    tools=[request_refund],
)

fulfillment_agent.handoffs = [refund_agent]


async def metrics_hooks_demo():
    print("\n" + "=" * 50)
    print("PART 3: Context-accumulating hooks — metrics + audit trail")
    print("=" * 50)

    metrics = RunMetrics()

    print("\nUser: Look up order ORD-1, then refund it.")
    result = await Runner.run(
        fulfillment_agent,
        input="Look up order ORD-1 and then process a refund for it.",
        context=metrics,
        hooks=MetricsRunHooks(),
    )

    print(f"\nFinal: {result.final_output}")
    print(f"\n[Run metrics]")
    print(f"  agents_involved  : {metrics.agents_involved}")
    print(f"  handoffs         : {metrics.handoffs}")
    print(f"  tool_calls       : {metrics.tool_calls}")
    print(f"  total_tool_calls : {metrics.total_tool_calls}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await run_hooks_demo()
    await agent_hooks_demo()
    await metrics_hooks_demo()


if __name__ == "__main__":
    asyncio.run(main())
