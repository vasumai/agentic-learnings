"""
Lesson 08 — Streaming
======================
Covers:
  - Runner.run_streamed() — streaming variant of Runner.run()
  - stream_events() — async iterator over StreamEvent objects
  - Key event types: RawResponsesStreamEvent, RunItemStreamEvent, AgentUpdatedStreamEvent
  - Printing tokens as they arrive (real-time output)
  - Streaming with tools — seeing tool calls and results in the stream
  - Streaming with handoffs — observing agent transitions live

Key concept:
  Runner.run_streamed() returns a StreamedRunResult immediately.
  Nothing executes until you iterate stream_events().
  Each event describes one thing that happened: a token chunk,
  a tool call, a tool result, a handoff, or the run completing.

  Use streaming when:
    - You want to show output token-by-token (chat UX)
    - You need to observe/log what's happening mid-run
    - You're building a UI that reacts to agent actions in real time
"""

import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, RunContextWrapper, function_tool, handoff
from agents.stream_events import (
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    AgentUpdatedStreamEvent,
)
from agents.run_context import TResponseInputItem
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Basic streaming — print tokens as they arrive
# ---------------------------------------------------------------------------

async def basic_streaming_demo():
    print("=" * 50)
    print("PART 1: Basic streaming — tokens as they arrive")
    print("=" * 50)

    agent = Agent(
        name="Storyteller",
        instructions="You are a creative storyteller. Tell short, vivid stories in 4–5 sentences.",
        model="gpt-4o-mini",
    )

    prompt = "Tell me a short story about a robot who discovers music."
    print(f"\nUser: {prompt}")
    print("Agent: ", end="", flush=True)

    result = Runner.run_streamed(agent, input=prompt)

    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            # RawResponsesStreamEvent wraps OpenAI's raw streaming events
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                print(data.delta, end="", flush=True)

    print()  # newline after streaming completes
    print(f"\n[Run complete] final_output length: {len(result.final_output)} chars")


# ---------------------------------------------------------------------------
# 2. RunItemStreamEvent — observe tool calls and results
# ---------------------------------------------------------------------------

@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.
    """
    # Mock data
    data = {
        "London":   "13°C, overcast",
        "Tokyo":    "22°C, sunny",
        "New York": "8°C, windy",
        "Sydney":   "27°C, partly cloudy",
    }
    return data.get(city, f"Weather data unavailable for {city}.")


@function_tool
def get_local_time(city: str) -> str:
    """Get the current local time for a city.

    Args:
        city: The city name.
    """
    times = {
        "London":   "14:30 GMT",
        "Tokyo":    "23:30 JST",
        "New York": "09:30 EST",
        "Sydney":   "01:30 AEDT",
    }
    return times.get(city, f"Time data unavailable for {city}.")


async def tool_streaming_demo():
    print("\n" + "=" * 50)
    print("PART 2: Streaming with tools — observe tool calls")
    print("=" * 50)

    agent = Agent(
        name="TravelAgent",
        instructions="You help with travel info. Use tools to get weather and local time.",
        model="gpt-4o-mini",
        tools=[get_weather, get_local_time],
    )

    prompt = "What's the weather and local time in Tokyo and London right now?"
    print(f"\nUser: {prompt}\n")

    result = Runner.run_streamed(agent, input=prompt)

    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                print(data.delta, end="", flush=True)

        elif isinstance(event, RunItemStreamEvent):
            item = event.item
            item_type = type(item).__name__

            if item_type == "ToolCallItem":
                print(f"\n  [tool call] {item.raw_item.name}({item.raw_item.arguments})")
            elif item_type == "ToolCallOutputItem":
                print(f"  [tool result] {item.output}")

    print()


# ---------------------------------------------------------------------------
# 3. AgentUpdatedStreamEvent — observe handoffs live
# ---------------------------------------------------------------------------

support_agent = Agent(
    name="SupportAgent",
    instructions=(
        "You are a general support agent. "
        "For billing questions, hand off to BillingAgent. "
        "For technical questions, hand off to TechAgent. "
        "Otherwise answer directly."
    ),
    model="gpt-4o-mini",
)

billing_specialist = Agent(
    name="BillingAgent",
    instructions="You are a billing specialist. Handle invoice and payment questions concisely.",
    model="gpt-4o-mini",
)

tech_specialist = Agent(
    name="TechAgent",
    instructions="You are a tech support specialist. Handle API and error questions concisely.",
    model="gpt-4o-mini",
)

support_agent.handoffs = [billing_specialist, tech_specialist]


async def handoff_streaming_demo():
    print("\n" + "=" * 50)
    print("PART 3: Streaming with handoffs — observe agent transitions")
    print("=" * 50)

    queries = [
        "I have a question about my invoice from last month.",
        "I'm getting a 500 error from your API.",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        print("Stream: ", end="", flush=True)

        current_agent = support_agent.name
        result = Runner.run_streamed(support_agent, input=query)

        async for event in result.stream_events():
            if isinstance(event, AgentUpdatedStreamEvent):
                new_agent = event.new_agent.name
                if new_agent != current_agent:
                    print(f"\n  [handoff] {current_agent} → {new_agent}")
                    current_agent = new_agent
                    print("  ", end="", flush=True)

            elif isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)

        print(f"\n  [ended with] {result.current_agent.name}")


# ---------------------------------------------------------------------------
# 4. Stream events summary — print all event types for one run
# ---------------------------------------------------------------------------

async def event_types_demo():
    print("\n" + "=" * 50)
    print("PART 4: Event type inspector — see everything in a run")
    print("=" * 50)

    agent = Agent(
        name="Assistant",
        instructions="Answer briefly.",
        model="gpt-4o-mini",
        tools=[get_weather],
    )

    prompt = "What's the weather in Sydney?"
    print(f"\nUser: {prompt}\n")

    result = Runner.run_streamed(agent, input=prompt)
    seen_types: dict[str, int] = {}

    async for event in result.stream_events():
        etype = type(event).__name__
        seen_types[etype] = seen_types.get(etype, 0) + 1

    print("Event type counts for this run:")
    for etype, count in sorted(seen_types.items()):
        print(f"  {etype}: {count}")

    print(f"\nFinal output: {result.final_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await basic_streaming_demo()
    await tool_streaming_demo()
    await handoff_streaming_demo()
    await event_types_demo()


if __name__ == "__main__":
    asyncio.run(main())
