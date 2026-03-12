"""
Lesson 08 — ChatCompletionAgent
=================================
ChatCompletionAgent is SK's first-class agent abstraction — a named,
instructed entity that wraps a chat completion service + plugins into
a single cohesive object.

Before SK 1.x agents you had to wire everything manually:
  kernel + service + ChatHistory + service.get_chat_message_content()
ChatCompletionAgent packages all of that into one clean object.

Comparison to what you know:
  LangGraph  → create_react_agent(llm, tools) — returns a compiled graph
  CrewAI     → Agent(role, goal, backstory, tools, llm) — role-based agent
  SK 1.40    → ChatCompletionAgent(kernel, name, instructions) — named agent
               Shares the same Kernel (services + plugins) you've used so far

ChatCompletionAgent API:
  agent.get_response(messages, thread)  → ChatMessageContent  (single reply)
  agent.invoke(messages, thread)        → AsyncGenerator[ChatMessageContent]
                                          (streaming, one item per chunk)

ChatHistoryAgentThread:
  Maintains conversation history across multiple get_response() calls.
  Pass the same thread object on every call — agent accumulates context.
  Equivalent to manually appending to ChatHistory in earlier lessons.

Topics covered:
  1. Basic agent — name + instructions, single response
  2. Agent with plugins — function calling inside the agent
  3. Multi-turn conversation — ChatHistoryAgentThread
  4. Streaming responses — agent.invoke() async generator
  5. Multiple specialist agents — same kernel, different instructions
"""

import asyncio
from typing import Annotated
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import kernel_function, KernelArguments

load_dotenv()

print("=" * 62)
print("Lesson 08 — ChatCompletionAgent")
print("=" * 62)


# ---------------------------------------------------------------------------
# Shared plugins used throughout the lesson
# ---------------------------------------------------------------------------
class MathPlugin:
    @kernel_function(name="add", description="Add two numbers.")
    def add(
        self,
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> Annotated[float, "Sum"]:
        return a + b

    @kernel_function(name="multiply", description="Multiply two numbers.")
    def multiply(
        self,
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> Annotated[float, "Product"]:
        return a * b


class WeatherPlugin:
    """Simulated weather data — no real API call needed."""

    @kernel_function(name="get_weather", description="Get the current weather for a city.")
    def get_weather(
        self,
        city: Annotated[str, "City name"],
    ) -> Annotated[str, "Weather description"]:
        data = {
            "london":    "Cloudy, 12°C, light drizzle",
            "tokyo":     "Sunny, 22°C, gentle breeze",
            "new york":  "Partly cloudy, 18°C, humid",
            "sydney":    "Clear, 26°C, sunny",
        }
        return data.get(city.lower(), f"No weather data for '{city}'")


def make_kernel() -> sk.Kernel:
    """Create a fresh kernel with the Anthropic service registered."""
    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    return kernel


# ---------------------------------------------------------------------------
# Default execution settings — FunctionChoiceBehavior.Auto() lets the agent
# call its plugins autonomously.  Pass via KernelArguments(settings=...) at
# each get_response() / invoke() call site — NOT in the agent constructor.
# ---------------------------------------------------------------------------
def auto_settings(max_tokens: int = 300) -> PromptExecutionSettings:
    return PromptExecutionSettings(
        max_tokens=max_tokens,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )


# ---------------------------------------------------------------------------
# 1. Basic Agent — get_response()
# ---------------------------------------------------------------------------
# The simplest use: create an agent with a name and instructions,
# call get_response() with a plain string message.
# No plugins, no history — just a named, instructed LLM call.

async def basic_agent_example():
    print("\n--- 1. Basic Agent (get_response) ---")

    kernel = make_kernel()

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Haiku_Bot",
        instructions=(
            "You are a creative assistant who always responds in haiku format "
            "(5-7-5 syllables). Nothing else — just the haiku."
        ),
    )

    prompts = [
        "Explain what a kernel is in Semantic Kernel.",
        "Describe the feeling of debugging at 2am.",
    ]

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        response = await agent.get_response(messages=prompt)
        print(f"{agent.name}: {response}")


# ---------------------------------------------------------------------------
# 2. Agent with Plugins — function calling inside the agent
# ---------------------------------------------------------------------------
# Add plugins to the kernel before creating the agent.
# The agent automatically uses FunctionChoiceBehavior.Auto() when
# execution_settings include it, picking the right tools for each query.

async def agent_with_plugins_example():
    print("\n--- 2. Agent with Plugins (function calling) ---")

    kernel = make_kernel()
    kernel.add_plugin(MathPlugin(), plugin_name="math")
    kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Assistant",
        instructions="You are a helpful assistant. Use your tools when appropriate. Be concise.",
    )

    queries = [
        "What is 47 multiplied by 23?",
        "What's the weather like in Tokyo right now?",
        "What is 15 + 27, and what's the weather in London?",
    ]

    args = KernelArguments(settings=auto_settings(max_tokens=400))
    for query in queries:
        print(f"\nUser: {query}")
        response = await agent.get_response(messages=query, arguments=args)
        print(f"{agent.name}: {response}")


# ---------------------------------------------------------------------------
# 3. Multi-Turn Conversation — ChatHistoryAgentThread
# ---------------------------------------------------------------------------
# ChatHistoryAgentThread accumulates messages across calls.
# The agent remembers what was said earlier in the thread without
# you manually appending to a ChatHistory object.
#
# LangGraph equivalent:  checkpointer + thread_id
# CrewAI equivalent:     task context=[] wiring (much less flexible)

async def multi_turn_example():
    print("\n--- 3. Multi-Turn Conversation (ChatHistoryAgentThread) ---")

    kernel = make_kernel()
    kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Travel_Advisor",
        instructions=(
            "You are a friendly travel advisor. Remember what the user tells you "
            "and use that context in follow-up answers. Keep responses short."
        ),
    )

    # One thread = one conversation session
    thread = ChatHistoryAgentThread()
    args = KernelArguments(settings=auto_settings(max_tokens=300))

    turns = [
        "I'm planning a trip and considering Tokyo or Sydney.",
        "What's the weather like in both cities right now?",
        "Based on the weather, which city would you recommend?",
        "What's one thing I should definitely do there?",
    ]

    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        response = await agent.get_response(messages=user_msg, thread=thread, arguments=args)
        print(f"{agent.name}: {response}")

    # Inspect how many messages accumulated in the thread
    print(f"\n[Thread has {len(thread._chat_history)} messages in history]")


# ---------------------------------------------------------------------------
# 4. Streaming — agent.invoke() async generator
# ---------------------------------------------------------------------------
# agent.invoke() streams the response token by token.
# Each yielded item is a ChatMessageContent with partial text.
# Use this when you want to display output progressively (like ChatGPT).
#
# Note: with Anthropic, streaming arrives as full chunks rather than
# single tokens — the pattern still works identically.

async def streaming_example():
    print("\n--- 4. Streaming Responses (agent.invoke) ---")

    kernel = make_kernel()
    kernel.add_plugin(MathPlugin(), plugin_name="math")

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Streaming_Demo",
        instructions="You are a concise assistant. Show your reasoning step by step.",
    )

    prompt = "Calculate 123 + 456, then multiply the result by 2. Show each step."
    print(f"User: {prompt}")
    print(f"{agent.name}: ", end="", flush=True)

    args = KernelArguments(settings=auto_settings(max_tokens=300))
    async for chunk in agent.invoke(messages=prompt, arguments=args):
        # Each chunk is a ChatMessageContent — print content without newline
        print(chunk.content, end="", flush=True)

    print()  # final newline


# ---------------------------------------------------------------------------
# 5. Multiple Specialist Agents — same kernel, different instructions
# ---------------------------------------------------------------------------
# A single kernel (with all plugins) can back multiple agents.
# Each agent has its own name, persona, and instructions.
# This is the foundation for multi-agent systems (Lesson 09).
#
# LangGraph equivalent:  separate nodes with different system prompts
# CrewAI equivalent:     Agent(role=..., goal=..., backstory=...)

async def specialist_agents_example():
    print("\n--- 5. Specialist Agents (same kernel, different personas) ---")

    kernel = make_kernel()
    kernel.add_plugin(MathPlugin(), plugin_name="math")
    kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

    args = KernelArguments(settings=auto_settings(max_tokens=200))

    # Agent 1: Math specialist — only cares about numbers
    math_agent = ChatCompletionAgent(
        kernel=kernel,
        name="Math_Expert",
        instructions=(
            "You are a math specialist. Only answer math questions using your tools. "
            "For non-math questions, say 'That's outside my expertise.'"
        ),
    )

    # Agent 2: Travel concierge — focuses on weather and travel advice
    travel_agent = ChatCompletionAgent(
        kernel=kernel,
        name="Travel_Concierge",
        instructions=(
            "You are an enthusiastic travel concierge. Use the weather tool to give "
            "personalized travel tips. Always end with a travel emoji."
        ),
    )

    # Same question, different agents — same kernel, completely different behaviour
    query_math   = "What is 144 divided by 12?"
    query_travel = "Should I visit New York or London this week?"

    print(f"Sending '{query_math}' to both agents:\n")
    r1 = await math_agent.get_response(messages=query_math, arguments=args)
    print(f"  {math_agent.name}: {r1}")
    r2 = await travel_agent.get_response(messages=query_math, arguments=args)
    print(f"  {travel_agent.name}: {r2}")

    print(f"\nSending '{query_travel}' to both agents:\n")
    r3 = await math_agent.get_response(messages=query_travel, arguments=args)
    print(f"  {math_agent.name}: {r3}")
    r4 = await travel_agent.get_response(messages=query_travel, arguments=args)
    print(f"  {travel_agent.name}: {r4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await basic_agent_example()
    await agent_with_plugins_example()
    await multi_turn_example()
    await streaming_example()
    await specialist_agents_example()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • ChatCompletionAgent = kernel + name + instructions (system prompt)")
    print("  • get_response(messages, thread) → single ChatMessageContent reply")
    print("  • invoke(messages, thread)       → async generator (streaming)")
    print("  • ChatHistoryAgentThread tracks multi-turn context automatically")
    print("  • Plugins added to the kernel are available to all agents on it")
    print("  • Multiple agents can share one kernel — different personas, same tools")
    print("  • Pass KernelArguments(settings=PromptExecutionSettings(...)) at call sites")
    print("  • Next: Lesson 09 — Multi-Agent Collaboration (AgentGroupChat)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
