"""
Lesson 02 — Plugins (Native Functions)
========================================
In Semantic Kernel, a Plugin is a class that groups related functions the
Kernel can discover and call. Each callable function is decorated with
@kernel_function.

Comparison to what you know:
  CrewAI  → @tool / BaseTool — functions an Agent can use
  SK      → @kernel_function inside a Plugin class — functions the Kernel calls
  LangGraph → tool-decorated functions bound to a node's LLM

Three ways to invoke plugin functions:
  1. kernel.invoke()            — direct, explicit call by plugin+function name
  2. kernel.invoke_prompt()     — LLM picks which function to call (auto-invoke)
  3. service.get_chat_message_content() with tool_choice — low-level tool loop

This lesson covers (1) and a taste of (2).
"""

import asyncio
import random
from typing import Annotated
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import kernel_function, KernelArguments

load_dotenv()

# ---------------------------------------------------------------------------
# Setup Kernel (same as Lesson 01)
# ---------------------------------------------------------------------------
kernel = sk.Kernel()
kernel.add_service(
    AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
)

print("=" * 55)
print("Lesson 02 — Plugins (Native Functions)")
print("=" * 55)


# ---------------------------------------------------------------------------
# 1. Define a Plugin
# ---------------------------------------------------------------------------
# A plugin is just a plain Python class.
# Each method decorated with @kernel_function becomes discoverable by the Kernel.
# Type annotations on parameters become the function's schema (used by the LLM).

class MathPlugin:
    """A plugin with basic math utilities."""

    @kernel_function(
        name="add",
        description="Add two numbers together.",
    )
    def add(
        self,
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> Annotated[float, "Sum of a and b"]:
        return a + b

    @kernel_function(
        name="multiply",
        description="Multiply two numbers.",
    )
    def multiply(
        self,
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> Annotated[float, "Product of a and b"]:
        return a * b

    @kernel_function(
        name="percentage",
        description="Calculate what percent 'part' is of 'total'.",
    )
    def percentage(
        self,
        part: Annotated[float, "The partial value"],
        total: Annotated[float, "The total value"],
    ) -> Annotated[str, "Percentage string"]:
        pct = (part / total) * 100
        return f"{pct:.1f}%"


class WeatherPlugin:
    """Simulates a weather lookup tool (no real API — educational only)."""

    FAKE_DATA = {
        "london":    {"temp": 12, "condition": "cloudy"},
        "new york":  {"temp": 18, "condition": "sunny"},
        "tokyo":     {"temp": 22, "condition": "partly cloudy"},
        "sydney":    {"temp": 25, "condition": "sunny"},
    }

    @kernel_function(
        name="get_weather",
        description="Get current weather for a city. Returns temperature in Celsius.",
    )
    def get_weather(
        self,
        city: Annotated[str, "Name of the city"],
    ) -> Annotated[str, "Weather description"]:
        data = self.FAKE_DATA.get(city.lower())
        if data:
            return f"{city}: {data['temp']}°C, {data['condition']}"
        return f"No weather data for {city}"

    @kernel_function(
        name="get_forecast",
        description="Get a 3-day weather forecast for a city.",
    )
    def get_forecast(
        self,
        city: Annotated[str, "Name of the city"],
    ) -> Annotated[str, "3-day forecast"]:
        conditions = ["sunny", "cloudy", "rainy", "windy", "partly cloudy"]
        forecast = [
            f"Day {i+1}: {random.randint(10, 28)}°C, {random.choice(conditions)}"
            for i in range(3)
        ]
        return f"Forecast for {city}:\n" + "\n".join(forecast)


# ---------------------------------------------------------------------------
# 2. Register Plugins with the Kernel
# ---------------------------------------------------------------------------
# add_plugin(instance, plugin_name) — the plugin_name is used when invoking.
math_plugin    = kernel.add_plugin(MathPlugin(),   plugin_name="math")
weather_plugin = kernel.add_plugin(WeatherPlugin(), plugin_name="weather")


# ---------------------------------------------------------------------------
# 3. Direct Invocation — kernel.invoke()
# ---------------------------------------------------------------------------
# Call a specific plugin function by name with typed arguments.

async def direct_invocation_example():
    print("\n--- 3. Direct Invocation (kernel.invoke) ---")

    # KernelArguments is a typed dict for passing args to plugin functions
    result = await kernel.invoke(
        plugin_name="math",
        function_name="add",
        arguments=KernelArguments(a=42, b=58),
    )
    print(f"math.add(42, 58) = {result}")

    result = await kernel.invoke(
        plugin_name="math",
        function_name="percentage",
        arguments=KernelArguments(part=73, total=200),
    )
    print(f"math.percentage(73, 200) = {result}")

    result = await kernel.invoke(
        plugin_name="weather",
        function_name="get_weather",
        arguments=KernelArguments(city="Tokyo"),
    )
    print(f"weather.get_weather('Tokyo') = {result}")

    result = await kernel.invoke(
        plugin_name="weather",
        function_name="get_forecast",
        arguments=KernelArguments(city="London"),
    )
    print(f"weather.get_forecast('London'):\n{result}")


# ---------------------------------------------------------------------------
# 4. Inspect registered plugins
# ---------------------------------------------------------------------------
# The Kernel exposes all registered functions — useful for debugging.

async def inspect_plugins_example():
    print("\n--- 4. Inspecting Registered Plugins ---")
    for plugin_name, plugin in kernel.plugins.items():
        print(f"Plugin: {plugin_name}")
        for fn_name, fn in plugin.functions.items():
            desc = fn.description or "(no description)"
            print(f"  • {fn_name}: {desc}")


# ---------------------------------------------------------------------------
# 5. Auto-Invocation — LLM decides which function to call
# ---------------------------------------------------------------------------
# FunctionChoiceBehavior.Auto() tells the LLM it can call any registered
# plugin function to answer the prompt. SK handles the tool-call loop.
# This is the SK equivalent of CrewAI's agent.tools=[...] auto-selection.

async def auto_invocation_example():
    print("\n--- 5. Auto-Invocation (LLM picks the function) ---")

    settings = PromptExecutionSettings(
        max_tokens=300,
        # Tell SK: let the LLM auto-select and call any registered function
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    prompts = [
        "What is 144 multiplied by 7?",
        "What's the weather like in New York right now?",
    ]

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = await kernel.invoke_prompt(prompt=prompt, settings=settings)
        print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await direct_invocation_example()
    await inspect_plugins_example()
    await auto_invocation_example()

    print("\n" + "=" * 55)
    print("Key Takeaways:")
    print("  • @kernel_function marks a method as SK-callable")
    print("  • Annotated[type, 'desc'] provides the LLM schema")
    print("  • kernel.add_plugin() registers a plugin by name")
    print("  • kernel.invoke() = explicit call by plugin+function")
    print("  • FunctionChoiceBehavior.Auto() = LLM picks the function")
    print("  • Next: Lesson 03 — Prompt Templates with variables")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
