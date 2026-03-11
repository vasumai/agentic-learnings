"""
Lesson 06 — Function Calling & Auto-Orchestration
===================================================
In SK 0.x there were explicit Planner classes (SequentialPlanner,
StepwisePlanner, BasicPlanner). In SK 1.x these are gone — replaced by
the standard LLM tool-calling mechanism controlled via FunctionChoiceBehavior.

The concept is the same as the old planners:
  • You register plugins (tools)
  • You send a goal/query to the LLM
  • The LLM decides which tools to call and in what order
  • SK handles the tool-call loop automatically

Comparison to what you know:
  LangGraph  → conditional edges + ToolNode — you define the routing explicitly
  CrewAI     → hierarchical Crew / @router — manager agent decides delegation
  SK 1.40    → FunctionChoiceBehavior.Auto() — LLM picks tools inside a loop
               No explicit graph/routing needed; the LLM IS the planner

FunctionChoiceBehavior has three modes:
  Auto()       — LLM decides which tool(s) to call; SK auto-invokes; loops until done
  Required()   — LLM MUST call one of the provided functions (guaranteed tool use)
  NoneInvoke() — Tools are described to the LLM but NOT called; dry-run/inspect mode

Key control levers:
  maximum_auto_invoke_attempts  — max tool-call rounds (default 5)
  filters = {"included_functions": [...]}  — restrict which tools are available
  filters = {"excluded_plugins": [...]}    — hide entire plugins from the LLM
"""

import asyncio
import json
import random
from datetime import date
from typing import Annotated
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function, KernelArguments

load_dotenv()

# ---------------------------------------------------------------------------
# Setup Kernel
# ---------------------------------------------------------------------------
kernel = sk.Kernel()
service = AnthropicChatCompletion(
    ai_model_id="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    service_id="anthropic_chat",
)
kernel.add_service(service)

print("=" * 62)
print("Lesson 06 — Function Calling & Auto-Orchestration")
print("=" * 62)


# ---------------------------------------------------------------------------
# Define Plugins
# ---------------------------------------------------------------------------
# We need a few plugins with clear, distinct purposes so the LLM can reason
# about WHICH tool to call for each part of a multi-step goal.

class ResearchPlugin:
    """Provides research and data lookup tools."""

    @kernel_function(name="get_framework_info", description="Get information about an AI framework by name.")
    def get_framework_info(
        self,
        framework: Annotated[str, "Name of the AI framework (e.g. LangGraph, CrewAI, Semantic Kernel)"],
    ) -> Annotated[str, "Framework description"]:
        data = {
            "langgraph": "LangGraph is a graph-based stateful agent framework built on LangChain. Best for complex routing and state machines.",
            "crewai":    "CrewAI is a role-based multi-agent framework. Agents have roles, goals, and backstories. Best for team-oriented pipelines.",
            "semantic kernel": "Semantic Kernel is Microsoft's enterprise AI SDK. Plugin-based architecture with memory, planners, and MCP support.",
        }
        return data.get(framework.lower(), f"No data found for '{framework}'")

    @kernel_function(name="compare_frameworks", description="Compare two AI frameworks side by side.")
    def compare_frameworks(
        self,
        framework_a: Annotated[str, "First framework name"],
        framework_b: Annotated[str, "Second framework name"],
    ) -> Annotated[str, "Comparison result"]:
        return (
            f"Comparison: {framework_a} vs {framework_b}\n"
            f"  {framework_a}: graph/state-machine approach, explicit routing\n"
            f"  {framework_b}: {('role-based agents' if 'crew' in framework_b.lower() else 'plugin/kernel architecture')}\n"
            f"  Use {framework_a} for: complex state machines with branching\n"
            f"  Use {framework_b} for: {'team delegation' if 'crew' in framework_b.lower() else 'enterprise plugin architecture'}"
        )


class MathPlugin:
    """Math utilities."""

    @kernel_function(name="calculate", description="Evaluate a simple math expression like '12 * 7 + 3'.")
    def calculate(
        self,
        expression: Annotated[str, "A Python-evaluable math expression"],
    ) -> Annotated[str, "The result"]:
        try:
            # safe eval for basic arithmetic only
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: only basic arithmetic allowed"
            return str(eval(expression))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    @kernel_function(name="percentage_change", description="Calculate the percentage change from old to new value.")
    def percentage_change(
        self,
        old_value: Annotated[float, "Original value"],
        new_value: Annotated[float, "New value"],
    ) -> Annotated[str, "Percentage change"]:
        change = ((new_value - old_value) / old_value) * 100
        direction = "increase" if change >= 0 else "decrease"
        return f"{abs(change):.1f}% {direction}"


class DatePlugin:
    """Date and time utilities."""

    @kernel_function(name="today", description="Get today's date.")
    def today(self) -> Annotated[str, "Today's date"]:
        return date.today().strftime("%B %d, %Y")

    @kernel_function(name="days_until", description="Calculate how many days until a given future date (YYYY-MM-DD).")
    def days_until(
        self,
        target_date: Annotated[str, "Target date in YYYY-MM-DD format"],
    ) -> Annotated[str, "Days remaining"]:
        from datetime import datetime
        try:
            target = datetime.strptime(target_date, "%Y-%m-%d").date()
            delta = (target - date.today()).days
            if delta < 0:
                return f"{abs(delta)} days ago"
            return f"{delta} days from now"
        except ValueError:
            return "Invalid date format. Use YYYY-MM-DD."


# Register all plugins
kernel.add_plugin(ResearchPlugin(), plugin_name="research")
kernel.add_plugin(MathPlugin(),     plugin_name="math")
kernel.add_plugin(DatePlugin(),     plugin_name="date")


# ---------------------------------------------------------------------------
# Helper — run a prompt with given FunctionChoiceBehavior
# ---------------------------------------------------------------------------
async def run_with_tools(prompt: str, behavior: FunctionChoiceBehavior, max_tokens: int = 400) -> str:
    history = ChatHistory()
    history.add_user_message(prompt)
    settings = PromptExecutionSettings(
        max_tokens=max_tokens,
        function_choice_behavior=behavior,
    )
    response = await service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,          # ← kernel required for auto-invoke
    )
    return str(response)


# ---------------------------------------------------------------------------
# 1. FunctionChoiceBehavior.Auto() — LLM picks the right tool
# ---------------------------------------------------------------------------
# The LLM reads all registered function descriptions and decides which to call.
# SK executes the tool, feeds the result back, and the LLM continues.
# This is the SK equivalent of LangGraph's ToolNode + conditional_edges.

async def auto_mode_example():
    print("\n--- 1. Auto Mode (LLM picks the tool) ---")

    prompts = [
        "What is today's date?",
        "What is 15% of 340?",
        "Tell me about the CrewAI framework.",
    ]

    behavior = FunctionChoiceBehavior.Auto()

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = await run_with_tools(prompt, behavior)
        print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# 2. Multi-Step Tool Chaining
# ---------------------------------------------------------------------------
# The LLM can call multiple tools in sequence within a single response.
# It calls tool A, gets the result, decides to call tool B, and so on.
# This is the "planner" behaviour — the LLM creates and executes the plan.

async def multi_step_example():
    print("\n--- 2. Multi-Step Tool Chaining ---")

    prompt = (
        "I need to compare LangGraph and CrewAI frameworks. "
        "Also, what is today's date so I know when I ran this comparison?"
    )
    print(f"User: {prompt}")

    result = await run_with_tools(
        prompt,
        FunctionChoiceBehavior.Auto(),
        max_tokens=500,
    )
    print(f"Claude:\n{result}")


# ---------------------------------------------------------------------------
# 3. FunctionChoiceBehavior.Required() — force tool use
# ---------------------------------------------------------------------------
# The LLM MUST call at least one of the provided functions.
# Useful when you want to guarantee structured output via a tool call,
# rather than letting the LLM answer in free text.

async def required_mode_example():
    print("\n--- 3. Required Mode (LLM must use a tool) ---")

    # Force the LLM to use the math.calculate tool
    behavior = FunctionChoiceBehavior.Required(
        filters={"included_functions": ["math-calculate"]}
    )

    prompt = "What is 123 multiplied by 456?"
    print(f"User: {prompt}")
    result = await run_with_tools(prompt, behavior)
    print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# 4. Disabling auto-invoke — inspect tool calls before executing
# ---------------------------------------------------------------------------
# FunctionChoiceBehavior.NoneInvoke() is NOT supported by Anthropic
# (it maps to tool_choice='none' which Anthropic's API rejects).
#
# The Anthropic-compatible equivalent: set maximum_auto_invoke_attempts=0.
# SK will pass tool descriptions to the LLM, but will NOT execute any calls.
# The raw ChatMessageContent will contain tool_use blocks you can inspect.
# This is the SK pattern for "human approval before execution" (HITL Lesson 10).

async def none_invoke_example():
    print("\n--- 4. Manual tool-call inspection (auto_invoke disabled) ---")

    history = ChatHistory()
    history.add_user_message("What is 99 * 99?")

    behavior = FunctionChoiceBehavior.Auto()
    behavior.maximum_auto_invoke_attempts = 0   # tools visible to LLM but NOT called

    settings = PromptExecutionSettings(
        max_tokens=200,
        function_choice_behavior=behavior,
    )

    response = await service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )
    # The response contains tool_use blocks — SK didn't execute them
    print(f"Raw response type: {type(response).__name__}")
    print(f"Content items: {len(response.items)}")
    for item in response.items:
        print(f"  Item type: {type(item).__name__} — {item}")
    print("(Tool was described by LLM but NOT executed — inspect before calling)")


# ---------------------------------------------------------------------------
# 5. Filtering — restrict which tools the LLM can see
# ---------------------------------------------------------------------------
# Use filters to expose only a subset of registered plugins/functions.
# This is important for security and for guiding the LLM to relevant tools.

async def filters_example():
    print("\n--- 5. Filters — controlling tool visibility ---")

    # Only allow the date plugin — hide math and research
    date_only = FunctionChoiceBehavior.Auto(
        filters={"included_plugins": ["date"]}
    )
    print("With only 'date' plugin exposed:")
    result = await run_with_tools("What is today's date and what is 5 * 5?", date_only)
    print(f"Claude: {result}")
    print("(Note: Claude answered the date but couldn't compute 5*5 — no math tool available)")

    # Exclude just the research plugin
    no_research = FunctionChoiceBehavior.Auto(
        filters={"excluded_plugins": ["research"]}
    )
    print("\nWith 'research' plugin excluded:")
    result = await run_with_tools("What is 7 + 8, and tell me about LangGraph.", no_research)
    print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# 6. Controlling the Tool Loop Depth
# ---------------------------------------------------------------------------
# maximum_auto_invoke_attempts controls how many tool-call rounds SK allows.
# Default is 5. Set to 1 to allow only a single tool call round.

async def loop_depth_example():
    print("\n--- 6. Loop Depth Control (maximum_auto_invoke_attempts) ---")

    # Allow up to 3 tool-call rounds for a multi-step task
    behavior = FunctionChoiceBehavior.Auto()
    behavior.maximum_auto_invoke_attempts = 3

    prompt = (
        "Please: (1) get today's date, (2) tell me about Semantic Kernel, "
        "and (3) calculate 2024 * 3."
    )
    print(f"User: {prompt}")
    result = await run_with_tools(prompt, behavior, max_tokens=600)
    print(f"Claude:\n{result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await auto_mode_example()
    await multi_step_example()
    await required_mode_example()
    await none_invoke_example()
    await filters_example()
    await loop_depth_example()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • SK 1.40 has NO explicit Planner classes — FunctionChoiceBehavior IS the planner")
    print("  • Auto()       = LLM picks tools freely; SK loops until done")
    print("  • Required()   = LLM must call a tool (guaranteed structured output)")
    print("  • NoneInvoke() = NOT supported by Anthropic; use max_auto_invoke_attempts=0 instead")
    print("  • filters={}   = include/exclude plugins or individual functions")
    print("  • maximum_auto_invoke_attempts controls the tool-call loop depth")
    print("  • Pass kernel= to service.get_chat_message_content() for auto-invoke to work")
    print("  • Next: Lesson 07 — Filters & Middleware (invocation hooks)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
