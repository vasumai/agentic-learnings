"""
Lesson 07 — Filters & Middleware
==================================
Filters are SK's middleware / interceptor system. They let you hook into
the execution pipeline at three points:

  FUNCTION_INVOCATION    — wraps every @kernel_function call
  PROMPT_RENDERING       — wraps every prompt template render
  AUTO_FUNCTION_INVOCATION — wraps every tool call in the auto-invoke loop

Think of them as decorators around SK's internal execution, not around your
own functions. The pattern is always:

    async def my_filter(context, next):
        # --- BEFORE ---
        await next(context)     # call the actual function / next filter in chain
        # --- AFTER ---

This is identical to:
  LangGraph  → Callbacks (on_llm_start / on_tool_start / on_tool_end)
  CrewAI     → No direct equivalent (CrewAI has callbacks via LangChain)
  SK 1.40    → kernel.add_filter(FilterTypes.X, async_fn) — first-class, typed

Three filter types and their contexts:
  FunctionInvocationContext     → .function, .arguments, .result
  PromptRenderContext           → .function, .rendered_prompt, .function_result
  AutoFunctionInvocationContext → .function, .function_call_content, .terminate,
                                  .request_sequence_index, .function_sequence_index

Practical uses covered here:
  1. Logging — record every function call + result
  2. Timing — measure execution time of each plugin function
  3. Input validation / sanitisation — modify arguments before the call
  4. Output modification — rewrite results after the call
  5. Prompt inspection — log or modify the rendered prompt before LLM call
  6. Auto-invoke guard — block specific tool calls in the auto-invoke loop
  7. Multiple filters chained — execution order matters
"""

import asyncio
import time
from typing import Annotated, Callable, Awaitable
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
from semantic_kernel.filters.prompts.prompt_render_context import PromptRenderContext
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.functions import kernel_function, KernelArguments

load_dotenv()

print("=" * 62)
print("Lesson 07 — Filters & Middleware")
print("=" * 62)


# ---------------------------------------------------------------------------
# Plugin used throughout the lesson
# ---------------------------------------------------------------------------
class CalculatorPlugin:
    @kernel_function(name="add", description="Add two numbers.")
    def add(
        self,
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> Annotated[float, "Sum"]:
        return a + b

    @kernel_function(name="divide", description="Divide a by b.")
    def divide(
        self,
        a: Annotated[float, "Numerator"],
        b: Annotated[float, "Denominator"],
    ) -> Annotated[str, "Result or error"]:
        if b == 0:
            return "Error: division by zero"
        return str(a / b)

    @kernel_function(name="secret_function", description="A function that should never run.")
    def secret_function(self) -> str:
        return "TOP SECRET DATA"


# ---------------------------------------------------------------------------
# 1. Logging Filter  (FUNCTION_INVOCATION)
# ---------------------------------------------------------------------------
# Called for every kernel.invoke() and @kernel_function call.
# Logs function name, arguments, and result.

def make_logging_filter():
    """Returns a FUNCTION_INVOCATION filter that logs calls and results."""

    async def logging_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        plugin = context.function.plugin_name
        fn     = context.function.name
        args   = dict(context.arguments) if context.arguments else {}

        print(f"  [LOG] → Calling {plugin}.{fn}  args={args}")
        await next(context)   # execute the actual function
        print(f"  [LOG] ← Result: {context.result}")

    return logging_filter


async def logging_filter_example():
    print("\n--- 1. Logging Filter (FUNCTION_INVOCATION) ---")

    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    kernel.add_plugin(CalculatorPlugin(), plugin_name="calc")
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_logging_filter())

    await kernel.invoke(plugin_name="calc", function_name="add", arguments=KernelArguments(a=10, b=20))
    await kernel.invoke(plugin_name="calc", function_name="divide", arguments=KernelArguments(a=100, b=4))


# ---------------------------------------------------------------------------
# 2. Timing Filter  (FUNCTION_INVOCATION)
# ---------------------------------------------------------------------------
# Measures wall-clock time for every plugin function call.

def make_timing_filter():
    async def timing_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        start = time.perf_counter()
        await next(context)
        elapsed_ms = (time.perf_counter() - start) * 1000
        fn = f"{context.function.plugin_name}.{context.function.name}"
        print(f"  [TIMER] {fn} took {elapsed_ms:.2f}ms")

    return timing_filter


async def timing_filter_example():
    print("\n--- 2. Timing Filter ---")

    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    kernel.add_plugin(CalculatorPlugin(), plugin_name="calc")
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_timing_filter())

    for a, b in [(1, 2), (999, 333), (0, 0)]:
        result = await kernel.invoke(plugin_name="calc", function_name="divide", arguments=KernelArguments(a=a, b=b))
        print(f"  divide({a}, {b}) = {result}")


# ---------------------------------------------------------------------------
# 3. Input Sanitisation Filter  (FUNCTION_INVOCATION)
# ---------------------------------------------------------------------------
# Modify arguments BEFORE the function runs.
# Example: clamp values to a safe range.

def make_clamp_filter(max_value: float = 1000.0):
    """Clamp any numeric argument above max_value back to max_value."""

    async def clamp_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        if context.arguments:
            for key, val in context.arguments.items():
                try:
                    if float(val) > max_value:
                        print(f"  [CLAMP] {key}={val} clamped to {max_value}")
                        context.arguments[key] = max_value
                except (TypeError, ValueError):
                    pass
        await next(context)

    return clamp_filter


async def input_sanitisation_example():
    print("\n--- 3. Input Sanitisation Filter ---")

    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    kernel.add_plugin(CalculatorPlugin(), plugin_name="calc")
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_clamp_filter(max_value=1000.0))

    # This value exceeds the clamp threshold
    result = await kernel.invoke(plugin_name="calc", function_name="add", arguments=KernelArguments(a=5000, b=200))
    print(f"  add(5000, 200) with clamp → {result}")


# ---------------------------------------------------------------------------
# 4. Output Modification Filter  (FUNCTION_INVOCATION)
# ---------------------------------------------------------------------------
# Intercept and rewrite the result AFTER the function runs.
# Example: add a currency symbol to all numeric results.

def make_currency_filter():
    async def currency_filter(
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        await next(context)   # run the function first
        # Rewrite the result if it's a number
        if context.result is not None:
            try:
                value = float(str(context.result))
                from semantic_kernel.functions import FunctionResult
                context.result = FunctionResult(
                    function=context.result.function,
                    value=f"${value:,.2f}",
                )
            except (ValueError, AttributeError):
                pass  # not a number — leave result as-is

    return currency_filter


async def output_modification_example():
    print("\n--- 4. Output Modification Filter ---")

    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    kernel.add_plugin(CalculatorPlugin(), plugin_name="calc")
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_currency_filter())

    result = await kernel.invoke(plugin_name="calc", function_name="add", arguments=KernelArguments(a=1200, b=350))
    print(f"  add(1200, 350) with currency format → {result}")


# ---------------------------------------------------------------------------
# 5. Prompt Inspection Filter  (PROMPT_RENDERING)
# ---------------------------------------------------------------------------
# See the fully rendered prompt string before it is sent to the LLM.
# Useful for: debugging, PII scrubbing, prompt logging.

def make_prompt_logger():
    async def prompt_logger(
        context: PromptRenderContext,
        next: Callable[[PromptRenderContext], Awaitable[None]],
    ) -> None:
        await next(context)   # render the prompt first
        prompt = context.rendered_prompt or "(no rendered prompt)"
        print(f"  [PROMPT] fn={context.function.name}")
        # Truncate for display — real prompts can be long
        preview = prompt[:200].replace("\n", " ↵ ")
        print(f"  [PROMPT] rendered: {preview}{'...' if len(prompt) > 200 else ''}")

    return prompt_logger


async def prompt_inspection_example():
    print("\n--- 5. Prompt Inspection Filter (PROMPT_RENDERING) ---")

    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    kernel.add_filter(FilterTypes.PROMPT_RENDERING, make_prompt_logger())

    kernel.add_function(
        plugin_name="demo",
        function_name="greet",
        prompt="Say hello to {{$name}} in {{$language}}. Be brief.",
        prompt_execution_settings=PromptExecutionSettings(max_tokens=50),
    )

    result = await kernel.invoke(
        plugin_name="demo", function_name="greet",
        arguments=KernelArguments(name="Srini", language="French"),
    )
    print(f"  Claude: {result}")


# ---------------------------------------------------------------------------
# 6. Auto-Invoke Guard Filter  (AUTO_FUNCTION_INVOCATION)
# ---------------------------------------------------------------------------
# Intercepts each tool call in the auto-invoke loop.
# You can inspect what tool is about to run and terminate the loop or skip it.
# This is the foundation for HITL (Lesson 10).

def make_guard_filter(blocked_functions: list[str]):
    """Block specific functions by name during auto-invocation."""

    async def guard_filter(
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name
        if fn_name in blocked_functions:
            print(f"  [GUARD] BLOCKED: {context.function.plugin_name}.{fn_name}")
            # Set terminate=True to stop the entire auto-invoke loop
            context.terminate = True
            return   # don't call next — skip this tool
        print(f"  [GUARD] ALLOWED: {context.function.plugin_name}.{fn_name}")
        await next(context)

    return guard_filter


async def auto_invoke_guard_example():
    print("\n--- 6. Auto-Invoke Guard (AUTO_FUNCTION_INVOCATION) ---")

    kernel = sk.Kernel()
    service = AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
    kernel.add_service(service)
    kernel.add_plugin(CalculatorPlugin(), plugin_name="calc")

    # Block 'secret_function' from ever being auto-invoked
    kernel.add_filter(
        FilterTypes.AUTO_FUNCTION_INVOCATION,
        make_guard_filter(blocked_functions=["secret_function"]),
    )

    history = ChatHistory()
    history.add_user_message("What is 7 + 8? Also call secret_function.")

    settings = PromptExecutionSettings(
        max_tokens=200,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
    response = await service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )
    print(f"  Claude: {response}")


# ---------------------------------------------------------------------------
# 7. Chaining Multiple Filters
# ---------------------------------------------------------------------------
# Filters run in registration order (first in, first called).
# Each filter wraps the next — like nested decorators.
# Order: Filter A → Filter B → actual function → Filter B → Filter A

async def chained_filters_example():
    print("\n--- 7. Chained Filters (execution order) ---")

    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    kernel.add_plugin(CalculatorPlugin(), plugin_name="calc")

    # Register in order: clamp first, then log, then time
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_clamp_filter(max_value=500.0))
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_logging_filter())
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, make_timing_filter())

    print("  Calling add(9999, 1) — clamp runs first, then log, then timer:")
    result = await kernel.invoke(plugin_name="calc", function_name="add", arguments=KernelArguments(a=9999, b=1))
    print(f"  Final result: {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await logging_filter_example()
    await timing_filter_example()
    await input_sanitisation_example()
    await output_modification_example()
    await prompt_inspection_example()
    await auto_invoke_guard_example()
    await chained_filters_example()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • kernel.add_filter(FilterTypes.X, async_fn) — three filter types")
    print("  • Filter signature: async def f(context, next) — always await next()")
    print("  • FUNCTION_INVOCATION: context.arguments (before) / context.result (after)")
    print("  • PROMPT_RENDERING:    context.rendered_prompt (after next())")
    print("  • AUTO_FUNCTION_INVOCATION: context.terminate=True stops the tool loop")
    print("  • Multiple filters chain in registration order — outermost first")
    print("  • Filters = SK's answer to LangGraph callbacks / middleware")
    print("  • Next: Lesson 08 — ChatCompletionAgent (the modern SK agent)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
