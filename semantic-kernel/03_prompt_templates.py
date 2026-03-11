"""
Lesson 03 — Prompt Templates
==============================
In Lesson 01 we used raw strings with invoke_prompt.
In Lesson 02 we wrote Python functions as plugins.

Now we bridge the two: SK's *semantic functions* — prompt templates stored
as text that the LLM executes. Variables are injected via {{$var_name}}.

Comparison to what you know:
  CrewAI  → Task(description="...", agent=...) — prompt lives on the Task
  SK      → kernel.add_function(prompt=...) — prompt registered as a Kernel function
  LangGraph → PromptTemplate from LangChain — nearly identical concept

Three ways to create a prompt template in SK:
  1. Inline via kernel.add_function(prompt=...)      — quick, all in code
  2. From file via KernelFunctionFromPrompt           — reads from .txt/.yaml
  3. Chat-style via ChatPromptTemplate               — role-aware multi-turn

This lesson covers (1) in depth, with a taste of (3).

Key template syntax:
  {{$variable}}              — inject a variable
  {{plugin.function_name}}   — call another kernel function inline
  {{- ... -}}               — trim whitespace (optional)
"""

import asyncio
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.prompt_template import PromptTemplateConfig

load_dotenv()

# ---------------------------------------------------------------------------
# Setup Kernel
# ---------------------------------------------------------------------------
kernel = sk.Kernel()
kernel.add_service(
    AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
)

SETTINGS = PromptExecutionSettings(max_tokens=400)

print("=" * 60)
print("Lesson 03 — Prompt Templates")
print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Inline Semantic Function — single variable
# ---------------------------------------------------------------------------
# kernel.add_function() registers a named prompt as a callable kernel function.
# The prompt string uses {{$variable}} placeholders.
# At call time you pass values via KernelArguments.

async def single_variable_example():
    print("\n--- 1. Single Variable Template ---")

    # Register the prompt as a kernel function under plugin "writing", name "summarize"
    summarize_fn = kernel.add_function(
        plugin_name="writing",
        function_name="summarize",
        prompt="Summarize the following text in exactly 2 sentences:\n\n{{$input}}",
        prompt_execution_settings=SETTINGS,
    )

    text = (
        "Semantic Kernel is an open-source SDK from Microsoft that lets developers "
        "combine AI models with plugins, memory, and planning. It supports Python, "
        "C#, and Java, and works with OpenAI, Azure OpenAI, and Anthropic models. "
        "It is designed for enterprise-grade AI application development."
    )

    result = await kernel.invoke(
        plugin_name="writing",
        function_name="summarize",
        arguments=KernelArguments(input=text),
    )
    print(f"Summary:\n{result}")


# ---------------------------------------------------------------------------
# 2. Multi-variable Template
# ---------------------------------------------------------------------------
# Use multiple {{$var}} placeholders in one prompt.

async def multi_variable_example():
    print("\n--- 2. Multi-Variable Template ---")

    kernel.add_function(
        plugin_name="writing",
        function_name="translate",
        prompt=(
            "Translate the following text from {{$source_lang}} to {{$target_lang}}.\n"
            "Return only the translation, nothing else.\n\n"
            "Text: {{$text}}"
        ),
        prompt_execution_settings=SETTINGS,
    )

    result = await kernel.invoke(
        plugin_name="writing",
        function_name="translate",
        arguments=KernelArguments(
            source_lang="English",
            target_lang="French",
            text="The kernel is the central orchestrator in Semantic Kernel.",
        ),
    )
    print(f"Translation: {result}")


# ---------------------------------------------------------------------------
# 3. Reusable Template — called multiple times with different arguments
# ---------------------------------------------------------------------------
# Once registered, a semantic function is reusable like any kernel function.

async def reusable_template_example():
    print("\n--- 3. Reusable Template (called multiple times) ---")

    kernel.add_function(
        plugin_name="writing",
        function_name="explain_to_audience",
        prompt=(
            "Explain '{{$concept}}' to a {{$audience}} in 2-3 sentences. "
            "Use analogies appropriate for that audience."
        ),
        prompt_execution_settings=SETTINGS,
    )

    calls = [
        {"concept": "neural networks", "audience": "10-year-old child"},
        {"concept": "neural networks", "audience": "senior software engineer"},
    ]

    for call in calls:
        result = await kernel.invoke(
            plugin_name="writing",
            function_name="explain_to_audience",
            arguments=KernelArguments(**call),
        )
        print(f"\nFor {call['audience']}:")
        print(f"  {result}")


# ---------------------------------------------------------------------------
# 4. Template Calling Another Plugin Function  {{plugin.function}}
# ---------------------------------------------------------------------------
# SK templates can call registered plugin functions inline.
# The result is injected into the prompt before sending to the LLM.

class DatePlugin:
    """Provides date/time info."""

    @kernel_function(name="today", description="Return today's date as a string.")
    def today(self) -> str:
        from datetime import date
        return date.today().strftime("%B %d, %Y")


async def template_calling_plugin_example():
    print("\n--- 4. Template Calling a Plugin Function ---")

    # Register the helper plugin first
    kernel.add_plugin(DatePlugin(), plugin_name="date")

    # The template calls {{date.today}} at render time — no Python glue needed
    kernel.add_function(
        plugin_name="writing",
        function_name="daily_brief",
        prompt=(
            "Today is {{date.today}}.\n"
            "Write a one-sentence motivational message for a developer "
            "starting their workday on {{$topic}}."
        ),
        prompt_execution_settings=SETTINGS,
    )

    result = await kernel.invoke(
        plugin_name="writing",
        function_name="daily_brief",
        arguments=KernelArguments(topic="AI agents"),
    )
    print(f"Daily brief: {result}")


# ---------------------------------------------------------------------------
# 5. Using PromptTemplateConfig for richer control
# ---------------------------------------------------------------------------
# PromptTemplateConfig lets you define the template name, description,
# input variables with defaults, and execution settings — all in one place.
# This is the "production" way to define semantic functions.

async def prompt_template_config_example():
    print("\n--- 5. PromptTemplateConfig (structured definition) ---")

    config = PromptTemplateConfig(
        name="code_review",
        description="Reviews a code snippet and gives structured feedback.",
        template=(
            "You are an expert {{$language}} developer.\n"
            "Review the following code and provide:\n"
            "1. What it does (1 sentence)\n"
            "2. One strength\n"
            "3. One improvement suggestion\n\n"
            "Code:\n```{{$language}}\n{{$code}}\n```"
        ),
        template_format="semantic-kernel",
        execution_settings={"anthropic_chat": SETTINGS},
    )

    review_fn = kernel.add_function(
        plugin_name="engineering",
        function_name="code_review",
        prompt_template_config=config,
    )

    code_snippet = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    result = await kernel.invoke(
        plugin_name="engineering",
        function_name="code_review",
        arguments=KernelArguments(language="Python", code=code_snippet.strip()),
    )
    print(f"Code Review:\n{result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await single_variable_example()
    await multi_variable_example()
    await reusable_template_example()
    await template_calling_plugin_example()
    await prompt_template_config_example()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  • kernel.add_function(prompt=...) creates a semantic function")
    print("  • {{$variable}} injects a value at call time via KernelArguments")
    print("  • {{plugin.function}} calls another plugin inline inside a template")
    print("  • Semantic functions are reusable — register once, call many times")
    print("  • PromptTemplateConfig = structured, production-grade definition")
    print("  • Next: Lesson 04 — Chat History & Conversation State")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
