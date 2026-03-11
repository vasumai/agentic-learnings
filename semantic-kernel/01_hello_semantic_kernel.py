"""
Lesson 01 — Hello Semantic Kernel
==================================
Semantic Kernel (SK) is Microsoft's open-source SDK for building AI agents
and orchestrating LLM calls in Python, Java, and C#.

Core concepts introduced here:
  Kernel        — the central orchestrator; holds services, plugins, memory
  ChatHistory   — structured conversation history (system/user/assistant turns)
  ChatCompletionService — the AI backend (OpenAI, Anthropic, Azure OpenAI, etc.)
  invoke_prompt — SK's built-in shortcut for one-shot prompt calls

Comparison to what you know:
  CrewAI  → Crew(agents, tasks)  runs a multi-agent pipeline
  SK      → Kernel.invoke() / invoke_prompt() calls registered functions/services
  LangGraph → Graph(nodes, edges) routes messages between nodes
"""

import asyncio
from dotenv import load_dotenv
import os

# Semantic Kernel imports
import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Create the Kernel
# ---------------------------------------------------------------------------
# The Kernel is the heart of SK. Think of it as a dependency-injection
# container that wires together: AI services, plugins, memory, and filters.
kernel = sk.Kernel()

# ---------------------------------------------------------------------------
# 2. Register a Chat Completion Service
# ---------------------------------------------------------------------------
# SK supports many backends. Here we use Anthropic (Claude).
# The service_id lets you register multiple services and pick one per call.
chat_service = AnthropicChatCompletion(
    ai_model_id="claude-sonnet-4-6",          # model to use
    api_key=os.environ["ANTHROPIC_API_KEY"],
    service_id="anthropic_chat",              # identifier within the kernel
)
kernel.add_service(chat_service)

print("=" * 55)
print("Lesson 01 — Hello Semantic Kernel")
print("=" * 55)


# ---------------------------------------------------------------------------
# 3. Direct chat completion with ChatHistory
# ---------------------------------------------------------------------------
# ChatHistory tracks the conversation: system prompt + alternating user/assistant.
# This is the low-level approach — useful when you need full control.

async def direct_chat_example():
    print("\n--- 3. Direct Chat Completion ---")

    history = ChatHistory()
    history.add_system_message(
        "You are a concise AI tutor. Answer in 2-3 sentences max."
    )
    history.add_user_message("What is Semantic Kernel in one paragraph?")

    # Retrieve the service we registered and call it directly
    service: AnthropicChatCompletion = kernel.get_service("anthropic_chat")
    settings = PromptExecutionSettings(max_tokens=200)

    result = await service.get_chat_message_content(
        chat_history=history,
        settings=settings,
    )
    print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# 4. invoke_prompt — SK's high-level semantic function shortcut
# ---------------------------------------------------------------------------
# invoke_prompt wraps a prompt string into an in-line "semantic function"
# and runs it through the default registered service. Much less boilerplate
# for one-shot calls — analogous to chain.invoke() in LangChain.

async def invoke_prompt_example():
    print("\n--- 4. invoke_prompt (Semantic Function) ---")

    # Double-brace {{ }} for literal braces; single {{$var}} for template vars.
    # Here we use a plain string — no variables yet (that's Lesson 03).
    result = await kernel.invoke_prompt(
        prompt="List 3 key differences between LangChain and Semantic Kernel. Be brief.",
        settings=PromptExecutionSettings(max_tokens=300),
    )
    print(f"Claude:\n{result}")


# ---------------------------------------------------------------------------
# 5. Multi-turn conversation
# ---------------------------------------------------------------------------
# Build a simple back-and-forth by appending to ChatHistory.

async def multi_turn_example():
    print("\n--- 5. Multi-Turn Conversation ---")

    history = ChatHistory()
    history.add_system_message("You are a helpful Python tutor. Be concise.")

    turns = [
        "What is async/await in Python?",
        "Give me the shortest possible working example.",
    ]

    service: AnthropicChatCompletion = kernel.get_service("anthropic_chat")
    settings = PromptExecutionSettings(max_tokens=250)

    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        history.add_user_message(user_msg)

        response = await service.get_chat_message_content(
            chat_history=history,
            settings=settings,
        )
        print(f"Claude: {response}")

        # Append assistant reply so the next turn has full context
        history.add_assistant_message(str(response))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await direct_chat_example()
    await invoke_prompt_example()
    await multi_turn_example()

    print("\n" + "=" * 55)
    print("Key Takeaways:")
    print("  • Kernel = central DI container for services + plugins")
    print("  • ChatHistory = typed message list (system/user/assistant)")
    print("  • invoke_prompt = quick one-shot call (no setup needed)")
    print("  • Direct service call = full control over history & settings")
    print("  • Next: Lesson 02 — Plugins (native functions in SK)")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
