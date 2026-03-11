"""
Lesson 04 — Chat History & Conversation State
===============================================
In Lesson 01 we briefly touched ChatHistory for multi-turn chat.
This lesson goes deep on managing conversation state — the SK-native way
to build stateful, context-aware chatbots and assistants.

Comparison to what you know:
  LangGraph → MessagesState / add_messages reducer — messages are graph state
  CrewAI    → Task context=[] wiring — sequential context passing between tasks
  SK        → ChatHistory — an ordered list of ChatMessageContent objects
              The developer owns the history; SK doesn't auto-manage it for you.

ChatHistory is essentially a typed list of messages:
  - SystemChatMessage    (role="system")
  - UserChatMessage      (role="user")
  - AssistantChatMessage (role="assistant")
  - ToolCallResultMessage (role="tool") — covered in Lesson 05

Key patterns covered here:
  1. Build a multi-turn assistant with persistent history
  2. System prompt injection and persona control
  3. Trimming / windowing history to stay within context limits
  4. Saving and restoring chat history (serialization)
  5. Injecting history into a prompt template
"""

import asyncio
import json
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import KernelArguments

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

SETTINGS = PromptExecutionSettings(max_tokens=400)

print("=" * 60)
print("Lesson 04 — Chat History & Conversation State")
print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Basic Multi-Turn Conversation
# ---------------------------------------------------------------------------
# ChatHistory is a list you control. After each LLM reply you append the
# assistant message so the next call has full context.
# This is the same pattern as Lesson 01 but now we go deeper.

async def basic_multi_turn():
    print("\n--- 1. Basic Multi-Turn Conversation ---")

    history = ChatHistory()
    history.add_system_message(
        "You are a concise Python tutor. Keep answers under 3 sentences."
    )

    questions = [
        "What is a decorator in Python?",
        "Show me the simplest possible decorator example.",
        "How does that relate to @kernel_function in Semantic Kernel?",
    ]

    for question in questions:
        print(f"\nUser: {question}")
        history.add_user_message(question)

        response = await service.get_chat_message_content(
            chat_history=history,
            settings=SETTINGS,
        )
        print(f"Claude: {response}")

        # IMPORTANT: append assistant reply so next turn has context
        history.add_assistant_message(str(response))

    print(f"\n[History has {len(history)} messages total]")


# ---------------------------------------------------------------------------
# 2. Persona Control via System Prompt
# ---------------------------------------------------------------------------
# The system message defines the assistant's persona, constraints, and format.
# Changing it completely changes the tone/behaviour without touching the logic.

async def persona_control():
    print("\n--- 2. Persona Control via System Prompt ---")

    personas = [
        {
            "name": "Enthusiastic Intern",
            "system": (
                "You are an enthusiastic junior developer who uses lots of exclamation marks "
                "and emojis. Answer in 2 sentences max."
            ),
        },
        {
            "name": "Gruff Senior Engineer",
            "system": (
                "You are a terse senior engineer who gives only the essential facts. "
                "No fluff. One sentence max."
            ),
        },
    ]

    question = "What is dependency injection?"

    for persona in personas:
        history = ChatHistory()
        history.add_system_message(persona["system"])
        history.add_user_message(question)

        response = await service.get_chat_message_content(
            chat_history=history,
            settings=SETTINGS,
        )
        print(f"\n[{persona['name']}]: {response}")


# ---------------------------------------------------------------------------
# 3. History Windowing (context limit management)
# ---------------------------------------------------------------------------
# LLMs have finite context windows. For long conversations you must trim old
# messages. The simplest strategy: keep the system message + last N turns.
#
# LangGraph equivalent: MessagesState with a trim_messages node
# CrewAI: no direct equivalent — tasks are stateless by design

def trim_history(history: ChatHistory, max_pairs: int = 3) -> ChatHistory:
    """
    Keep the system message (if any) + the last `max_pairs` user/assistant pairs.
    Returns a new ChatHistory — does not mutate the original.
    """
    trimmed = ChatHistory()

    # Preserve system message
    if history.messages and history.messages[0].role == AuthorRole.SYSTEM:
        trimmed.add_system_message(str(history.messages[0].content))

    # Keep last max_pairs * 2 non-system messages
    non_system = [m for m in history.messages if m.role != AuthorRole.SYSTEM]
    recent = non_system[-(max_pairs * 2):]

    for msg in recent:
        if msg.role == AuthorRole.USER:
            trimmed.add_user_message(str(msg.content))
        elif msg.role == AuthorRole.ASSISTANT:
            trimmed.add_assistant_message(str(msg.content))

    return trimmed


async def history_windowing():
    print("\n--- 3. History Windowing ---")

    history = ChatHistory()
    history.add_system_message("You are a helpful assistant. Be brief.")

    # Simulate a long conversation (5 turns)
    fake_turns = [
        ("What is Python?",        "A high-level interpreted programming language."),
        ("What is async/await?",   "Syntax for writing asynchronous code."),
        ("What is a generator?",   "A function that yields values lazily."),
        ("What is a context manager?", "An object managing resource setup/teardown via with."),
        ("What is a metaclass?",   "A class whose instances are classes themselves."),
    ]

    for user_msg, assistant_reply in fake_turns:
        history.add_user_message(user_msg)
        history.add_assistant_message(assistant_reply)

    print(f"Full history length: {len(history)} messages")

    # Trim to last 2 pairs
    windowed = trim_history(history, max_pairs=2)
    print(f"Windowed history length: {len(windowed)} messages")

    # Now ask a follow-up — Claude only sees the last 2 pairs + system
    windowed.add_user_message(
        "Based on our conversation so far, what was the last topic we discussed?"
    )
    response = await service.get_chat_message_content(
        chat_history=windowed,
        settings=SETTINGS,
    )
    print(f"Claude (with windowed history): {response}")


# ---------------------------------------------------------------------------
# 4. Serialize / Deserialize ChatHistory
# ---------------------------------------------------------------------------
# Persist history to JSON so conversations survive process restarts.
# LangGraph equivalent: SQLite checkpointer — same idea, different API.

def serialize_history(history: ChatHistory) -> str:
    """Serialize ChatHistory to a JSON string."""
    messages = []
    for msg in history.messages:
        messages.append({
            "role": str(msg.role),
            "content": str(msg.content),
        })
    return json.dumps(messages, indent=2)


def deserialize_history(json_str: str) -> ChatHistory:
    """Restore a ChatHistory from a JSON string."""
    history = ChatHistory()
    for item in json.loads(json_str):
        role = item["role"]
        content = item["content"]
        if "system" in role:
            history.add_system_message(content)
        elif "user" in role:
            history.add_user_message(content)
        elif "assistant" in role:
            history.add_assistant_message(content)
    return history


async def serialize_example():
    print("\n--- 4. Serialize / Deserialize ChatHistory ---")

    # Build a small history
    history = ChatHistory()
    history.add_system_message("You are a helpful assistant.")
    history.add_user_message("Remember: my favourite language is Python.")
    history.add_assistant_message("Got it! I'll keep that in mind.")

    # Serialize
    json_str = serialize_history(history)
    print("Serialized history (JSON):")
    print(json_str)

    # Restore and continue
    restored = deserialize_history(json_str)
    restored.add_user_message("What is my favourite programming language?")

    response = await service.get_chat_message_content(
        chat_history=restored,
        settings=SETTINGS,
    )
    print(f"\nClaude (from restored history): {response}")


# ---------------------------------------------------------------------------
# 5. Chat History inside a Prompt Template
# ---------------------------------------------------------------------------
# SK 1.x blocks passing a ChatHistory object directly as a {{$variable}} —
# it's a security guard against prompt injection from arbitrary objects.
#
# The correct pattern: serialise the history to a plain string first,
# then pass that string as the template variable.
# The LLM sees the conversation as formatted text; the kernel stays safe.

def history_to_text(history: ChatHistory) -> str:
    """Format ChatHistory as plain text for use in a prompt template."""
    lines = []
    for msg in history.messages:
        role = str(msg.role).split(".")[-1].capitalize()  # e.g. "User", "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


async def history_in_template():
    print("\n--- 5. Chat History in a Prompt Template ---")

    kernel.add_function(
        plugin_name="chat",
        function_name="assistant",
        prompt=(
            "You are a concise technical assistant.\n\n"
            "Conversation so far:\n{{$chat_history_text}}\n\n"
            "Now answer this question: {{$user_input}}"
        ),
        prompt_execution_settings=SETTINGS,
    )

    # Build some prior context
    history = ChatHistory()
    history.add_user_message("I'm building an AI assistant in Python.")
    history.add_assistant_message("Great! What framework are you using?")
    history.add_user_message("Semantic Kernel.")
    history.add_assistant_message("Excellent choice for enterprise applications!")

    # Serialise to string — SK templates only accept str/int/float variables
    history_text = history_to_text(history)

    result = await kernel.invoke(
        plugin_name="chat",
        function_name="assistant",
        arguments=KernelArguments(
            chat_history_text=history_text,
            user_input="What are the main SK concepts I should learn first?",
        ),
    )
    print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await basic_multi_turn()
    await persona_control()
    await history_windowing()
    await serialize_example()
    await history_in_template()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  • ChatHistory is an ordered list you control — SK doesn't manage it")
    print("  • Always append assistant replies to maintain context")
    print("  • System message = persona / constraints for the conversation")
    print("  • Trim history to avoid hitting context limits (window strategy)")
    print("  • Serialize/deserialize history = poor-man's persistence")
    print("  • {{$chat_history}} embeds history directly in prompt templates")
    print("  • Next: Lesson 05 — Semantic Memory + Embeddings")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
