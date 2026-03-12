"""
Lesson 09 — Multi-Agent Collaboration (AgentGroupChat)
========================================================
AgentGroupChat is SK's built-in multi-agent runtime.
Multiple ChatCompletionAgents participate in a shared conversation,
each taking turns responding until a termination condition is met.

Comparison to what you know:
  LangGraph  → multiple nodes with conditional edges; you wire routing explicitly
  CrewAI     → Crew(agents=[...], tasks=[...], process=Process.sequential/hierarchical)
  SK 1.40    → AgentGroupChat(agents=[...], selection_strategy, termination_strategy)
               The kernel handles turn-taking; you configure the rules.

Key classes:
  AgentGroupChat               — the multi-agent runtime
  SequentialSelectionStrategy  — round-robin: each agent takes turns in order
  KernelFunctionSelectionStrategy  — LLM prompt decides which agent speaks next
  KernelFunctionTerminationStrategy — LLM prompt decides when to stop
  DefaultTerminationStrategy   — stop after N iterations (simplest)

Usage pattern:
  chat = AgentGroupChat(agents=[a, b], ...)
  await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content="..."))
  async for msg in chat.invoke():
      print(f"{msg.name}: {msg.content}")

Topics covered:
  1. Round-robin chat (SequentialSelectionStrategy + max iterations)
  2. Writer + Critic collaboration (round-robin with termination keyword)
  3. LLM-driven agent selection (KernelFunctionSelectionStrategy)
  4. LLM-driven termination (KernelFunctionTerminationStrategy)
"""

import asyncio
from typing import Annotated
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
    DefaultTerminationStrategy,
)
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import kernel_function, KernelArguments, KernelFunctionMetadata

load_dotenv()

print("=" * 62)
print("Lesson 09 — Multi-Agent Collaboration (AgentGroupChat)")
print("=" * 62)

SERVICE_ID = "anthropic_chat"


def make_kernel() -> sk.Kernel:
    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id=SERVICE_ID,
    ))
    return kernel


def auto_args(max_tokens: int = 400) -> KernelArguments:
    return KernelArguments(settings=PromptExecutionSettings(
        max_tokens=max_tokens,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    ))


# ---------------------------------------------------------------------------
# 1. Round-Robin Chat — SequentialSelectionStrategy
# ---------------------------------------------------------------------------
# Agents take turns in the order they were registered.
# DefaultTerminationStrategy(maximum_iterations=N) stops after N total turns.
# This is the simplest multi-agent pattern — no LLM needed for routing.
#
# LangGraph equivalent: cycle through nodes in a fixed order
# CrewAI equivalent:    Process.sequential with agents taking each task in turn

async def round_robin_example():
    print("\n--- 1. Round-Robin Chat (SequentialSelectionStrategy) ---")

    kernel = make_kernel()

    optimist = ChatCompletionAgent(
        kernel=kernel,
        name="Optimist",
        instructions=(
            "You are an enthusiastic optimist. In 1-2 sentences, highlight the positive "
            "side of whatever topic is raised. Be upbeat and encouraging."
        ),
    )
    pessimist = ChatCompletionAgent(
        kernel=kernel,
        name="Pessimist",
        instructions=(
            "You are a cautious realist. In 1-2 sentences, raise one potential concern "
            "or downside of whatever has been said. Be constructive, not doom-and-gloom."
        ),
    )
    moderator = ChatCompletionAgent(
        kernel=kernel,
        name="Moderator",
        instructions=(
            "You are a balanced moderator. In 1 sentence, synthesise the last two "
            "perspectives into a balanced summary. Then pose one follow-up question."
        ),
    )

    chat = AgentGroupChat(
        agents=[optimist, pessimist, moderator],
        selection_strategy=SequentialSelectionStrategy(),
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=6),
    )

    topic = "Is remote work better than office work for software engineers?"
    await chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=topic)
    )

    print(f"Topic: {topic}\n")
    async for msg in chat.invoke():
        print(f"  [{msg.name}]: {msg.content}\n")


# ---------------------------------------------------------------------------
# 2. Writer + Critic Collaboration
# ---------------------------------------------------------------------------
# A Writer produces content; a Critic reviews it.
# They alternate until the Critic is satisfied (outputs "APPROVED").
# We use KernelFunctionTerminationStrategy with a simple approval prompt.
#
# This is the "reflection" pattern — one agent generates, another judges.
# CrewAI equivalent: two Agents with sequential Tasks and context passing.

async def writer_critic_example():
    print("\n--- 2. Writer + Critic (reflection pattern) ---")

    # Two separate kernels — agents can share one, but separate is also fine
    kernel = make_kernel()

    writer = ChatCompletionAgent(
        kernel=kernel,
        name="Writer",
        instructions=(
            "You are a technical writer. Write a clear, concise 3-sentence description "
            "of the concept the user asks about. If the Critic has given feedback, "
            "incorporate it in your next version."
        ),
    )
    critic = ChatCompletionAgent(
        kernel=kernel,
        name="Critic",
        instructions=(
            "You are a senior technical editor. Review the Writer's output. "
            "If it is clear and accurate, respond ONLY with: APPROVED. "
            "Otherwise give ONE specific improvement suggestion (1 sentence)."
        ),
    )

    # Termination: stop when Critic's message contains "APPROVED"
    termination_kernel = make_kernel()
    termination_fn = termination_kernel.add_function(
        plugin_name="termination",
        function_name="check",
        prompt=(
            "Determine if the conversation should end.\n"
            "The last message was: {{$last_message}}\n"
            "Reply with only 'yes' if it contains 'APPROVED', otherwise 'no'."
        ),
        prompt_execution_settings=PromptExecutionSettings(max_tokens=5),
    )

    termination_strategy = KernelFunctionTerminationStrategy(
        agents=[critic],          # only check termination after Critic speaks
        function=termination_fn,
        kernel=termination_kernel,
        result_parser=lambda r: str(r.value[0]).strip().lower() == "yes",
        history_variable_name="last_message",
        maximum_iterations=8,
    )

    chat = AgentGroupChat(
        agents=[writer, critic],
        selection_strategy=SequentialSelectionStrategy(),
        termination_strategy=termination_strategy,
    )

    task = "Explain what a Semantic Kernel filter is and why it's useful."
    await chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=task)
    )

    print(f"Task: {task}\n")
    async for msg in chat.invoke():
        print(f"  [{msg.name}]: {msg.content}\n")


# ---------------------------------------------------------------------------
# 3. LLM-Driven Agent Selection — KernelFunctionSelectionStrategy
# ---------------------------------------------------------------------------
# Instead of fixed turn order, a prompt decides which agent should speak next.
# The prompt receives the conversation history and returns an agent name.
#
# This is the SK equivalent of a "manager" or "supervisor" pattern.
# LangGraph equivalent: conditional edge with a routing function.
# CrewAI equivalent:    Process.hierarchical with a manager_llm.

async def llm_selection_example():
    print("\n--- 3. LLM-Driven Agent Selection (KernelFunctionSelectionStrategy) ---")

    kernel = make_kernel()

    researcher = ChatCompletionAgent(
        kernel=kernel,
        name="Researcher",
        instructions=(
            "You are a researcher. When asked a factual question about AI frameworks, "
            "provide a concise 2-sentence factual answer."
        ),
    )
    explainer = ChatCompletionAgent(
        kernel=kernel,
        name="Explainer",
        instructions=(
            "You are a teacher. When the Researcher has given facts, "
            "restate them in simple terms that a beginner would understand. "
            "Use an analogy."
        ),
    )
    summariser = ChatCompletionAgent(
        kernel=kernel,
        name="Summariser",
        instructions=(
            "You are a summariser. After the Explainer has simplified the facts, "
            "give a single bullet-point summary."
        ),
    )

    agents = [researcher, explainer, summariser]
    agent_names = ", ".join(a.name for a in agents)

    # Selection prompt: given history, return exactly one agent name
    selection_kernel = make_kernel()
    selection_fn = selection_kernel.add_function(
        plugin_name="selection",
        function_name="pick_agent",
        prompt=(
            f"You are coordinating a discussion between: {agent_names}.\n"
            "The conversation so far:\n{{$history}}\n\n"
            "Who should speak next? Consider:\n"
            "  - If no one has spoken yet → Researcher\n"
            "  - If Researcher just spoke → Explainer\n"
            "  - If Explainer just spoke → Summariser\n"
            f"Reply with ONLY one of: {agent_names}"
        ),
        prompt_execution_settings=PromptExecutionSettings(max_tokens=10),
    )

    selection_strategy = KernelFunctionSelectionStrategy(
        function=selection_fn,
        kernel=selection_kernel,
        result_parser=lambda r: str(r.value[0]).strip(),
        history_variable_name="history",
    )

    chat = AgentGroupChat(
        agents=agents,
        selection_strategy=selection_strategy,
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=3),
    )

    question = "What is the difference between SK plugins and LangChain tools?"
    await chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=question)
    )

    print(f"Question: {question}\n")
    async for msg in chat.invoke():
        print(f"  [{msg.name}]: {msg.content}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await round_robin_example()
    await writer_critic_example()
    await llm_selection_example()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • AgentGroupChat(agents, selection_strategy, termination_strategy)")
    print("  • SequentialSelectionStrategy  — round-robin turn order")
    print("  • DefaultTerminationStrategy   — stop after N iterations (simplest)")
    print("  • KernelFunctionTerminationStrategy — LLM prompt decides when to stop")
    print("  • KernelFunctionSelectionStrategy   — LLM prompt picks next agent")
    print("  • Add user message first: chat.add_chat_message(ChatMessageContent(...))")
    print("  • Iterate: async for msg in chat.invoke() → each agent's response")
    print("  • Writer+Critic = reflection pattern (generate → review → revise)")
    print("  • LLM selection = supervisor/manager pattern (no fixed routing)")
    print("  • Next: Lesson 10 — Human-in-the-Loop (HITL)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
