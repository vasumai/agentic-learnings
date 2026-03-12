"""
Lesson 09 — Multi-Agent Collaboration
=======================================
NOTE ON AgentGroupChat + Anthropic:
  SK's built-in AgentGroupChat uses a shared ChatHistoryChannel where agents
  append their responses as "assistant" messages. Anthropic's API rejects any
  request where the last message is an assistant message ("assistant message
  prefill" is not supported). This makes AgentGroupChat incompatible with
  Anthropic out of the box.

  The Anthropic-compatible solution: MANUAL ORCHESTRATION.
  Each agent turn is driven by a fresh user message that summarises the
  conversation so far.  The conversation always ends with a user message,
  so Anthropic is happy.  This is also more transparent and educational.

Comparison to what you know:
  LangGraph  → nodes with conditional edges; you wire routing explicitly
  CrewAI     → Crew(agents, tasks, process=sequential/hierarchical)
  SK+Anthropic → manual orchestration loop:
                 build context string → call agent → collect output → repeat

Patterns covered:
  1. Round-robin — fixed turn order, each agent sees all previous responses
  2. Writer + Critic — two-agent reflection loop until Critic approves
  3. LLM-driven routing — a selector agent decides who speaks next
"""

import asyncio
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelArguments

load_dotenv()

print("=" * 62)
print("Lesson 09 — Multi-Agent Collaboration")
print("=" * 62)


def make_kernel() -> sk.Kernel:
    kernel = sk.Kernel()
    kernel.add_service(AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    ))
    return kernel


def base_args(max_tokens: int = 300) -> KernelArguments:
    return KernelArguments(settings=PromptExecutionSettings(max_tokens=max_tokens))


# ---------------------------------------------------------------------------
# Core helper: call one agent with a fully-constructed user message
# ---------------------------------------------------------------------------
# The key insight: agent.get_response(messages=<string>) creates a fresh
# [system, user] pair — always ends with a user message → Anthropic-safe.
# We embed the entire conversation context inside that user message.

async def call_agent(
    agent: ChatCompletionAgent,
    user_message: str,
    max_tokens: int = 300,
) -> str:
    """Call a single agent with a constructed user message and return its reply."""
    response = await agent.get_response(
        messages=user_message,
        arguments=base_args(max_tokens),
    )
    return str(response).strip()


# ---------------------------------------------------------------------------
# 1. Round-Robin — fixed turn order, shared context
# ---------------------------------------------------------------------------
# Each agent receives: topic + all previous contributions + its turn prompt.
# That message always ends with "Now respond as <name>." → user message → safe.
#
# LangGraph equivalent: cycle through nodes in a fixed order
# CrewAI equivalent:    Process.sequential with each agent handling one task

async def round_robin_example():
    print("\n--- 1. Round-Robin (fixed turn order) ---")

    kernel = make_kernel()

    optimist = ChatCompletionAgent(
        kernel=kernel,
        name="Optimist",
        instructions=(
            "You highlight the positive side of topics. "
            "Keep responses to 2 sentences."
        ),
    )
    pessimist = ChatCompletionAgent(
        kernel=kernel,
        name="Pessimist",
        instructions=(
            "You raise one constructive concern or downside. "
            "Keep responses to 2 sentences."
        ),
    )
    moderator = ChatCompletionAgent(
        kernel=kernel,
        name="Moderator",
        instructions=(
            "You synthesise both sides into a balanced 1-sentence summary, "
            "then ask one short follow-up question."
        ),
    )

    agents = [optimist, pessimist, moderator]
    topic = "Is remote work better than office work for software engineers?"
    contributions: list[tuple[str, str]] = []   # (agent_name, message)

    print(f"Topic: {topic}\n")

    # Two full rounds
    for round_num in range(2):
        for agent in agents:
            # Build conversation context from all previous turns
            context = "\n".join(
                f"[{name}]: {msg}" for name, msg in contributions
            )
            user_msg = f"Topic: {topic}"
            if context:
                user_msg += f"\n\nConversation so far:\n{context}"
            user_msg += f"\n\nYou are {agent.name}. Respond now (round {round_num + 1})."

            reply = await call_agent(agent, user_msg, max_tokens=200)
            contributions.append((agent.name, reply))
            print(f"  [{agent.name}]: {reply}\n")


# ---------------------------------------------------------------------------
# 2. Writer + Critic — reflection loop
# ---------------------------------------------------------------------------
# Writer produces content. Critic reviews it.
# If the Critic approves, the loop ends. Otherwise the Writer revises.
# This is the "reflection" pattern used in many production agentic systems.
#
# LangGraph equivalent: generate node → evaluate node → conditional edge back
# CrewAI equivalent:    two Tasks with context= wiring and human_input=True

async def writer_critic_example():
    print("\n--- 2. Writer + Critic (reflection loop) ---")

    kernel = make_kernel()

    writer = ChatCompletionAgent(
        kernel=kernel,
        name="Writer",
        instructions=(
            "You are a technical writer. Write a clear 3-sentence explanation "
            "of the concept requested. If feedback is given, revise accordingly."
        ),
    )
    critic = ChatCompletionAgent(
        kernel=kernel,
        name="Critic",
        instructions=(
            "You are a technical editor. If the writing is clear and accurate, "
            "respond ONLY with: APPROVED\n"
            "Otherwise give ONE specific improvement (1 sentence max)."
        ),
    )

    task = "Explain what a Semantic Kernel filter is and why it is useful."
    draft = ""
    feedback = ""
    max_rounds = 4

    print(f"Task: {task}\n")

    for round_num in range(max_rounds):
        # --- Writer's turn ---
        writer_prompt = f"Write a 3-sentence explanation of: {task}"
        if feedback:
            writer_prompt += f"\n\nPrevious draft:\n{draft}\n\nCritic's feedback: {feedback}\nPlease revise."
        draft = await call_agent(writer, writer_prompt, max_tokens=200)
        print(f"  [Writer] (round {round_num + 1}): {draft}\n")

        # --- Critic's turn ---
        critic_prompt = (
            f"Review this technical explanation:\n\n{draft}\n\n"
            "If it is clear and accurate, reply ONLY with: APPROVED\n"
            "Otherwise give ONE improvement suggestion."
        )
        feedback = await call_agent(critic, critic_prompt, max_tokens=100)
        print(f"  [Critic] (round {round_num + 1}): {feedback}\n")

        if "APPROVED" in feedback.upper():
            print("  [Loop] Critic approved — stopping.\n")
            break
    else:
        print("  [Loop] Max rounds reached.\n")

    print(f"  Final approved draft:\n  {draft}")


# ---------------------------------------------------------------------------
# 3. LLM-Driven Routing — selector agent picks who speaks next
# ---------------------------------------------------------------------------
# A lightweight "Selector" agent reads the conversation and names the
# next participant. This is the SK-with-Anthropic equivalent of:
#   LangGraph: conditional edge with a routing function
#   CrewAI:    Process.hierarchical with a manager_llm

async def llm_routing_example():
    print("\n--- 3. LLM-Driven Routing (selector picks next agent) ---")

    kernel = make_kernel()

    researcher = ChatCompletionAgent(
        kernel=kernel,
        name="Researcher",
        instructions="Provide a concise 2-sentence factual answer about AI frameworks.",
    )
    explainer = ChatCompletionAgent(
        kernel=kernel,
        name="Explainer",
        instructions="Restate the Researcher's facts in plain language using a simple analogy.",
    )
    summariser = ChatCompletionAgent(
        kernel=kernel,
        name="Summariser",
        instructions="Condense the Explainer's message into one bullet-point takeaway.",
    )

    selector = ChatCompletionAgent(
        kernel=kernel,
        name="Selector",
        instructions=(
            "You coordinate a 3-agent pipeline: Researcher → Explainer → Summariser.\n"
            "Based on who has spoken last, name who should speak next.\n"
            "Rules: if Researcher spoke last → Explainer; "
            "if Explainer spoke last → Summariser; "
            "if no one spoke → Researcher.\n"
            "Reply with ONLY the agent name."
        ),
    )

    agent_map = {
        "Researcher": researcher,
        "Explainer": explainer,
        "Summariser": summariser,
    }

    question = "What is the difference between SK plugins and LangChain tools?"
    contributions: list[tuple[str, str]] = []

    print(f"Question: {question}\n")

    for step in range(3):
        # Ask the selector who should speak next
        context = "\n".join(f"{n}: {m}" for n, m in contributions)
        selector_prompt = (
            f"Conversation so far:\n{context if context else '(none yet)'}\n\n"
            "Who should speak next? Reply with only: Researcher, Explainer, or Summariser."
        )
        chosen_name = await call_agent(selector, selector_prompt, max_tokens=10)
        # Clean up — model may add punctuation
        chosen_name = chosen_name.strip().strip(".").split()[0]

        agent = agent_map.get(chosen_name)
        if agent is None:
            print(f"  [Selector] Unknown agent '{chosen_name}', defaulting to Researcher.")
            agent = researcher
            chosen_name = "Researcher"

        print(f"  [Selector] → {chosen_name}")

        # Call the chosen agent
        context_str = "\n".join(f"[{n}]: {m}" for n, m in contributions)
        agent_prompt = f"Question: {question}"
        if context_str:
            agent_prompt += f"\n\nContext:\n{context_str}"
        agent_prompt += f"\n\nYou are {chosen_name}. Respond now."

        reply = await call_agent(agent, agent_prompt, max_tokens=200)
        contributions.append((chosen_name, reply))
        print(f"  [{chosen_name}]: {reply}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await round_robin_example()
    await writer_critic_example()
    await llm_routing_example()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • AgentGroupChat + Anthropic: incompatible — Anthropic forbids")
    print("    conversations ending with an assistant message (no prefill)")
    print("  • Anthropic-safe pattern: inject conversation context into a")
    print("    fresh user message for every agent turn")
    print("  • Round-robin: fixed order, each agent sees prior contributions")
    print("  • Writer + Critic: reflection loop — revise until approved")
    print("  • LLM routing: a Selector agent names who speaks next")
    print("  • All three patterns work by constructing the right user message")
    print("  • Next: Lesson 10 — Human-in-the-Loop (HITL)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
