"""
Lesson 03 — Handoffs
====================
Covers:
  - handoff() — the primary multi-agent pattern in OpenAI Agents SDK
  - Triage agent routing to specialist agents
  - on_handoff callback — know when a handoff happens
  - Handoff with input filter — pass structured data to the receiving agent

Key concept:
  Handoffs are how agents delegate work to other agents.
  When an agent decides a task is better handled by a specialist,
  it emits a handoff — the SDK transfers control to the target agent
  with the full conversation history intact.

  This is THE signature pattern of the OpenAI Agents SDK.
  Unlike CrewAI (tasks → crews) or LangGraph (nodes → graph),
  OpenAI Agents routes via natural language decisions + handoff tools.
"""

import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Simple triage — one orchestrator, two specialists
# ---------------------------------------------------------------------------

billing_agent = Agent(
    name="BillingAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a billing specialist. Handle questions about invoices, "
        "payments, refunds, and subscription plans. Be concise and helpful."
    ),
    model="gpt-4o-mini",
)

tech_agent = Agent(
    name="TechSupportAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a technical support specialist. Handle questions about "
        "bugs, errors, integrations, and how the product works. Be precise."
    ),
    model="gpt-4o-mini",
)

triage_agent = Agent(
    name="TriageAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a customer service triage agent. Your ONLY job is to route "
        "the customer to the right specialist — do not answer questions yourself.\n"
        "- Billing questions (invoices, payments, refunds) → BillingAgent\n"
        "- Technical questions (bugs, errors, how-to) → TechSupportAgent"
    ),
    model="gpt-4o-mini",
    handoffs=[billing_agent, tech_agent],   # agents become handoff targets automatically
)


async def simple_triage_demo():
    print("=" * 50)
    print("PART 1: Simple triage handoffs")
    print("=" * 50)

    questions = [
        "I was charged twice for my subscription this month. Can you help?",
        "I'm getting a 403 error when I call your API with a valid token.",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        result = await Runner.run(triage_agent, input=q)
        print(f"Final answer ({result.last_agent.name}): {result.final_output}")


# ---------------------------------------------------------------------------
# 2. on_handoff callback — observe routing decisions
# ---------------------------------------------------------------------------

def on_handoff_to_billing(ctx: RunContextWrapper[None]):
    print("  [handoff] → BillingAgent triggered")

def on_handoff_to_tech(ctx: RunContextWrapper[None]):
    print("  [handoff] → TechSupportAgent triggered")


billing_agent_v2 = Agent(
    name="BillingAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a billing specialist. Handle questions about invoices, "
        "payments, and refunds. Be concise."
    ),
    model="gpt-4o-mini",
)

tech_agent_v2 = Agent(
    name="TechSupportAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a technical support specialist. Handle bugs and errors. Be precise."
    ),
    model="gpt-4o-mini",
)

triage_agent_v2 = Agent(
    name="TriageAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a customer service triage agent. Route to the right specialist.\n"
        "- Billing questions → BillingAgent\n"
        "- Technical questions → TechSupportAgent"
    ),
    model="gpt-4o-mini",
    handoffs=[
        handoff(billing_agent_v2, on_handoff=on_handoff_to_billing),
        handoff(tech_agent_v2,    on_handoff=on_handoff_to_tech),
    ],
)


async def handoff_callback_demo():
    print("\n" + "=" * 50)
    print("PART 2: on_handoff callback — observe routing")
    print("=" * 50)

    queries = [
        "How do I cancel my plan and get a refund?",
        "Your SDK throws a KeyError on RunResult.final_output sometimes.",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        result = await Runner.run(triage_agent_v2, input=q)
        print(f"Final answer ({result.last_agent.name}): {result.final_output}")


# ---------------------------------------------------------------------------
# 3. result.last_agent — know which agent actually answered
# ---------------------------------------------------------------------------

async def last_agent_demo():
    print("\n" + "=" * 50)
    print("PART 3: Inspecting the run — last_agent, agent history")
    print("=" * 50)

    questions = [
        "My invoice shows the wrong amount.",
        "What does error code 429 mean in your API?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        result = await Runner.run(triage_agent, input=q)

        print(f"  Started with : {triage_agent.name}")
        print(f"  Answered by  : {result.last_agent.name}")
        print(f"  Answer       : {result.final_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await simple_triage_demo()
    await handoff_callback_demo()
    await last_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
