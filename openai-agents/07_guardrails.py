"""
Lesson 07 — Guardrails
=======================
Covers:
  - @input_guardrail  — screen user input BEFORE the agent processes it
  - @output_guardrail — screen agent output BEFORE it reaches the user
  - GuardrailFunctionOutput(tripwire_triggered=True) — halt the run
  - InputGuardrailTripwireTriggered / OutputGuardrailTripwireTriggered exceptions
  - Combining multiple guardrails on one agent

Key concept:
  Guardrails are safety wrappers that run in PARALLEL with the agent.
  Input guardrails fire before the agent sees the message.
  Output guardrails fire after the agent responds but before you get the result.

  When tripwire_triggered=True, the SDK raises an exception immediately —
  the agent's response is discarded and you handle the exception.
  This lets you enforce topic restrictions, PII filters, compliance rules, etc.
"""

import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Input guardrail — block off-topic requests
# ---------------------------------------------------------------------------

class TopicCheckOutput(BaseModel):
    is_off_topic: bool
    reason: str


guardrail_agent = Agent(
    name="TopicGuardrail",
    instructions=(
        "You are a topic classifier for a Python coding assistant. "
        "Determine if the user's message is off-topic (not about Python or programming). "
        "Politics, personal advice, medical questions, or anything unrelated to coding is off-topic."
    ),
    model="gpt-4o-mini",
    output_type=TopicCheckOutput,
)


@input_guardrail
async def topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list
) -> GuardrailFunctionOutput:
    """Block any message not related to Python or programming."""
    text = input if isinstance(input, str) else str(input)
    result = await Runner.run(guardrail_agent, input=text, context=ctx.context)
    check: TopicCheckOutput = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=check.is_off_topic,
    )


coding_assistant = Agent(
    name="CodingAssistant",
    instructions="You are a helpful Python coding assistant. Answer questions about Python and programming.",
    model="gpt-4o-mini",
    input_guardrails=[topic_guardrail],
)


async def input_guardrail_demo():
    print("=" * 50)
    print("PART 1: Input guardrail — off-topic blocker")
    print("=" * 50)

    messages = [
        "How do I use list comprehensions in Python?",       # allowed
        "What's the best diet for losing weight quickly?",   # blocked
        "Explain Python decorators with an example.",        # allowed
        "Who should I vote for in the next election?",       # blocked
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        try:
            result = await Runner.run(coding_assistant, input=msg)
            print(f"Agent: {result.final_output}")
        except InputGuardrailTripwireTriggered as e:
            print(f"[BLOCKED] Input guardrail triggered — off-topic request rejected.")


# ---------------------------------------------------------------------------
# 2. Output guardrail — catch PII or sensitive content in responses
# ---------------------------------------------------------------------------

class PIICheckOutput(BaseModel):
    contains_pii: bool
    pii_types_found: list[str]


pii_checker_agent = Agent(
    name="PIIChecker",
    instructions=(
        "You are a PII detector. Check if the given text contains any Personally "
        "Identifiable Information such as: email addresses, phone numbers, SSNs, "
        "credit card numbers, or full names combined with addresses. "
        "List the types of PII found."
    ),
    model="gpt-4o-mini",
    output_type=PIICheckOutput,
)


@output_guardrail
async def pii_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, output: str
) -> GuardrailFunctionOutput:
    """Block any agent response that contains PII."""
    result = await Runner.run(pii_checker_agent, input=output, context=ctx.context)
    check: PIICheckOutput = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=check.contains_pii,
    )


# This agent is deliberately instructed to repeat user data — so we can trigger the guardrail
leaky_agent = Agent(
    name="DataAgent",
    instructions=(
        "You are a data assistant. When the user shares their information, "
        "confirm it back to them verbatim so they know it was received."
    ),
    model="gpt-4o-mini",
    output_guardrails=[pii_guardrail],
)


async def output_guardrail_demo():
    print("\n" + "=" * 50)
    print("PART 2: Output guardrail — PII filter")
    print("=" * 50)

    messages = [
        "Just confirming my order is ready.",                                          # clean
        "My email is john.doe@example.com and phone is 555-867-5309. Got it?",        # PII
        "The answer is 42.",                                                           # clean
        "Please confirm: my SSN is 123-45-6789 and card ending in 4242.",             # PII
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        try:
            result = await Runner.run(leaky_agent, input=msg)
            print(f"Agent: {result.final_output}")
        except OutputGuardrailTripwireTriggered as e:
            print(f"[BLOCKED] Output guardrail triggered — response contained PII, suppressed.")


# ---------------------------------------------------------------------------
# 3. Multiple guardrails — input + output on same agent
# ---------------------------------------------------------------------------

class LengthCheckOutput(BaseModel):
    is_too_long: bool
    word_count: int


length_checker = Agent(
    name="LengthChecker",
    instructions=(
        "Count the words in the given text. "
        "Flag it as too_long if it exceeds 60 words."
    ),
    model="gpt-4o-mini",
    output_type=LengthCheckOutput,
)


@input_guardrail
async def brevity_input_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list
) -> GuardrailFunctionOutput:
    """Block user messages longer than 60 words."""
    text = input if isinstance(input, str) else str(input)
    result = await Runner.run(length_checker, input=text, context=ctx.context)
    check: LengthCheckOutput = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=check.is_too_long,
    )


@output_guardrail
async def brevity_output_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, output: str
) -> GuardrailFunctionOutput:
    """Block agent responses longer than 60 words."""
    result = await Runner.run(length_checker, input=output, context=ctx.context)
    check: LengthCheckOutput = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=check.is_too_long,
    )


concise_agent = Agent(
    name="ConciseAgent",
    instructions=(
        "You are a concise assistant. Answer in 1–2 sentences only. "
        "Never write more than 60 words in a response."
    ),
    model="gpt-4o-mini",
    input_guardrails=[brevity_input_guardrail],
    output_guardrails=[brevity_output_guardrail],
)


async def combined_guardrails_demo():
    print("\n" + "=" * 50)
    print("PART 3: Combined input + output guardrails")
    print("=" * 50)

    messages = [
        "What is Python?",    # short input, short output — passes both
        # Long input — should trip input guardrail
        (
            "I have a very detailed and complex question about Python that requires "
            "a lot of context. I want to understand how decorators work, what metaclasses "
            "are, how the GIL affects threading, and also how asyncio event loops function "
            "under the hood. Can you explain all of this?"
        ),
    ]

    for msg in messages:
        display = msg[:60] + "..." if len(msg) > 60 else msg
        print(f"\nUser: {display}")
        try:
            result = await Runner.run(concise_agent, input=msg)
            print(f"Agent: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            print("[BLOCKED] Input too long — input guardrail triggered.")
        except OutputGuardrailTripwireTriggered:
            print("[BLOCKED] Response too long — output guardrail triggered.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await input_guardrail_demo()
    await output_guardrail_demo()
    await combined_guardrails_demo()


if __name__ == "__main__":
    asyncio.run(main())
