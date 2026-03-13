"""
Lesson 12 — Capstone: All Patterns Combined
=============================================
Scenario: AI-powered Customer Support Platform

Every pattern from lessons 01–11 appears here:
  01 - Agent + Runner.run() + to_input_list()
  02 - @function_tool with RunContextWrapper
  03 - handoff() — triage orchestrator → specialists
  04 - RunContextWrapper[SupportSession] — shared context
  05 - output_type=PydanticModel — structured resolution report
  06 - Multi-agent triage with escalation
  07 - @input_guardrail (abuse filter) + @output_guardrail (tone check)
  08 - Runner.run_streamed() + stream_events()
  09 - RunHooks — audit trail and metrics
  10 - needs_approval=True — HITL for destructive tools
  11 - MCPServerStdio — knowledge base + ticket creation via MCP

Architecture:
  User message
      ↓ [input guardrail: abuse filter]
  OrchestratorAgent
      ↓ [handoff]
  BillingAgent / TechAgent / AccountAgent
      ↓ [tools: some with needs_approval]
      ↓ [MCP: search_kb, create_ticket, get_customer_history]
      ↓ [output guardrail: tone check]
  ResolutionReport (structured output)
      ↓ [RunHooks: audit trail written to SupportSession]
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from agents import (
    Agent, Runner, RunContextWrapper, RunHooks,
    function_tool, handoff,
    input_guardrail, output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents.exceptions import MaxTurnsExceeded
from agents.stream_events import RawResponsesStreamEvent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()

MCP_SERVER = str(Path(__file__).parent / "12_capstone_mcp.py")
PYTHON = sys.executable


# ===========================================================================
# SHARED CONTEXT  (Lesson 04)
# ===========================================================================

@dataclass
class SupportSession:
    customer_id: str
    customer_name: str
    tier: Literal["free", "pro", "enterprise"]
    audit: list[str] = field(default_factory=list)
    tool_calls: int = 0
    handoffs_made: int = 0

    def log(self, msg: str):
        self.audit.append(msg)


# ===========================================================================
# RUN HOOKS — audit trail + metrics  (Lesson 09)
# ===========================================================================

class AuditHooks(RunHooks):
    async def on_agent_start(self, context: RunContextWrapper[SupportSession], agent: Agent) -> None:
        context.context.log(f"agent_start: {agent.name}")

    async def on_tool_start(self, context: RunContextWrapper[SupportSession], agent: Agent, tool) -> None:
        context.context.tool_calls += 1
        context.context.log(f"tool_call: {agent.name}.{tool.name}")

    async def on_handoff(self, context: RunContextWrapper[SupportSession], from_agent: Agent, to_agent: Agent) -> None:
        context.context.handoffs_made += 1
        context.context.log(f"handoff: {from_agent.name} → {to_agent.name}")


# ===========================================================================
# STRUCTURED OUTPUT — resolution report  (Lesson 05)
# ===========================================================================

class ResolutionReport(BaseModel):
    ticket_id: str | None = Field(default=None, description="MCP ticket ID if one was created")
    category: Literal["billing", "technical", "account", "general"]
    resolution: str = Field(description="What was done to resolve the issue")
    follow_up_required: bool
    follow_up_notes: str | None = None


# ===========================================================================
# GUARDRAILS  (Lesson 07)
# ===========================================================================

class AbuseCheck(BaseModel):
    is_abusive: bool
    reason: str

class ToneCheck(BaseModel):
    is_unprofessional: bool
    reason: str


abuse_classifier = Agent(
    name="AbuseClassifier",
    instructions=(
        "Classify if the message contains abusive language, threats, or spam. "
        "Normal frustration about a service issue is NOT abusive."
    ),
    model="gpt-4o-mini",
    output_type=AbuseCheck,
)

tone_classifier = Agent(
    name="ToneClassifier",
    instructions=(
        "Check if the support response is professional. "
        "Flag if it contains slang, dismissive language, or is under 20 words for a real issue."
    ),
    model="gpt-4o-mini",
    output_type=ToneCheck,
)


@input_guardrail
async def abuse_guardrail(ctx: RunContextWrapper[SupportSession], agent: Agent, input: str | list) -> GuardrailFunctionOutput:
    text = input if isinstance(input, str) else str(input)
    result = await Runner.run(abuse_classifier, input=text, context=ctx.context)
    check: AbuseCheck = result.final_output
    if check.is_abusive:
        ctx.context.log(f"guardrail_blocked: abusive input")
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=check.is_abusive)


@output_guardrail
async def tone_guardrail(ctx: RunContextWrapper[SupportSession], agent: Agent, output) -> GuardrailFunctionOutput:
    # output may be a Pydantic model when agent has output_type set
    text = output if isinstance(output, str) else output.model_dump_json()
    result = await Runner.run(tone_classifier, input=text, context=ctx.context)
    check: ToneCheck = result.final_output
    if check.is_unprofessional:
        ctx.context.log(f"guardrail_blocked: unprofessional output")
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=check.is_unprofessional)


# ===========================================================================
# TOOLS  (Lesson 02 + 10)
# ===========================================================================

@function_tool
def lookup_invoice(ctx: RunContextWrapper[SupportSession], invoice_id: str) -> str:
    """Look up an invoice by ID.

    Args:
        invoice_id: Invoice identifier (e.g. INV-001).
    """
    ctx.context.log(f"tool: lookup_invoice({invoice_id})")
    mock = {
        "INV-001": {"amount": "$99", "date": "2026-02-01", "status": "paid"},
        "INV-002": {"amount": "$99", "date": "2026-03-01", "status": "overdue"},
    }
    inv = mock.get(invoice_id)
    return f"{invoice_id}: {inv}" if inv else f"Invoice {invoice_id} not found."


@function_tool(needs_approval=True)
def issue_refund(ctx: RunContextWrapper[SupportSession], invoice_id: str, reason: str) -> str:
    """Issue a refund for an invoice. Requires human approval.

    Args:
        invoice_id: Invoice to refund.
        reason: Reason for the refund.
    """
    ctx.context.log(f"tool: issue_refund({invoice_id})")
    return f"Refund issued for {invoice_id}. Reason: {reason}. Credit appears in 3–5 days."


@function_tool
def get_account_info(ctx: RunContextWrapper[SupportSession]) -> str:
    """Get the current customer's account information."""
    s = ctx.context
    return f"Account: {s.customer_name} | ID: {s.customer_id} | Tier: {s.tier}"


@function_tool(needs_approval=True)
def downgrade_account(ctx: RunContextWrapper[SupportSession], reason: str) -> str:
    """Downgrade the customer's account to free tier. Requires human approval.

    Args:
        reason: Reason for the downgrade.
    """
    old_tier = ctx.context.tier
    ctx.context.tier = "free"
    ctx.context.log(f"tool: downgrade_account({old_tier}→free)")
    return f"Account downgraded from {old_tier} to free. Reason: {reason}"


# ===========================================================================
# AGENTS  (Lessons 03, 06)
# ===========================================================================

def build_agents(mcp_server: MCPServerStdio) -> Agent:
    """Build the full agent network. Returns the orchestrator."""

    # --- Escalation agent (catch-all for complex cases) ---
    escalation_agent = Agent(
        name="EscalationAgent",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are a senior support specialist handling escalated issues. "
            "Acknowledge the complexity, summarise what's known, and provide a clear "
            "resolution path or commit to a 24-hour follow-up. "
            "Always create a ticket via MCP for escalated cases."
        ),
        model="gpt-4o-mini",
        mcp_servers=[mcp_server],
        output_type=ResolutionReport,
        output_guardrails=[tone_guardrail],
    )

    # --- Billing specialist ---
    billing_agent = Agent(
        name="BillingAgent",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are a billing specialist. Handle invoice lookups, refund requests, "
            "and billing questions. Use MCP to check customer history and create tickets. "
            "Refunds require approval — ask the tool, don't skip it."
        ),
        model="gpt-4o-mini",
        tools=[lookup_invoice, issue_refund],
        mcp_servers=[mcp_server],
        handoffs=[escalation_agent],
        output_type=ResolutionReport,
        output_guardrails=[tone_guardrail],
    )

    # --- Tech specialist ---
    tech_agent = Agent(
        name="TechSupportAgent",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are a technical support specialist. Search the knowledge base for answers. "
            "Create a ticket if the issue needs engineering follow-up. "
            "Be precise and reference KB articles in your answer."
        ),
        model="gpt-4o-mini",
        mcp_servers=[mcp_server],
        handoffs=[escalation_agent],
        output_type=ResolutionReport,
        output_guardrails=[tone_guardrail],
    )

    # --- Account specialist ---
    account_agent = Agent(
        name="AccountAgent",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are an account specialist. Handle account info lookups, downgrades, "
            "and account-level questions. Downgrades require approval."
        ),
        model="gpt-4o-mini",
        tools=[get_account_info, downgrade_account],
        mcp_servers=[mcp_server],
        handoffs=[escalation_agent],
        output_type=ResolutionReport,
        output_guardrails=[tone_guardrail],
    )

    # --- Triage orchestrator ---
    orchestrator = Agent(
        name="SupportOrchestrator",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are a customer support triage agent. Route to the right specialist — "
            "do NOT answer questions yourself.\n"
            "- Billing, invoices, refunds → BillingAgent\n"
            "- API errors, technical issues → TechSupportAgent\n"
            "- Account info, plan changes → AccountAgent"
        ),
        model="gpt-4o-mini",
        input_guardrails=[abuse_guardrail],
        handoffs=[
            handoff(billing_agent,  on_handoff=lambda ctx: ctx.context.log("route→BillingAgent")),
            handoff(tech_agent,     on_handoff=lambda ctx: ctx.context.log("route→TechSupportAgent")),
            handoff(account_agent,  on_handoff=lambda ctx: ctx.context.log("route→AccountAgent")),
        ],
    )

    return orchestrator


# ===========================================================================
# HITL approval helper  (Lesson 10)
# ===========================================================================

def handle_interruptions(result, session: SupportSession) -> tuple:
    """Prompt for approval on each interrupted tool. Returns (state, any_rejected)."""
    state = result.to_state()
    any_rejected = False
    for item in result.interruptions:
        args = json.loads(item.raw_item.arguments)
        print(f"\n  *** APPROVAL REQUIRED ***")
        print(f"  Tool     : {item.raw_item.name}")
        print(f"  Arguments: {args}")
        answer = input("  Approve? (y/n): ").strip().lower()
        if answer == "y":
            state.approve(item)
            session.log(f"approved: {item.raw_item.name}")
        else:
            state.reject(item, always_reject=True)
            session.log(f"rejected: {item.raw_item.name}")
            any_rejected = True
    return state, any_rejected


# ===========================================================================
# SCENARIO A — Full pipeline with HITL + structured output  (no streaming)
# ===========================================================================

async def run_support_request(orchestrator: Agent, message: str, session: SupportSession):
    print(f"\n{'═' * 55}")
    print(f"Customer : {session.customer_name} ({session.tier}) [{session.customer_id}]")
    print(f"Message  : {message}")
    print(f"{'═' * 55}")

    try:
        result = await Runner.run(
            orchestrator,
            input=message,
            context=session,
            hooks=AuditHooks(),
            max_turns=20,
        )

        while result.interruptions:
            state, _ = handle_interruptions(result, session)
            result = await Runner.run(orchestrator, input=state, max_turns=20)

        report: ResolutionReport = result.final_output_as(ResolutionReport)
        print(f"\nHandled by : {result.last_agent.name}")
        print(f"Category   : {report.category}")
        print(f"Ticket     : {report.ticket_id or 'none'}")
        print(f"Resolution : {report.resolution}")
        print(f"Follow-up  : {'Yes — ' + report.follow_up_notes if report.follow_up_required else 'No'}")

    except InputGuardrailTripwireTriggered:
        print("[BLOCKED] Message flagged as abusive — not processed.")
    except OutputGuardrailTripwireTriggered:
        print("[BLOCKED] Response failed tone check — not delivered.")
    except MaxTurnsExceeded:
        print("[ENDED] Max turns reached after rejection.")


# ===========================================================================
# SCENARIO B — Streaming response  (Lesson 08)
# ===========================================================================

async def run_support_streaming(orchestrator: Agent, message: str, session: SupportSession):
    print(f"\n{'═' * 55}")
    print(f"[STREAMING] {session.customer_name}: {message}")
    print(f"{'═' * 55}")
    print("Response: ", end="", flush=True)

    result = Runner.run_streamed(
        orchestrator,
        input=message,
        context=session,
        hooks=AuditHooks(),
    )

    try:
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
    except InputGuardrailTripwireTriggered:
        print("\n[BLOCKED] Abusive message — not processed.")
        return

    print(f"\n[ended with: {result.current_agent.name}]")


# ===========================================================================
# MAIN
# ===========================================================================

async def main():
    async with MCPServerStdio(
        params=MCPServerStdioParams(command=PYTHON, args=[MCP_SERVER]),
        cache_tools_list=True,
        name="SupportKB",
    ) as mcp_server:

        orchestrator = build_agents(mcp_server)

        # --- Scenario A: normal support requests ---
        print("\n" + "=" * 55)
        print("SCENARIO A — Full pipeline (HITL + structured output)")
        print("=" * 55)

        scenarios = [
            ("C-001", "Srini",  "pro",        "I need a refund for invoice INV-002. I was charged but cancelled before the period."),
            ("C-002", "Priya",  "enterprise", "I'm hitting rate limits on the API even though I'm on the enterprise plan. What are my limits?"),
            ("C-003", "Alex",   "free",        "What's the refund policy?"),
            ("C-001", "Srini",  "pro",        "YOU IDIOTS RUINED MY BUSINESS!! I WANT MY MONEY BACK NOW!!"),  # blocked by guardrail
        ]

        for cid, name, tier, message in scenarios:
            session = SupportSession(customer_id=cid, customer_name=name, tier=tier)
            await run_support_request(orchestrator, message, session)
            print(f"\n[Audit] tool_calls={session.tool_calls} handoffs={session.handoffs_made}")
            print(f"[Trail] {' | '.join(session.audit)}")

        # --- Scenario B: streaming ---
        print("\n\n" + "=" * 55)
        print("SCENARIO B — Streaming response")
        print("=" * 55)

        streaming_session = SupportSession(customer_id="C-003", customer_name="Alex", tier="free")
        await run_support_streaming(
            orchestrator,
            "How do I enable MFA on my account?",
            streaming_session,
        )


if __name__ == "__main__":
    asyncio.run(main())
