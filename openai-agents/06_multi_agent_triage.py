"""
Lesson 06 — Multi-Agent Triage
================================
Covers:
  - Orchestrator + multiple specialist agents (real-world scale)
  - Chained handoffs — specialist handing off to another specialist
  - Specialists with their own tools
  - Context flowing through a multi-hop handoff chain
  - result.last_agent to trace where the run ended up

Key concept:
  Lesson 03 showed basic triage with 2 agents.
  This lesson builds a realistic support system:
    - One orchestrator triages by topic
    - Each specialist has domain-specific tools
    - Some paths chain through two handoffs (escalation)
    - Shared context (ticket) is mutated along the way

  The pattern: thin orchestrator, fat specialists.
  The orchestrator only routes — specialists do the real work.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, RunContextWrapper, function_tool, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()


# ---------------------------------------------------------------------------
# Shared context — flows through all agents in the chain
# ---------------------------------------------------------------------------

@dataclass
class SupportContext:
    ticket_id: str
    customer: str
    plan: Literal["free", "pro", "enterprise"]
    history: list[str] = field(default_factory=list)   # audit trail

    def log(self, agent: str, action: str):
        self.history.append(f"[{agent}] {action}")


# ---------------------------------------------------------------------------
# Specialist tools
# ---------------------------------------------------------------------------

@function_tool
def lookup_invoice(ctx: RunContextWrapper[SupportContext], invoice_id: str) -> str:
    """Look up an invoice by ID and return its details.

    Args:
        invoice_id: The invoice identifier (e.g. INV-001).
    """
    ctx.context.log("BillingAgent", f"looked up invoice {invoice_id}")
    # Mock data
    invoices = {
        "INV-001": {"amount": "$99", "date": "2026-02-01", "status": "paid"},
        "INV-002": {"amount": "$99", "date": "2026-03-01", "status": "overdue"},
    }
    inv = invoices.get(invoice_id)
    if not inv:
        return f"Invoice {invoice_id} not found."
    return f"Invoice {invoice_id}: {inv['amount']} on {inv['date']} — {inv['status']}"


@function_tool
def issue_refund(ctx: RunContextWrapper[SupportContext], invoice_id: str, reason: str) -> str:
    """Issue a refund for an invoice.

    Args:
        invoice_id: The invoice to refund.
        reason: Short reason for the refund.
    """
    ctx.context.log("BillingAgent", f"issued refund for {invoice_id}: {reason}")
    return f"Refund issued for {invoice_id}. Reason: {reason}. Credits appear in 3–5 days."


@function_tool
def check_api_status(ctx: RunContextWrapper[SupportContext], endpoint: str) -> str:
    """Check the status of an API endpoint.

    Args:
        endpoint: The API endpoint path (e.g. /v1/agents/run).
    """
    ctx.context.log("TechAgent", f"checked API status for {endpoint}")
    statuses = {
        "/v1/agents/run": "operational",
        "/v1/agents/stream": "degraded — elevated latency",
        "/v1/files": "operational",
    }
    status = statuses.get(endpoint, "unknown endpoint")
    return f"Status of {endpoint}: {status}"


@function_tool
def create_bug_report(ctx: RunContextWrapper[SupportContext], summary: str, severity: str) -> str:
    """Create a bug report for an engineering issue.

    Args:
        summary: Short description of the bug.
        severity: One of: low, medium, high, critical.
    """
    ctx.context.log("TechAgent", f"created bug report: [{severity}] {summary}")
    report_id = f"BUG-{abs(hash(summary)) % 9000 + 1000}"
    return f"Bug report {report_id} created. Severity: {severity}. Engineering notified."


@function_tool
def check_account_status(ctx: RunContextWrapper[SupportContext]) -> str:
    """Check the current customer's account and plan status."""
    c = ctx.context
    ctx.context.log("AccountAgent", "checked account status")
    return (
        f"Account: {c.customer} | Plan: {c.plan} | Ticket: {c.ticket_id} | Status: active"
    )


@function_tool
def upgrade_plan(ctx: RunContextWrapper[SupportContext], new_plan: str) -> str:
    """Upgrade the customer's subscription plan.

    Args:
        new_plan: Target plan name (e.g. 'pro', 'enterprise').
    """
    old = ctx.context.plan
    ctx.context.plan = new_plan  # type: ignore
    ctx.context.log("AccountAgent", f"upgraded plan: {old} → {new_plan}")
    return f"Plan upgraded from {old} to {new_plan}. Changes effective immediately."


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

billing_agent = Agent(
    name="BillingAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a billing specialist. Handle invoices, payments, and refunds. "
        "Use your tools to look up invoices and issue refunds. Be concise and helpful."
    ),
    model="gpt-4o-mini",
    tools=[lookup_invoice, issue_refund],
)

tech_agent = Agent(
    name="TechSupportAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a technical support specialist. Handle API issues, bugs, and errors. "
        "Use tools to check API status and file bug reports. Be precise."
    ),
    model="gpt-4o-mini",
    tools=[check_api_status, create_bug_report],
)

account_agent = Agent(
    name="AccountAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are an account specialist. Handle plan upgrades, account status, "
        "and account-level questions. Use tools to check and update accounts."
    ),
    model="gpt-4o-mini",
    tools=[check_account_status, upgrade_plan],
)

# ---------------------------------------------------------------------------
# Escalation agent — receives handoffs from specialists for complex cases
# ---------------------------------------------------------------------------

escalation_agent = Agent(
    name="EscalationAgent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a senior support specialist handling escalated cases. "
        "The customer has a complex or unresolved issue. "
        "Acknowledge the escalation, summarise what's known, and provide "
        "a clear resolution path or promise a follow-up within 24 hours."
    ),
    model="gpt-4o-mini",
)

# Give specialists the ability to escalate
billing_agent.handoffs = [escalation_agent]
tech_agent.handoffs    = [escalation_agent]
account_agent.handoffs = [escalation_agent]


# ---------------------------------------------------------------------------
# Orchestrator — thin router only
# ---------------------------------------------------------------------------

def on_billing(ctx: RunContextWrapper[SupportContext]):
    ctx.context.log("Orchestrator", "routed → BillingAgent")

def on_tech(ctx: RunContextWrapper[SupportContext]):
    ctx.context.log("Orchestrator", "routed → TechSupportAgent")

def on_account(ctx: RunContextWrapper[SupportContext]):
    ctx.context.log("Orchestrator", "routed → AccountAgent")


orchestrator = Agent(
    name="SupportOrchestrator",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a customer support triage agent. Route to the right specialist — "
        "do NOT answer questions yourself.\n"
        "- Invoices, payments, refunds → BillingAgent\n"
        "- API errors, bugs, technical issues → TechSupportAgent\n"
        "- Plan upgrades, account status → AccountAgent"
    ),
    model="gpt-4o-mini",
    handoffs=[
        handoff(billing_agent,  on_handoff=on_billing),
        handoff(tech_agent,     on_handoff=on_tech),
        handoff(account_agent,  on_handoff=on_account),
    ],
)


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------

async def run_scenario(label: str, message: str, ctx: SupportContext):
    print(f"\n{'─' * 50}")
    print(f"Scenario : {label}")
    print(f"Customer : {ctx.customer} ({ctx.plan} plan)")
    print(f"Message  : {message}")
    result = await Runner.run(orchestrator, input=message, context=ctx)
    print(f"Handled by: {result.last_agent.name}")
    print(f"Response  : {result.final_output}")
    print(f"Audit trail:")
    for entry in ctx.history:
        print(f"  {entry}")


async def main():
    print("=" * 50)
    print("Multi-Agent Triage — 4 scenarios")
    print("=" * 50)

    await run_scenario(
        label="Billing — refund request",
        message="I need a refund for invoice INV-001. I was charged but never used the service.",
        ctx=SupportContext(ticket_id="T-101", customer="Srini", plan="pro"),
    )

    await run_scenario(
        label="Tech — API issue",
        message="I'm seeing high latency on /v1/agents/stream. Is there an outage?",
        ctx=SupportContext(ticket_id="T-102", customer="Priya", plan="enterprise"),
    )

    await run_scenario(
        label="Account — plan upgrade",
        message="I want to upgrade my account from free to pro.",
        ctx=SupportContext(ticket_id="T-103", customer="Alex", plan="free"),
    )

    await run_scenario(
        label="Tech — bug report + escalation",
        message=(
            "Your API returns a 500 error on /v1/agents/run about 30% of the time. "
            "This is a critical production issue. We've already lost revenue. "
            "I need this escalated to engineering immediately."
        ),
        ctx=SupportContext(ticket_id="T-104", customer="Jordan", plan="enterprise"),
    )


if __name__ == "__main__":
    asyncio.run(main())
