"""
Lesson 10 — Human-in-the-Loop (HITL)
======================================
Concepts covered:
  - before_tool_callback as an approval gate (intercept → ask human → allow/block)
  - FunctionTool(require_confirmation=...) — ADK's native confirmation mechanism
  - Risk-based routing: low-risk tools auto-approved, high-risk tools need sign-off
  - Simulating approvals in tests (no interactive input needed)

Key insight:
  ADK doesn't have a built-in pause/resume for interactive input like LangGraph's
  interrupt. Instead, HITL is implemented via before_tool_callback:
    - Return None          → tool runs (approved)
    - Return {"error":...} → tool blocked (rejected), agent sees the error and responds

  ADK also has a native FunctionTool(require_confirmation=True) for production UIs
  that support the request/response confirmation protocol. We show both patterns.

Two demos:
  1. Approval gate via before_tool_callback (practical, scriptable)
  2. Risk-based routing — auto-approve safe tools, gate dangerous tools
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()

# ── Helper ─────────────────────────────────────────────────────────────────────

async def run_turn(runner: Runner, session_id: str, message: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    reply = ""
    async for event in runner.run_async(
        user_id="learner", session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    reply = part.text
    return reply


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 1 — Approval gate via before_tool_callback
#
# Scenario: a database management agent that can read (safe) or delete (dangerous)
# We intercept the delete operation and ask the human to approve/reject.
# ══════════════════════════════════════════════════════════════════════════════

def query_database(table: str, filter: str = "") -> dict:
    """Queries a database table and returns matching records.

    Args:
        table: The table name to query (e.g. 'users', 'orders').
        filter: Optional filter condition (e.g. 'status=active').
    """
    # Simulated database
    data = {
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "orders": [{"id": 101, "amount": 250}, {"id": 102, "amount": 80}],
    }
    records = data.get(table, [])
    return {"table": table, "records": records, "count": len(records)}


def delete_records(table: str, filter: str) -> dict:
    """PERMANENTLY deletes records from a database table. This action is irreversible.

    Args:
        table: The table name to delete from (e.g. 'users', 'orders').
        filter: The filter condition for records to delete (e.g. 'status=inactive').
    """
    # In a real system this would execute DELETE FROM table WHERE filter
    return {"deleted_from": table, "filter": filter, "rows_deleted": 3, "status": "success"}


# ── Approval callback ──────────────────────────────────────────────────────────
#
# In a real application, this would send a notification (Slack, email, etc.)
# and wait asynchronously for a response. In this lesson we simulate with
# a configurable auto-response so the script can run non-interactively.

# Set to True to simulate user approving, False to simulate rejection
SIMULATED_APPROVAL = True


def approval_gate(
    tool: BaseTool, args: dict, tool_context: ToolContext
) -> dict | None:
    """Intercepts dangerous tool calls and requires human approval."""
    DANGEROUS_TOOLS = {"delete_records"}

    if tool.name not in DANGEROUS_TOOLS:
        print(f"  [HITL] '{tool.name}' is safe — auto-approved")
        return None  # proceed without asking

    # Present the pending action to the human
    print(f"\n  ┌─ APPROVAL REQUIRED ──────────────────────────────────")
    print(f"  │ Tool  : {tool.name}")
    print(f"  │ Args  : {args}")
    print(f"  │ Risk  : IRREVERSIBLE — this cannot be undone")
    print(f"  └──────────────────────────────────────────────────────")

    # In production: send Slack/email and await async response
    # In this demo: use simulated approval flag
    if SIMULATED_APPROVAL:
        print(f"  [HITL] Simulating: APPROVED by human operator\n")
        return None  # allow tool to run
    else:
        print(f"  [HITL] Simulating: REJECTED by human operator\n")
        return {
            "error": f"Human operator rejected the call to {tool.name}({args}). "
                     f"Action was NOT executed."
        }


async def demo_approval_gate():
    print("── Demo 1: Approval gate via before_tool_callback ──\n")

    agent = Agent(
        name="db_agent",
        model="gemini-2.5-flash",
        instruction=(
            "You are a database assistant. Use query_database to read data "
            "and delete_records to remove data when asked. "
            "Always inform the user of results."
        ),
        tools=[query_database, delete_records],
        before_tool_callback=approval_gate,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="10_demo1", session_service=session_service)
    session = await session_service.create_session(app_name="10_demo1", user_id="learner")

    # Safe operation — auto-approved
    print("Turn 1: Safe read operation")
    reply = await run_turn(runner, session.id, "Show me all users in the database.")
    print(f"Agent: {reply}\n")

    # Dangerous operation — requires approval
    print("Turn 2: Dangerous delete operation")
    reply = await run_turn(
        runner, session.id,
        "Delete all inactive users from the users table (filter: status=inactive)."
    )
    print(f"Agent: {reply}\n")

    # Now simulate rejection
    global SIMULATED_APPROVAL
    SIMULATED_APPROVAL = False
    print("Turn 3: Dangerous operation — this time REJECTED")
    session2 = await session_service.create_session(app_name="10_demo1", user_id="learner")
    reply = await run_turn(
        runner, session2.id,
        "Delete all orders from the orders table (filter: amount<100)."
    )
    print(f"Agent: {reply}\n")
    SIMULATED_APPROVAL = True  # reset


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 2 — Risk-based routing with FunctionTool(require_confirmation)
#
# ADK's native require_confirmation=True works as follows:
#   1. Tool is called → FunctionTool checks require_confirmation
#   2. If True → tool returns immediately with "awaiting confirmation" response
#      and sets tool_context.request_confirmation()
#   3. The caller (runner) must handle this by sending a FunctionResponse
#      with confirmed=True/False
#
# For interactive UIs (web app, Slack bot), this is the right approach.
# For learning, we show the simpler callback approach and explain the difference.
#
# Here we demonstrate risk scoring: before_tool_callback inspects the args
# to decide dynamically whether approval is needed.
# ══════════════════════════════════════════════════════════════════════════════

def send_email(recipient: str, subject: str, body: str) -> dict:
    """Sends an email to a recipient.

    Args:
        recipient: Email address of the recipient.
        subject: Subject line of the email.
        body: Body content of the email.
    """
    print(f"  [email service] Sending to {recipient}: '{subject}'")
    return {"sent": True, "recipient": recipient, "subject": subject}


def bulk_send_email(recipient_list: str, subject: str, body: str) -> dict:
    """Sends an email to a large list of recipients (bulk operation).

    Args:
        recipient_list: Comma-separated list of email addresses.
        subject: Subject line of the email.
        body: Body content of the email.
    """
    recipients = [r.strip() for r in recipient_list.split(",")]
    print(f"  [email service] Bulk sending to {len(recipients)} recipients")
    return {"sent": True, "recipient_count": len(recipients), "subject": subject}


BULK_APPROVED = True  # simulate approval for bulk send


def risk_based_gate(
    tool: BaseTool, args: dict, tool_context: ToolContext
) -> dict | None:
    """Auto-approves single emails; gates bulk sends based on recipient count."""

    if tool.name == "send_email":
        print(f"  [HITL] Single email to '{args.get('recipient')}' — auto-approved")
        return None

    if tool.name == "bulk_send_email":
        recipients = [r.strip() for r in args.get("recipient_list", "").split(",")]
        count = len(recipients)
        print(f"\n  ┌─ BULK EMAIL APPROVAL ─────────────────────────────")
        print(f"  │ Recipients : {count}")
        print(f"  │ Subject    : {args.get('subject')}")
        print(f"  └───────────────────────────────────────────────────")

        if BULK_APPROVED:
            print(f"  [HITL] APPROVED — bulk send to {count} recipients\n")
            return None
        else:
            print(f"  [HITL] REJECTED — bulk send blocked\n")
            return {"error": f"Bulk send to {count} recipients was rejected by operator."}

    return None


async def demo_risk_based():
    print("── Demo 2: Risk-based routing ──\n")

    agent = Agent(
        name="email_agent",
        model="gemini-2.5-flash",
        instruction=(
            "You are an email assistant. Use send_email for individual recipients "
            "and bulk_send_email when sending to multiple people."
        ),
        tools=[send_email, bulk_send_email],
        before_tool_callback=risk_based_gate,
    )
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="10_demo2", session_service=session_service)
    session = await session_service.create_session(app_name="10_demo2", user_id="learner")

    print("Turn 1: Single email (auto-approved)")
    reply = await run_turn(
        runner, session.id,
        "Send an email to alice@example.com with subject 'Meeting tomorrow' "
        "and body 'Don't forget our 10am meeting.'"
    )
    print(f"Agent: {reply}\n")

    print("Turn 2: Bulk email (gated — simulated approval)")
    reply = await run_turn(
        runner, session.id,
        "Send a newsletter with subject 'Monthly Update' to "
        "bob@example.com, carol@example.com, dave@example.com, eve@example.com. "
        "Body: 'Check out our latest updates!'"
    )
    print(f"Agent: {reply}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=== Lesson 10 — Human-in-the-Loop ===\n")
    await demo_approval_gate()
    await demo_risk_based()


if __name__ == "__main__":
    asyncio.run(main())
