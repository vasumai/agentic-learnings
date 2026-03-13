"""
Lesson 10 — Human-in-the-Loop (HITL)
======================================
Covers:
  - needs_approval=True on @function_tool — pause before execution
  - needs_approval=async_fn — dynamic approval based on tool arguments
  - result.interruptions — list of ToolApprovalItem pending approval
  - result.to_state() → RunState — serializable pause point
  - state.approve() / state.reject() — human decision
  - Runner.run(agent, state) — resume after decision

Key concept:
  When a tool has needs_approval=True, the SDK does NOT execute it.
  Instead, Runner.run() returns early with result.interruptions populated.
  Your code inspects the pending calls, asks the human, then either:
    - state.approve(item) → resumes and executes the tool
    - state.reject(item)  → resumes but skips the tool (agent gets rejection message)

  This is the pattern for any agentic action that needs human oversight:
  file deletion, database writes, sending emails, financial transactions, etc.
"""

import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from agents import Agent, Runner, RunContextWrapper, function_tool, ToolApprovalItem
from agents.exceptions import MaxTurnsExceeded

load_dotenv()


# ---------------------------------------------------------------------------
# Helper — simulate a human approval prompt in the terminal
# ---------------------------------------------------------------------------

def ask_human(tool_name: str, tool_args: dict) -> bool:
    """Prompt the terminal user to approve or reject a tool call."""
    print(f"\n  *** APPROVAL REQUIRED ***")
    print(f"  Tool     : {tool_name}")
    print(f"  Arguments: {tool_args}")
    answer = input("  Approve? (y/n): ").strip().lower()
    return answer == "y"


# ---------------------------------------------------------------------------
# 1. Static approval — always requires human sign-off
# ---------------------------------------------------------------------------

@function_tool(needs_approval=True)
def delete_file(filepath: str) -> str:
    """Permanently delete a file from the filesystem.

    Args:
        filepath: The full path of the file to delete.
    """
    # Mock — in real life this would call os.remove()
    return f"File '{filepath}' deleted successfully."


@function_tool(needs_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
    """
    return f"Email sent to {to} with subject '{subject}'."


@function_tool
def list_files(directory: str) -> str:
    """List files in a directory.

    Args:
        directory: The directory path to list.
    """
    # Mock — no approval needed for read-only operations
    mock_files = {
        "/tmp": ["report.pdf", "old_data.csv", "backup.zip"],
        "/home": ["config.yaml", "notes.txt"],
    }
    files = mock_files.get(directory, ["file_a.txt", "file_b.log"])
    return f"Files in {directory}: {', '.join(files)}"


async def static_approval_demo():
    print("=" * 50)
    print("PART 1: Static approval — always requires sign-off")
    print("=" * 50)

    agent = Agent(
        name="FileAgent",
        instructions=(
            "You are a file management assistant. "
            "You can list files, delete files, and send email reports. "
            "Use the appropriate tool for each request."
        ),
        model="gpt-4o-mini",
        tools=[list_files, delete_file, send_email],
    )

    user_request = (
        "List files in /tmp, then delete old_data.csv, "
        "then email admin@example.com a summary with subject 'Cleanup Done'."
    )
    print(f"\nUser: {user_request}\n")

    # First run — will pause at the first tool needing approval
    result = await Runner.run(agent, input=user_request)

    # Loop: keep approving/rejecting until no more interruptions
    import json
    while result.interruptions:
        state = result.to_state()          # create state ONCE for this batch
        for item in result.interruptions:
            tool_name = item.raw_item.name
            try:
                args_dict = json.loads(item.raw_item.arguments)
            except Exception:
                args_dict = {"raw": item.raw_item.arguments}

            approved = ask_human(tool_name, args_dict)
            if approved:
                state.approve(item)
                print(f"  [approved] {tool_name}")
            else:
                state.reject(item, always_reject=True)
                print(f"  [rejected] {tool_name}")

        # Resume run with the updated state
        result = await Runner.run(agent, input=state)

    print(f"\nFinal response: {result.final_output}")


# ---------------------------------------------------------------------------
# 2. Dynamic approval — approve based on the actual arguments
# ---------------------------------------------------------------------------

async def needs_approval_for_large_amount(
    ctx: RunContextWrapper, args: dict, call_id: str
) -> bool:
    """Only require approval when transfer amount exceeds $500."""
    amount = args.get("amount", 0)
    return float(amount) > 500


@function_tool(needs_approval=needs_approval_for_large_amount)
def transfer_funds(account_id: str, amount: float, recipient: str) -> str:
    """Transfer funds from an account to a recipient.

    Args:
        account_id: Source account ID.
        amount: Amount in USD to transfer.
        recipient: Recipient name or account.
    """
    return f"Transferred ${amount:.2f} from {account_id} to {recipient}. ✓"


async def dynamic_approval_demo():
    print("\n" + "=" * 50)
    print("PART 2: Dynamic approval — based on tool arguments")
    print("=" * 50)

    agent = Agent(
        name="BankingAgent",
        instructions=(
            "You are a banking assistant. Process fund transfer requests. "
            "Use the transfer_funds tool for each transfer requested."
        ),
        model="gpt-4o-mini",
        tools=[transfer_funds],
    )

    # Two transfers: one small (auto-approved), one large (needs human)
    request = (
        "Transfer $50 from ACC-001 to Alice, "
        "and also transfer $1500 from ACC-001 to Bob."
    )
    print(f"\nUser: {request}\n")
    print("(Transfers ≤$500 are auto-approved. >$500 need human sign-off)\n")

    result = await Runner.run(agent, input=request, max_turns=20)

    import json
    try:
        while result.interruptions:
            state = result.to_state()          # create state ONCE for this batch
            for item in result.interruptions:
                args_dict = json.loads(item.raw_item.arguments)
                approved = ask_human(item.raw_item.name, args_dict)
                if approved:
                    state.approve(item)
                    print(f"  [approved] ${args_dict.get('amount')} to {args_dict.get('recipient')}")
                else:
                    state.reject(item, always_reject=True)
                    print(f"  [rejected] ${args_dict.get('amount')} to {args_dict.get('recipient')}")

            result = await Runner.run(agent, input=state, max_turns=20)

        print(f"\nFinal response: {result.final_output}")
    except MaxTurnsExceeded:
        print("\n[run ended] Agent could not complete after rejections — max turns reached.")


# ---------------------------------------------------------------------------
# 3. Role-based approval — same tool, different needs_approval per role
# ---------------------------------------------------------------------------

def _update_config_impl(key: str, value: str) -> str:
    """Update a system configuration value.

    Args:
        key: Configuration key to update.
        value: New value to set.
    """
    return f"Config updated: {key} = {value}"


# Two tool instances from the same function — approval differs by role
update_config_guarded  = function_tool(_update_config_impl, name_override="update_config", needs_approval=True)
update_config_freepass = function_tool(_update_config_impl, name_override="update_config", needs_approval=False)


def make_config_agent(role: str) -> Agent:
    """Return a ConfigAgent with approval gating based on role."""
    tool = update_config_guarded if role != "admin" else update_config_freepass
    return Agent(
        name="ConfigAgent",
        instructions="You manage system configuration. Use update_config to apply changes.",
        model="gpt-4o-mini",
        tools=[tool],
    )


async def context_approval_demo():
    print("\n" + "=" * 50)
    print("PART 3: Role-based approval — admin auto-approved, editor needs sign-off")
    print("=" * 50)

    import json
    for role in ["editor", "admin"]:
        print(f"\n[Role: {role}] User: Set max_retries to 5 and timeout to 30.")
        agent = make_config_agent(role)
        result = await Runner.run(agent, input="Set max_retries to 5 and timeout to 30.")

        while result.interruptions:
            state = result.to_state()      # create state ONCE for this batch
            for item in result.interruptions:
                args_dict = json.loads(item.raw_item.arguments)
                approved = ask_human(item.raw_item.name, args_dict)
                if approved:
                    state.approve(item)
                    print(f"  [approved] {item.raw_item.name}")
                else:
                    state.reject(item, always_reject=True)
                    print(f"  [rejected] {item.raw_item.name}")
            result = await Runner.run(agent, input=state)   # no context= on resume

        print(f"  Response: {result.final_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await static_approval_demo()
    await dynamic_approval_demo()
    await context_approval_demo()


if __name__ == "__main__":
    asyncio.run(main())
