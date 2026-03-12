"""
Lesson 10 — Human-in-the-Loop (HITL)
=======================================
HITL means pausing the agent's tool-call loop and asking a human to
approve, modify, or reject each action before it executes.

In SK, HITL is built on the AUTO_FUNCTION_INVOCATION filter (Lesson 07).
The filter runs BETWEEN the LLM deciding to call a tool and SK executing
it — the perfect interception point for human review.

Comparison to what you know:
  LangGraph  → interrupt() inside a node + graph.update_state() to resume
  CrewAI     → human_input=True on a Task — pauses and asks at runtime
  SK 1.40    → AUTO_FUNCTION_INVOCATION filter + context.terminate / skip

Key control points in the filter:
  await next(context)       → allow the tool call to proceed
  return (no next call)     → skip this tool call (result stays None)
  context.terminate = True  → stop the entire auto-invoke loop

IMPORTANT: The filter intercepts AFTER the LLM has already decided what to
call. The conversation at that point is:
  [system, user, assistant(tool_use)] — valid for Anthropic.
So HITL via filter is fully Anthropic-compatible.

Patterns covered:
  1. Basic approval gate — pause and ask before every tool call (simulated)
  2. Risk-tiered policy  — auto-allow safe tools, block/ask for risky ones
  3. Argument inspection — show human what arguments will be used
  4. Audit log           — record every decision for compliance
  5. Interactive mode    — real input() version (shown but not run in script)
"""

import asyncio
import time
from datetime import datetime
from typing import Annotated, Callable, Awaitable
from dotenv import load_dotenv
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.functions import kernel_function

load_dotenv()

print("=" * 62)
print("Lesson 10 — Human-in-the-Loop (HITL)")
print("=" * 62)


# ---------------------------------------------------------------------------
# Shared plugins
# ---------------------------------------------------------------------------
class FilePlugin:
    """Simulated file operations — no real disk access."""

    @kernel_function(name="read_file", description="Read a file by path.")
    def read_file(
        self,
        path: Annotated[str, "File path to read"],
    ) -> Annotated[str, "File contents"]:
        return f"[Contents of {path}]: Hello, this is simulated file content."

    @kernel_function(name="write_file", description="Write content to a file.")
    def write_file(
        self,
        path: Annotated[str, "File path to write"],
        content: Annotated[str, "Content to write"],
    ) -> Annotated[str, "Result"]:
        return f"[Written to {path}]: {len(content)} bytes saved."

    @kernel_function(name="delete_file", description="Permanently delete a file.")
    def delete_file(
        self,
        path: Annotated[str, "File path to delete"],
    ) -> Annotated[str, "Result"]:
        return f"[Deleted]: {path} has been permanently removed."


class DatabasePlugin:
    """Simulated database operations."""

    @kernel_function(name="query_db", description="Run a SELECT query on the database.")
    def query_db(
        self,
        query: Annotated[str, "SQL SELECT query"],
    ) -> Annotated[str, "Query results"]:
        return f"[DB Result]: 3 rows returned for: {query}"

    @kernel_function(name="update_db", description="Run an UPDATE or DELETE on the database.")
    def update_db(
        self,
        query: Annotated[str, "SQL UPDATE/DELETE query"],
    ) -> Annotated[str, "Result"]:
        return f"[DB Updated]: Query executed: {query}"


def make_kernel() -> tuple[sk.Kernel, AnthropicChatCompletion]:
    kernel = sk.Kernel()
    service = AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
    kernel.add_service(service)
    kernel.add_plugin(FilePlugin(), plugin_name="files")
    kernel.add_plugin(DatabasePlugin(), plugin_name="db")
    return kernel, service


def auto_settings(max_tokens: int = 300) -> PromptExecutionSettings:
    return PromptExecutionSettings(
        max_tokens=max_tokens,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )


async def run_prompt(service, kernel, prompt: str, max_tokens: int = 300) -> str:
    history = ChatHistory()
    history.add_user_message(prompt)
    response = await service.get_chat_message_content(
        chat_history=history,
        settings=auto_settings(max_tokens),
        kernel=kernel,
    )
    return str(response)


# ---------------------------------------------------------------------------
# 1. Basic Approval Gate
# ---------------------------------------------------------------------------
# Every tool call is intercepted. A simulated "human" either approves or
# denies it. In production, replace `simulated_approve()` with `input()`.
#
# LangGraph equivalent:  interrupt() + Command(resume=...) in the node
# CrewAI equivalent:     human_input=True on Task

# Simulated approval table — True = approve, False = deny
# In production: replace with input("Approve? [y/n]: ").strip().lower() == "y"
SIMULATED_ANSWERS: dict[str, bool] = {
    "read_file":  True,
    "write_file": True,
    "delete_file": False,   # always deny deletes in this demo
    "query_db":   True,
    "update_db":  False,    # deny DB writes
}


def simulated_approve(fn_name: str, args: dict) -> bool:
    """Simulate a human reading the tool call and deciding yes/no."""
    decision = SIMULATED_ANSWERS.get(fn_name, True)
    verdict = "APPROVED" if decision else "DENIED"
    print(f"    [HUMAN] Tool: {fn_name}  Args: {args}  → {verdict}")
    return decision


def make_basic_approval_filter():
    async def approval_filter(
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name
        args = {k: v for k, v in context.arguments.items()} if context.arguments else {}

        if simulated_approve(fn_name, args):
            await next(context)   # approved — run the tool
        else:
            # Denied — inject a refusal message as the result
            from semantic_kernel.functions import FunctionResult
            context.function_result = FunctionResult(
                function=context.function.metadata,
                value="[HITL] Tool call denied by human operator.",
            )
            context.terminate = True  # stop the loop

    return approval_filter


async def basic_approval_example():
    print("\n--- 1. Basic Approval Gate ---")
    kernel, service = make_kernel()
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, make_basic_approval_filter())

    prompts = [
        "Read the file at /etc/config.txt",
        "Delete the file at /tmp/old_report.csv",
    ]
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = await run_prompt(service, kernel, prompt)
        print(f"Agent: {result}")


# ---------------------------------------------------------------------------
# 2. Risk-Tiered Policy
# ---------------------------------------------------------------------------
# Three tiers:
#   SAFE    — auto-approve (read-only operations)
#   RISKY   — simulated human approval required
#   BLOCKED — always deny, no human needed
#
# This is how real production HITL works: not every action needs a human.

RISK_TIERS = {
    "read_file":  "SAFE",
    "query_db":   "SAFE",
    "write_file": "RISKY",
    "update_db":  "RISKY",
    "delete_file": "BLOCKED",
}


def make_tiered_filter():
    async def tiered_filter(
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name
        tier = RISK_TIERS.get(fn_name, "RISKY")

        if tier == "SAFE":
            print(f"  [TIER:SAFE]    Auto-approving {fn_name}")
            await next(context)

        elif tier == "RISKY":
            args = dict(context.arguments) if context.arguments else {}
            # Production: approved = input(f"Approve {fn_name}({args})? [y/n]: ") == "y"
            approved = simulated_approve(fn_name, args)
            if approved:
                await next(context)
            else:
                from semantic_kernel.functions import FunctionResult
                context.function_result = FunctionResult(
                    function=context.function.metadata,
                    value="[HITL] Operator declined this action.",
                )
                context.terminate = True

        else:  # BLOCKED
            print(f"  [TIER:BLOCKED] {fn_name} is always denied — policy.")
            from semantic_kernel.functions import FunctionResult
            context.function_result = FunctionResult(
                function=context.function.metadata,
                value="[HITL] This operation is blocked by policy.",
            )
            context.terminate = True

    return tiered_filter


async def tiered_policy_example():
    print("\n--- 2. Risk-Tiered Policy (SAFE / RISKY / BLOCKED) ---")
    kernel, service = make_kernel()
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, make_tiered_filter())

    prompts = [
        "Run this query: SELECT * FROM users WHERE id=1",
        "Update user email: UPDATE users SET email='x@y.com' WHERE id=1",
        "Delete the file /var/log/audit.log",
    ]
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = await run_prompt(service, kernel, prompt)
        print(f"Agent: {result}")


# ---------------------------------------------------------------------------
# 3. Audit Log
# ---------------------------------------------------------------------------
# Every tool call decision is written to an in-memory audit log.
# In production: write to a database, file, or observability system.
# This is important for compliance and debugging in agentic systems.

audit_log: list[dict] = []


def make_audit_filter():
    async def audit_filter(
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name
        args = dict(context.arguments) if context.arguments else {}
        tier = RISK_TIERS.get(fn_name, "RISKY")

        # Decide
        if tier == "SAFE":
            decision = "APPROVED"
            await next(context)
        elif tier == "RISKY":
            approved = simulated_approve(fn_name, args)
            decision = "APPROVED" if approved else "DENIED"
            if approved:
                await next(context)
            else:
                from semantic_kernel.functions import FunctionResult
                context.function_result = FunctionResult(
                    function=context.function.metadata,
                    value="[HITL] Action declined.",
                )
                context.terminate = True
        else:
            decision = "BLOCKED"
            from semantic_kernel.functions import FunctionResult
            context.function_result = FunctionResult(
                function=context.function.metadata,
                value="[HITL] Blocked by policy.",
            )
            context.terminate = True

        # Write audit entry
        audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "function":  f"{context.function.plugin_name}.{fn_name}",
            "args":      args,
            "decision":  decision,
        })

    return audit_filter


async def audit_log_example():
    print("\n--- 3. Audit Log ---")
    audit_log.clear()

    kernel, service = make_kernel()
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, make_audit_filter())

    prompts = [
        "Read /etc/hosts and then write a summary to /tmp/summary.txt",
        "Delete /tmp/old.log",
    ]
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        result = await run_prompt(service, kernel, prompt, max_tokens=400)
        print(f"Agent: {result}")

    print("\n  [AUDIT LOG]")
    for entry in audit_log:
        print(f"  {entry['timestamp']}  {entry['function']:30s}  {entry['decision']}")


# ---------------------------------------------------------------------------
# 4. Interactive mode reference (not run automatically)
# ---------------------------------------------------------------------------
# This shows what the real input() version looks like.
# Uncomment and run manually if you want interactive HITL.

def make_interactive_filter():
    """
    INTERACTIVE version — replace simulated_approve() with real input().
    Uncomment the input() line and remove the simulated line to use.
    """
    async def interactive_filter(
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        fn = f"{context.function.plugin_name}.{context.function.name}"
        args = dict(context.arguments) if context.arguments else {}
        print(f"\n  [HITL] Agent wants to call: {fn}")
        print(f"         Arguments: {args}")

        # --- PRODUCTION: use real input ---
        # answer = input("  Approve? [y/n]: ").strip().lower()
        # approved = answer == "y"

        # --- SCRIPT: simulate approval ---
        approved = simulated_approve(context.function.name, args)

        if approved:
            await next(context)
        else:
            from semantic_kernel.functions import FunctionResult
            context.function_result = FunctionResult(
                function=context.function.metadata,
                value="[HITL] Action rejected by operator.",
            )
            context.terminate = True

    return interactive_filter


async def interactive_example():
    print("\n--- 4. Interactive HITL (simulated — see make_interactive_filter) ---")
    kernel, service = make_kernel()
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, make_interactive_filter())

    prompt = "Read the file /etc/passwd and then delete /tmp/test.txt"
    print(f"User: {prompt}")
    result = await run_prompt(service, kernel, prompt)
    print(f"Agent: {result}")
    print("  (In production: replace simulated_approve() with input() for real approval)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await basic_approval_example()
    await tiered_policy_example()
    await audit_log_example()
    await interactive_example()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • HITL = AUTO_FUNCTION_INVOCATION filter as interception point")
    print("  • await next(context)          → allow the tool to run")
    print("  • skip next() + set result     → deny silently, return message")
    print("  • context.terminate = True     → stop the entire tool loop")
    print("  • Risk tiers: SAFE=auto / RISKY=ask-human / BLOCKED=always-deny")
    print("  • Audit log: record every call + decision for compliance")
    print("  • Production: replace simulated_approve() with input() or a UI")
    print("  • Filter runs AFTER LLM decides — Anthropic-compatible ✓")
    print("  • Next: Lesson 11 — MCP Integration")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
