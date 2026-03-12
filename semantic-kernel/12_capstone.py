"""
Lesson 12 — Capstone: Research & Report Pipeline
==================================================
This capstone integrates every major SK concept from lessons 01–11 into a
single realistic pipeline: a multi-agent research assistant that ingests a
topic, researches it, fact-checks the findings, applies a HITL approval gate
for sensitive operations, logs all tool calls, and produces a final formatted
report.

Concepts used:
  01  Kernel + ChatHistory + invoke_prompt
  02  Native Plugins (@kernel_function)
  03  Prompt Templates ({{$variable}})
  04  Chat History (multi-turn context)
  05  Semantic Memory (embedding search)
  06  Function Calling (FunctionChoiceBehavior.Auto)
  07  Filters (AUTO_FUNCTION_INVOCATION — audit + HITL)
  08  ChatCompletionAgent (name, instructions, get_response)
  09  Multi-Agent (manual orchestration for Anthropic)
  10  Human-in-the-Loop (risk-tiered approval filter)
  11  MCP (reference — where real data sources would plug in)

Pipeline stages:
  Stage 1  — Researcher agent gathers key facts (uses ResearchPlugin tools)
  Stage 2  — Analyst agent identifies themes and gaps
  Stage 3  — HITL gate — "publish" action requires simulated operator approval
  Stage 4  — Writer agent composes the final report using a prompt template
  Stage 5  — Memory — store the report summary; search past reports on demand

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Kernel (Anthropic claude-sonnet-4-6)                   │
  │  ├─ ResearchPlugin  (search_web, fetch_stats, cite)     │
  │  ├─ PublishPlugin   (publish_report — HITL-gated)       │
  │  ├─ AUTO_FUNCTION_INVOCATION filter (audit + HITL)      │
  │  └─ SemanticTextMemory (report archive)                 │
  │                                                         │
  │  Agents (all share the kernel):                         │
  │    Researcher  →  Analyst  →  [HITL gate]  →  Writer   │
  └─────────────────────────────────────────────────────────┘
"""

import asyncio
import os
from datetime import datetime
from typing import Annotated, Callable, Awaitable

from dotenv import load_dotenv

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig

load_dotenv()

print("=" * 62)
print("Lesson 12 — Capstone: Research & Report Pipeline")
print("=" * 62)


# ---------------------------------------------------------------------------
# Simulated data — in production these call real APIs / MCP servers
# ---------------------------------------------------------------------------

RESEARCH_DB = {
    "semantic kernel": {
        "summary": "Semantic Kernel (SK) is Microsoft's open-source AI orchestration SDK.",
        "stats": {"github_stars": 23000, "weekly_downloads": 85000, "version": "1.40"},
        "citations": [
            "Microsoft Blog: 'Semantic Kernel reaches GA' (2024)",
            "GitHub: github.com/microsoft/semantic-kernel",
        ],
    },
    "model context protocol": {
        "summary": "MCP is Anthropic's open standard for AI-tool communication via stdio or SSE.",
        "stats": {"spec_version": "2024-11", "implementations": 120, "clients": 15},
        "citations": [
            "Anthropic: modelcontextprotocol.io",
            "GitHub: github.com/modelcontextprotocol/servers",
        ],
    },
    "ai agent frameworks": {
        "summary": "LangGraph, CrewAI, and Semantic Kernel are the leading Python agent frameworks.",
        "stats": {"frameworks_compared": 3, "year": 2024},
        "citations": [
            "LangChain Blog: 'LangGraph 0.2 released'",
            "CrewAI Docs: docs.crewai.com",
            "Microsoft: learn.microsoft.com/semantic-kernel",
        ],
    },
}

PUBLISH_LOG: list[dict] = []
AUDIT_LOG:   list[dict] = []


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------

class ResearchPlugin:
    """Simulated research tools — production would call APIs or MCP servers."""

    @kernel_function(name="search_web", description="Search for a topic and return a summary.")
    def search_web(
        self,
        topic: Annotated[str, "Topic to search"],
    ) -> Annotated[str, "Summary of search results"]:
        key = topic.lower().strip()
        for db_key, data in RESEARCH_DB.items():
            if db_key in key or key in db_key:
                return f"[search_web] Topic: {topic}\n{data['summary']}"
        return f"[search_web] No results found for '{topic}'."

    @kernel_function(name="fetch_stats", description="Fetch numerical statistics for a topic.")
    def fetch_stats(
        self,
        topic: Annotated[str, "Topic to fetch statistics for"],
    ) -> Annotated[str, "Statistics as a formatted string"]:
        key = topic.lower().strip()
        for db_key, data in RESEARCH_DB.items():
            if db_key in key or key in db_key:
                stats = ", ".join(f"{k}={v}" for k, v in data["stats"].items())
                return f"[fetch_stats] {topic}: {stats}"
        return f"[fetch_stats] No statistics for '{topic}'."

    @kernel_function(name="get_citations", description="Get citations/sources for a topic.")
    def get_citations(
        self,
        topic: Annotated[str, "Topic to cite"],
    ) -> Annotated[str, "List of citations"]:
        key = topic.lower().strip()
        for db_key, data in RESEARCH_DB.items():
            if db_key in key or key in db_key:
                cites = "\n".join(f"  - {c}" for c in data["citations"])
                return f"[get_citations] {topic}:\n{cites}"
        return f"[get_citations] No citations found for '{topic}'."


class PublishPlugin:
    """Publishing tools — gated by HITL filter in production."""

    @kernel_function(name="publish_report", description="Publish the final report to the platform.")
    def publish_report(
        self,
        title: Annotated[str, "Report title"],
        content: Annotated[str, "Report content"],
    ) -> Annotated[str, "Publish confirmation"]:
        entry = {
            "id":        f"RPT-{len(PUBLISH_LOG) + 1:04d}",
            "title":     title,
            "content":   content,
            "published": datetime.now(datetime.UTC).isoformat(),
        }
        PUBLISH_LOG.append(entry)
        return f"[publish_report] Report '{title}' published as {entry['id']}."

    @kernel_function(name="save_draft", description="Save a draft of the report (no approval needed).")
    def save_draft(
        self,
        title: Annotated[str, "Draft title"],
        content: Annotated[str, "Draft content"],
    ) -> Annotated[str, "Draft save confirmation"]:
        return f"[save_draft] Draft '{title}' saved locally ({len(content)} chars)."


# ---------------------------------------------------------------------------
# HITL + Audit filter
# ---------------------------------------------------------------------------

# Risk tiers for this pipeline
RISK_TIERS = {
    "search_web":     "SAFE",
    "fetch_stats":    "SAFE",
    "get_citations":  "SAFE",
    "save_draft":     "SAFE",
    "publish_report": "RISKY",   # requires human approval
}

# Simulated decisions (production: replace with input())
SIMULATED_DECISIONS = {
    "publish_report": True,   # approve publishing
}


def make_pipeline_filter() -> Callable:
    """Combined audit + HITL filter for the pipeline."""

    async def pipeline_filter(
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name
        args    = dict(context.arguments) if context.arguments else {}
        tier    = RISK_TIERS.get(fn_name, "SAFE")
        decision = "APPROVED"

        if tier == "SAFE":
            print(f"  [FILTER:SAFE]  Auto-approving {fn_name}")
            await next(context)

        elif tier == "RISKY":
            # Simulate human decision
            approved = SIMULATED_DECISIONS.get(fn_name, True)
            verdict  = "APPROVED" if approved else "DENIED"
            print(f"  [FILTER:HITL]  {fn_name} → human review → {verdict}")

            if approved:
                await next(context)
            else:
                from semantic_kernel.functions import FunctionResult
                context.function_result = FunctionResult(
                    function=context.function.metadata,
                    value="[HITL] Operator declined this action.",
                )
                context.terminate = True
                decision = "DENIED"

        else:  # BLOCKED
            print(f"  [FILTER:BLOCK] {fn_name} blocked by policy")
            from semantic_kernel.functions import FunctionResult
            context.function_result = FunctionResult(
                function=context.function.metadata,
                value="[HITL] Blocked by policy.",
            )
            context.terminate = True
            decision = "BLOCKED"

        # Audit entry for every call
        AUDIT_LOG.append({
            "ts":       datetime.now(datetime.UTC).isoformat(),
            "function": f"{context.function.plugin_name}.{fn_name}",
            "args":     {k: str(v)[:60] for k, v in args.items()},
            "decision": decision,
        })

    return pipeline_filter


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def make_kernel() -> tuple[sk.Kernel, AnthropicChatCompletion]:
    kernel  = sk.Kernel()
    service = AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
    kernel.add_service(service)
    kernel.add_plugin(ResearchPlugin(), plugin_name="research")
    kernel.add_plugin(PublishPlugin(),  plugin_name="publish")
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, make_pipeline_filter())
    return kernel, service


def auto_args(max_tokens: int = 500) -> KernelArguments:
    return KernelArguments(
        settings=PromptExecutionSettings(
            max_tokens=max_tokens,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
    )


# ---------------------------------------------------------------------------
# Multi-agent helper (Anthropic-compatible — fresh user message per turn)
# ---------------------------------------------------------------------------

async def call_agent(agent: ChatCompletionAgent, user_message: str, max_tokens: int = 500) -> str:
    """Call an agent with a constructed user message and return the reply."""
    response = await agent.get_response(
        messages=user_message,
        arguments=auto_args(max_tokens),
    )
    return str(response).strip()


# ---------------------------------------------------------------------------
# Stage 1 — Researcher agent
# ---------------------------------------------------------------------------

async def stage_researcher(kernel: sk.Kernel, topic: str) -> str:
    print(f"\n{'─'*55}")
    print("Stage 1: Researcher")
    print(f"{'─'*55}")

    researcher = ChatCompletionAgent(
        kernel=kernel,
        name="Researcher",
        instructions=(
            "You are a research specialist. Given a topic, use your tools to:\n"
            "  1. search_web   — get a summary\n"
            "  2. fetch_stats  — get numerical data\n"
            "  3. get_citations — get sources\n"
            "Compile everything into a structured research brief. "
            "Be factual and concise."
        ),
    )

    prompt = (
        f"Research the following topic thoroughly: '{topic}'\n"
        "Use all three research tools and compile a research brief."
    )
    findings = await call_agent(researcher, prompt, max_tokens=600)
    print(f"\n[Researcher output]\n{findings}\n")
    return findings


# ---------------------------------------------------------------------------
# Stage 2 — Analyst agent
# ---------------------------------------------------------------------------

async def stage_analyst(kernel: sk.Kernel, topic: str, research_brief: str) -> str:
    print(f"\n{'─'*55}")
    print("Stage 2: Analyst")
    print(f"{'─'*55}")

    analyst = ChatCompletionAgent(
        kernel=kernel,
        name="Analyst",
        instructions=(
            "You are a senior analyst. Given a research brief, you:\n"
            "  1. Identify 3 key themes\n"
            "  2. Spot any gaps or questions that need answers\n"
            "  3. Recommend a narrative angle for the final report\n"
            "Be crisp — bullet points preferred."
        ),
    )

    prompt = (
        f"Topic: {topic}\n\n"
        f"Research Brief:\n{research_brief}\n\n"
        "Analyse the brief: identify themes, gaps, and a narrative angle."
    )
    analysis = await call_agent(analyst, prompt, max_tokens=400)
    print(f"\n[Analyst output]\n{analysis}\n")
    return analysis


# ---------------------------------------------------------------------------
# Stage 3 — Writer agent + Prompt Template
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """
You are a technical writer producing a professional report.

Topic: {{$topic}}

Research Brief:
{{$research}}

Analysis:
{{$analysis}}

Write a clear, well-structured report with:
  1. Executive Summary (2-3 sentences)
  2. Key Findings (3-5 bullet points with stats where available)
  3. Conclusions & Recommendations (2-3 sentences)

Tone: professional, informative, suitable for a technical audience.
"""


async def stage_writer(
    kernel: sk.Kernel,
    service: AnthropicChatCompletion,
    topic: str,
    research: str,
    analysis: str,
) -> str:
    print(f"\n{'─'*55}")
    print("Stage 3: Writer (with Prompt Template)")
    print(f"{'─'*55}")

    # Use a prompt template (Lesson 03 pattern)
    config = PromptTemplateConfig(
        template=REPORT_TEMPLATE,
        name="report_template",
        description="Generates a formatted research report.",
    )

    fn = kernel.add_function(
        function_name="generate_report",
        plugin_name="writer",
        prompt_template_config=config,
    )

    result = await kernel.invoke(
        function=fn,
        arguments=KernelArguments(
            topic=topic,
            research=research,
            analysis=analysis,
            settings=PromptExecutionSettings(max_tokens=600),
        ),
    )
    report = str(result).strip()
    print(f"\n[Writer output]\n{report}\n")
    return report


# ---------------------------------------------------------------------------
# Stage 4 — HITL-gated publish
# ---------------------------------------------------------------------------

async def stage_publish(kernel: sk.Kernel, service: AnthropicChatCompletion, topic: str, report: str) -> None:
    print(f"\n{'─'*55}")
    print("Stage 4: Publish (HITL-gated)")
    print(f"{'─'*55}")

    publisher = ChatCompletionAgent(
        kernel=kernel,
        name="Publisher",
        instructions=(
            "You handle report publishing. "
            "Always save a draft first with save_draft, then publish the final report with publish_report. "
            "Use the exact title and content provided."
        ),
    )

    title  = f"Research Report: {topic} ({datetime.now(datetime.UTC).strftime('%Y-%m-%d')})"
    prompt = (
        f"Please save a draft and then publish this report.\n\n"
        f"Title: {title}\n\n"
        f"Content:\n{report}"
    )
    result = await call_agent(publisher, prompt, max_tokens=300)
    print(f"\n[Publisher output]\n{result}\n")


# ---------------------------------------------------------------------------
# Stage 5 — Memory: store and search report summaries
# ---------------------------------------------------------------------------

async def stage_memory(kernel: sk.Kernel, service: AnthropicChatCompletion, topic: str, report: str) -> None:
    print(f"\n{'─'*55}")
    print("Stage 5: Memory (Semantic Search)")
    print(f"{'─'*55}")

    # Use SK's in-memory vector store (Lesson 05 pattern with volatile store)
    try:
        from semantic_kernel.memory import SemanticTextMemory
        from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
        from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

        embedding_service = OpenAITextEmbedding(
            ai_model_id="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            service_id="openai_embed",
        )

        memory_store = VolatileMemoryStore()
        memory = SemanticTextMemory(storage=memory_store, embeddings_generator=embedding_service)

        # Store the report summary
        summary = report[:300] + "..." if len(report) > 300 else report
        await memory.save_information(
            collection="reports",
            text=summary,
            id=f"report_{datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}",
            description=f"Research report on: {topic}",
        )
        print(f"  [Memory] Report summary stored for topic: '{topic}'")

        # Search for relevant past reports
        results = await memory.search(collection="reports", query=topic, limit=1)
        if results:
            print(f"  [Memory] Found related report: {results[0].description}")
            print(f"  [Memory] Relevance: {results[0].relevance:.2f}")
        else:
            print(f"  [Memory] No related reports found.")

    except Exception as e:
        print(f"  [Memory] Skipping (requires OpenAI embeddings): {e}")
        print("  [Memory] Pattern: save_information(collection, text, id) → search(collection, query)")


# ---------------------------------------------------------------------------
# Summary: Audit log + published reports
# ---------------------------------------------------------------------------

def print_pipeline_summary() -> None:
    print(f"\n{'='*62}")
    print("Pipeline Summary")
    print(f"{'='*62}")

    print(f"\n[AUDIT LOG] — {len(AUDIT_LOG)} tool calls intercepted:")
    for entry in AUDIT_LOG:
        fn   = entry["function"].split(".")[-1]
        tier = RISK_TIERS.get(fn, "SAFE")
        print(f"  {entry['ts']}  [{tier:5s}]  {entry['function']:35s}  {entry['decision']}")

    print(f"\n[PUBLISH LOG] — {len(PUBLISH_LOG)} reports published:")
    for p in PUBLISH_LOG:
        print(f"  {p['id']}  {p['title']}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def main():
    topic = "Semantic Kernel"

    print(f"\nPipeline topic: '{topic}'")
    print("Agents: Researcher → Analyst → Writer → Publisher")
    print("Filters: AUTO_FUNCTION_INVOCATION (audit + HITL)")

    kernel, service = make_kernel()

    # Run all pipeline stages
    research = await stage_researcher(kernel, topic)
    analysis = await stage_analyst(kernel, topic, research)
    report   = await stage_writer(kernel, service, topic, research, analysis)
    await stage_publish(kernel, service, topic, report)
    await stage_memory(kernel, service, topic, report)

    print_pipeline_summary()

    print(f"\n{'='*62}")
    print("Key Takeaways (all 12 lessons):")
    print("  01  Kernel is the central object — services + plugins + invoke")
    print("  02  @kernel_function turns any method into an SK tool")
    print("  03  Prompt templates: {{$variable}} for reusable, parameterised prompts")
    print("  04  ChatHistory tracks multi-turn context; windowing prevents bloat")
    print("  05  Semantic memory: embeddings → save/search past knowledge")
    print("  06  FunctionChoiceBehavior.Auto() → agent picks tools automatically")
    print("  07  Filters: intercept function calls, prompts, and auto-invoke loops")
    print("  08  ChatCompletionAgent: named, instructed agent with thread support")
    print("  09  Multi-agent: manual orchestration (Anthropic) or AgentGroupChat (OpenAI)")
    print("  10  HITL: AUTO_FUNCTION_INVOCATION filter + await next / terminate")
    print("  11  MCP: MCPStdioPlugin wraps any MCP server as SK plugin")
    print("  12  Capstone: all patterns compose into a production-like pipeline")
    print()
    print("  Real-world next step: swap simulated tools for real MCP servers,")
    print("  public APIs, and a production vector store.")
    print(f"{'='*62}")


if __name__ == "__main__":
    asyncio.run(main())
