# Semantic Kernel Learning Path

Microsoft's open-source AI orchestration SDK for building agents, plugins, and pipelines.

## Setup

```bash
cd semantic-kernel
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

## Lessons

| # | File | Concepts |
|---|------|----------|
| 01 | [01_hello_semantic_kernel.py](01_hello_semantic_kernel.py) | Kernel, ChatHistory, invoke_prompt, multi-turn chat |
| 02 | _coming soon_ | Native Plugins (functions the Kernel can call) |
| 03 | _coming soon_ | Prompt Templates with variables |
| 04 | _coming soon_ | Semantic Memory + Embeddings |
| 05 | _coming soon_ | Function Calling / Tool Use |
| 06 | _coming soon_ | Planners (auto function selection) |
| 07 | _coming soon_ | Filters & Middleware |
| 08 | _coming soon_ | Agents (ChatCompletionAgent) |
| 09 | _coming soon_ | Multi-Agent Collaboration |
| 10 | _coming soon_ | Human-in-the-Loop |
| 11 | _coming soon_ | MCP Integration |
| 12 | _coming soon_ | Capstone — Real-World Pipeline |

## How SK Compares

| Concept | Semantic Kernel | CrewAI | LangGraph |
|---------|----------------|--------|-----------|
| Core unit | Kernel | Crew | Graph |
| Functions | Plugins | Tools | Node functions |
| Multi-agent | Agent + AgentGroup | Crew(agents) | Nodes as agents |
| Memory | SK Memory / VectorStore | Built-in memory=True | LangGraph Checkpointer |
| Orchestration | Planner / invoke() | Task pipeline | Edge routing |
