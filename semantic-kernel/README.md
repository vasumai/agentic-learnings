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
| 02 | [02_plugins.py](02_plugins.py) | Native Plugins — @kernel_function, direct & auto invocation |
| 03 | [03_prompt_templates.py](03_prompt_templates.py) | Prompt Templates — {{$variable}}, reuse, PromptTemplateConfig |
| 04 | [04_chat_history.py](04_chat_history.py) | Chat History — multi-turn, persona, windowing, serialise, template |
| 05 | [05_semantic_memory.py](05_semantic_memory.py) | Semantic Memory — embeddings, save/search, collections, TextMemoryPlugin |
| 06 | [06_function_calling.py](06_function_calling.py) | Function Calling — Auto/Required/NoneInvoke, filters, loop depth |
| 07 | [07_filters.py](07_filters.py) | Filters & Middleware — FUNCTION_INVOCATION, PROMPT_RENDERING, AUTO_FUNCTION_INVOCATION |
| 08 | [08_chat_completion_agent.py](08_chat_completion_agent.py) | ChatCompletionAgent — name/instructions, plugins, ChatHistoryAgentThread, streaming |
| 09 | [09_multi_agent.py](09_multi_agent.py) | Multi-Agent — AgentGroupChat, SequentialSelection, KernelFunctionSelection/Termination |
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
