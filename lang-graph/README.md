# LangGraph Learning

Hands-on examples for learning [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building stateful, multi-actor LLM applications using graph-based orchestration.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

## Examples

| File | What you learn |
|------|---------------|
| [01_simple_graph.py](01_simple_graph.py) | Build a minimal `StateGraph` with one node and one edge. Understand how state flows through a graph and how `invoke()` works. |
| [02_conditional_edges.py](02_conditional_edges.py) | Bind tools to an LLM and let it decide whether to call them. Use `add_conditional_edges` to route to the tools node or end the graph. |
| [03_memory_checkpointing.py](03_memory_checkpointing.py) | Add `MemorySaver` to persist conversation history across multiple `invoke()` calls. Each `thread_id` gets its own isolated memory. |
| [04_human_in_the_loop.py](04_human_in_the_loop.py) | Pause the graph before executing sensitive tools using `interrupt_before`. Resume or reject based on human approval. |
| [05_multi_agent.py](05_multi_agent.py) | Supervisor pattern with two specialized agents (researcher + writer). Deterministic routing via a `completed` list in shared state. |
| [06_streaming.py](06_streaming.py) | Stream graph output three ways: node-by-node updates, full state snapshots, and individual LLM tokens for a typewriter effect. |
| [07_subgraphs.py](07_subgraphs.py) | Compose a parent graph from reusable compiled subgraphs. Each subgraph handles one stage (extract → summarize → classify) independently. |
| [08_parallel_execution.py](08_parallel_execution.py) | Run multiple nodes simultaneously. Static fan-out with parallel edges, and dynamic fan-out using the `Send` API to spawn N workers at runtime. |
| [09_sqlite_persistence.py](09_sqlite_persistence.py) | Replace `MemorySaver` with `SqliteSaver` for disk-based persistence. Conversations survive app restarts and can be resumed by `thread_id`. |
| [10_react_agent_from_scratch.py](10_react_agent_from_scratch.py) | Build the ReAct (Reason+Act) loop manually without prebuilts. See every step — reasoning, tool execution, observation — as the agent cycles until it has a final answer. |
| [11_end_to_end_agent.py](11_end_to_end_agent.py) | Capstone: combines ReAct loop, SQLite memory, human-in-the-loop approval, streaming, and multiple tools into a production-ready interactive research assistant. |

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangChain Docs](https://python.langchain.com/docs/)
- [LangSmith (tracing)](https://smith.langchain.com/)
