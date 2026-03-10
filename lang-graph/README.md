# LangGraph Learning

Learning [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building stateful, multi-actor applications with LLMs using graph-based orchestration.

## Core Concepts

- **StateGraph** — define a graph where nodes are functions and edges control flow
- **State** — a shared typed dict passed between all nodes
- **Nodes** — Python functions that read/write state
- **Edges** — connections between nodes (conditional or fixed)
- **Checkpointing** — built-in state persistence for long-running agents

## Setup

```bash
# Create virtual environment (from this folder)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your actual keys
```

## Examples

| File | Concept | Key APIs |
|------|---------|----------|
| [01_simple_graph.py](01_simple_graph.py) | StateGraph basics — nodes, edges, state | `StateGraph`, `add_edge`, `compile`, `invoke` |
| [02_conditional_edges.py](02_conditional_edges.py) | Tool calling and conditional routing | `bind_tools`, `ToolNode`, `add_conditional_edges` |
| [03_memory_checkpointing.py](03_memory_checkpointing.py) | Multi-turn memory with thread isolation | `MemorySaver`, `thread_id`, `get_state` |
| [04_human_in_the_loop.py](04_human_in_the_loop.py) | Pause graph for human approval before acting | `interrupt_before`, resume with `invoke(None, config)` |
| [05_multi_agent.py](05_multi_agent.py) | Supervisor pattern — researcher + writer agents | Multi-node routing, `completed` state tracking |
| [06_streaming.py](06_streaming.py) | Stream output token-by-token or node-by-node | `stream`, `astream_events`, `on_chat_model_stream` |

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangChain Docs](https://python.langchain.com/docs/)
- [LangSmith (tracing)](https://smith.langchain.com/)
