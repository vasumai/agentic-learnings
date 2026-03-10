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

*(Files will be added here as I progress)*

| File | Description |
|------|-------------|
| — | — |

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangChain Docs](https://python.langchain.com/docs/)
- [LangSmith (tracing)](https://smith.langchain.com/)
