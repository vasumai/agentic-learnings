# Agentic AI Learning Journey

Exploring agentic AI frameworks through hands-on examples.

## Frameworks

| Folder | Framework | Description |
|--------|-----------|-------------|
| [lang-graph/](lang-graph/) | LangGraph | Graph-based agent orchestration by LangChain |
| *(coming soon)* | CrewAI | Role-based multi-agent framework |

## Structure

Each framework lives in its own subfolder with:
- Its own Python **virtual environment** (`venv/`) — not committed
- Its own **`requirements.txt`**
- Its own **`README.md`** with setup and learning notes
- Its own **`.env`** file for API keys — not committed (use `.env.example` as template)

## Getting Started

1. Clone the repo
2. Navigate to a framework folder
3. Create and activate the virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env` and fill in your API keys
