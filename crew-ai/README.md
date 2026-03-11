# CrewAI Learning

Hands-on examples for learning [CrewAI](https://docs.crewai.com/) — a framework for building role-based multi-agent systems where AI agents collaborate like a team of specialists to accomplish complex goals.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

## Examples

| File | What you learn |
|------|----------------|
| [01_hello_agent.py](01_hello_agent.py) | Create your first Agent with a role, goal, and backstory. Wrap it in a Task and a Crew. Understand the three core primitives of CrewAI. |
| [02_agent_with_tools.py](02_agent_with_tools.py) | Equip an agent with `SerperDevTool` (web search) and `FileWriterTool`. Use `{placeholders}` in task descriptions and inject values via `kickoff(inputs={})`. |
| [03_first_crew.py](03_first_crew.py) | Two-agent pipeline: Researcher gathers findings, Writer produces a blog post. See how task output flows automatically to the next task in sequential mode. |
| [04_task_dependencies.py](04_task_dependencies.py) | Three-agent pipeline with explicit `context=[]` wiring. Control precisely which task output feeds into which task, instead of relying on implicit sequential handoff. |
| [05_hierarchical_crew.py](05_hierarchical_crew.py) | Add a Manager agent that reads the goal and delegates tasks to specialists. Use `Process.hierarchical` — no manual task assignment needed. |
| [06_custom_tools.py](06_custom_tools.py) | Build your own tools with the `@tool` decorator. Understand how CrewAI describes tools to the LLM and how agents decide when to use them. |
| [07_structured_output.py](07_structured_output.py) | Return typed, validated data from tasks using Pydantic models. Use `output_pydantic=` on tasks instead of raw text output. |
| [08_memory.py](08_memory.py) | Enable short-term, long-term, and entity memory on a Crew. Understand how agents remember context across tasks and across runs. |
| [09_human_in_the_loop.py](09_human_in_the_loop.py) | Pause agent execution mid-task to collect human input. Use `human_input=True` on tasks that require approval or additional context. |
| [10_flow_basics.py](10_flow_basics.py) | Introduction to CrewAI Flows — event-driven orchestration with `@start`, `@listen`, and structured state. The step up from Crew. |
| [11_flow_with_crew.py](11_flow_with_crew.py) | Combine Flows and Crews: a Flow that orchestrates multiple Crews, branches on conditions, and maintains shared state across the pipeline. |
| [12_real_world_pipeline.py](12_real_world_pipeline.py) | Capstone: a full research → analysis → write → review pipeline with real tools, structured outputs, memory, and human review gates. |

## API Keys Required

| Key | Where to get it | Used in |
|-----|----------------|---------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | All files |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) (free tier: 2500 searches) | Files 02, 12 |

## Resources

- [CrewAI Docs](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Tools Reference](https://docs.crewai.com/concepts/tools)
- [CrewAI Flows](https://docs.crewai.com/concepts/flows)

---

> New to agentic AI frameworks? See the [LangGraph vs CrewAI comparison](../LANGGRAPH_VS_CREWAI.md) at the repo root for a detailed side-by-side breakdown of when to use each.
