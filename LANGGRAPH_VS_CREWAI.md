# LangGraph vs CrewAI — A Practical Comparison

> Built from hands-on experience implementing 12 examples in each framework.
> Not marketing — real trade-offs, real decisions.

---

## The One-Line Summary

**LangGraph** = you are the architect. You design the graph, define state, draw every edge.
**CrewAI** = you are the manager. You hire agents, assign tasks, and let the crew figure it out.

---

## Core Philosophy

| | LangGraph | CrewAI |
|---|---|---|
| **Mental model** | State machine / graph | Team of specialists |
| **You control** | Every node, edge, and state transition | Agent roles, task descriptions, process type |
| **Framework controls** | Nothing — it just executes your graph | Task routing, context handoff, agent loops |
| **Abstraction level** | Low — close to the metal | High — opinionated and convention-driven |
| **Primary question** | "What is the shape of my computation?" | "Who should do what, in what order?" |

---

## The Five Core Concepts Side by Side

### 1. Unit of Work

| LangGraph | CrewAI |
|---|---|
| **Node** — a Python function | **Task** — a natural language description |
| You write the logic in code | You describe the goal in plain English |
| `def my_node(state): ...` | `Task(description="Research X and produce Y")` |

In LangGraph you *implement* the work. In CrewAI you *describe* the work and the agent figures out how to do it.

### 2. Who Does the Work

| LangGraph | CrewAI |
|---|---|
| **The graph** — deterministic function calls | **Agent** — an LLM persona with role + goal + backstory |
| No concept of "persona" | Agent backstory shapes behavior like a system prompt |
| Tools are bound directly to the LLM node | Tools are given to an Agent; it decides when to use them |

```python
# LangGraph: you bind tools to an LLM call
llm_with_tools = llm.bind_tools([search, calculator])

# CrewAI: you give tools to an Agent persona
agent = Agent(role="Analyst", tools=[search, calculator])
```

### 3. State Management

| LangGraph | CrewAI |
|---|---|
| **Explicit typed state** — you define a `TypedDict` schema | **No state schema** — task outputs are free text |
| State persists and updates across every node | Each task receives upstream output as context |
| You control what gets stored and how | CrewAI handles context passing automatically |
| Fine-grained — add/remove specific fields | Coarse-grained — full text flows forward |

```python
# LangGraph: you define state explicitly
class AgentState(TypedDict):
    messages: list
    tool_calls: int
    approved: bool

# CrewAI: no schema — output flows as text
# Task 1 output automatically becomes Task 2 context
```

### 4. Routing and Flow Control

| LangGraph | CrewAI |
|---|---|
| **You draw edges** — `add_edge`, `add_conditional_edges` | **Process type** — Sequential or Hierarchical |
| Routing logic is Python code you write | Routing is handled by the framework (or a Manager agent) |
| Can route based on any state value | Sequential: fixed order; Hierarchical: manager decides |
| Supports cycles, retries, complex branching | Linear pipelines are natural; complex branching needs Flows |

```python
# LangGraph: you define every route explicitly
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tool_node",
    "end": END
})

# CrewAI: sequential = automatic
crew = Crew(tasks=[task1, task2, task3], process=Process.sequential)

# CrewAI: hierarchical = manager delegates
crew = Crew(tasks=[task1, task2], process=Process.hierarchical, manager_llm=llm)
```

### 5. Memory and Persistence

| LangGraph | CrewAI |
|---|---|
| **Checkpointers** — MemorySaver, SqliteSaver, PostgresSaver | **Memory types** — short-term, long-term, entity, contextual |
| Tied to `thread_id` — resumes exact graph state | Long-term memory persists across crew runs |
| You choose when to checkpoint | CrewAI manages memory storage automatically |
| Best for conversation continuity | Best for agents that should "remember" facts over time |

---

## Architecture Diagrams

### LangGraph — You Draw the Graph

```
      ┌──────────────┐
      │  START       │
      └──────┬───────┘
             │
      ┌──────▼───────┐
      │  agent_node  │◄──────────────┐
      └──────┬───────┘               │
             │                       │
     ┌───────▼────────┐              │
     │ conditional    │              │
     │ edge (you)     │              │
     └───┬────────┬───┘              │
         │        │                  │
  ┌──────▼──┐  ┌──▼──────┐          │
  │tool_node│  │  END    │          │
  └──────┬──┘  └─────────┘          │
         │                           │
         └───────────────────────────┘
               (cycle back — you designed this)
```

### CrewAI Sequential — Framework Handles Flow

```
  kickoff(inputs)
       │
  ┌────▼──────────────┐
  │  Task 1           │  ← Agent A works on it
  │  (description)    │
  └────┬──────────────┘
       │  output (text)
       ▼
  ┌────────────────────┐
  │  Task 2            │  ← Agent B gets Task 1 output as context
  │  (description)     │
  └────┬───────────────┘
       │  output (text)
       ▼
  ┌────────────────────┐
  │  Task 3            │  ← Agent C gets Task 2 output as context
  └────────────────────┘
       │
  final result
```

### CrewAI Hierarchical — Manager Delegates

```
  kickoff(inputs)
       │
  ┌────▼──────────────┐
  │  Manager Agent    │  ← LLM decides who does what
  └──┬───┬────┬───────┘
     │   │    │
  ┌──▼┐ ┌▼─┐ ┌▼──┐
  │A1 │ │A2│ │A3 │  ← Specialists work in parallel or sequence
  └───┘ └──┘ └───┘
     │   │    │
  ┌──▼───▼────▼───┐
  │  Manager      │  ← Aggregates and produces final output
  └───────────────┘
```

---

## Real Decision Guide: When to Use Which

### Choose LangGraph when:

- **You need precise control** — exact routing logic, custom retry strategies, specific state transitions
- **Your workflow has complex branching** — multiple conditional paths, cycles, error recovery loops
- **State shape matters** — you need typed, structured state that different parts of the graph read/write independently
- **You want to stream granularly** — token-level streaming, node-by-node updates, custom stream modes
- **You are building infrastructure** — a reusable agentic component that other systems will call
- **Human-in-the-loop is critical** — interrupt at specific nodes, review exact state, resume with precision
- **Debugging matters a lot** — LangSmith tracing gives you full visibility into every state transition

### Choose CrewAI when:

- **You think in roles** — your problem naturally maps to "a researcher does X, a writer does Y"
- **Speed of development matters** — you want agents collaborating in 20 lines of code
- **Tasks are naturally language-driven** — the work is described better in English than in code
- **You want a manager pattern** — let a Manager agent dynamically delegate without hardcoding routing
- **Your pipeline is mostly linear** — research → analyze → write → review is a natural fit
- **Structured outputs matter** — Pydantic model outputs from tasks with minimal boilerplate
- **You want built-in memory** — entity memory, long-term memory across runs without DIY storage

---

## Code Complexity Comparison

### Same task: "Research a topic, then write a blog post"

**LangGraph version:**
```python
class State(TypedDict):
    topic: str
    research: str
    blog_post: str

def research_node(state):
    result = llm.invoke(f"Research: {state['topic']}")
    return {"research": result.content}

def write_node(state):
    result = llm.invoke(f"Write blog post using: {state['research']}")
    return {"blog_post": result.content}

graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge(START, "research")
graph.add_edge("research", "write")
graph.add_edge("write", END)
app = graph.compile()
result = app.invoke({"topic": "AI agents"})
```

**CrewAI version:**
```python
researcher = Agent(role="Researcher", goal="Research {topic}", backstory="...")
writer = Agent(role="Writer", goal="Write blog post about {topic}", backstory="...")

research_task = Task(description="Research {topic}", agent=researcher)
write_task = Task(description="Write blog post using research", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff(inputs={"topic": "AI agents"})
```

CrewAI is significantly less boilerplate for standard pipelines. LangGraph gives you more control over what happens inside each step.

---

## Honest Trade-offs

| | LangGraph | CrewAI |
|---|---|---|
| **Learning curve** | Steeper — graph thinking required | Gentler — role/task thinking is intuitive |
| **Flexibility** | Very high — build anything | Medium — opinionated structure |
| **Boilerplate** | More — state, nodes, edges | Less — agents and tasks in plain English |
| **Debugging** | Excellent (LangSmith tracing) | Improving (verbose logs, CrewAI traces) |
| **Streaming** | First-class, granular | Basic (final output focused) |
| **Production maturity** | High | Growing rapidly |
| **Complex branching** | Natural | Requires Flows (added complexity) |
| **Multi-agent patterns** | Manual but precise | Built-in and natural |
| **Typed state** | Yes — enforced | No — free text between tasks |
| **Parallelism** | Built-in (Send API, fan-out) | Limited in sequential; more in Flows |

---

## The "Aha" Insight

After building 12 examples in each framework, the key realization is:

> **LangGraph and CrewAI solve the same problem at different abstraction levels.**

LangGraph is like writing SQL — you control the execution plan exactly.
CrewAI is like writing a query in plain English — it figures out the plan for you.

Neither is better. The question is: **how much control do you need vs how fast do you want to move?**

For **prototyping and role-based pipelines** → start with CrewAI.
For **production systems with complex logic** → LangGraph gives you the precision you'll eventually need.
For **the best of both** → use CrewAI Flows to orchestrate multiple CrewAI Crews, which gets you event-driven control without the full graph model.

---

## Quick Reference

```
Need typed state?                → LangGraph
Need a manager to delegate?     → CrewAI (hierarchical)
Need to pause for human review? → Both (different APIs)
Need precise routing logic?     → LangGraph
Need role-based collaboration?  → CrewAI
Need token-level streaming?     → LangGraph
Need long-term agent memory?    → CrewAI
Building reusable components?   → LangGraph (subgraphs)
Prototyping fast?               → CrewAI
Need to debug deeply?           → LangGraph (LangSmith)
```

---

*This comparison was built through hands-on experience — 12 LangGraph examples and 12 CrewAI examples, from basics to production pipelines. See [lang-graph/](lang-graph/) and [crew-ai/](crew-ai/) for the full code.*
