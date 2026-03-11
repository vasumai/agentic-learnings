"""
01_hello_agent.py — Your first CrewAI Agent

Concepts covered:
- Agent: the "person" with a role, goal, and backstory
- Task: a unit of work with a description and expected output
- Crew: the orchestrator that runs agents + tasks
- Sequential process: tasks run one after another
"""

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# ── 1. Define an Agent ──────────────────────────────────────────────────────
# Think of this as hiring a person. You give them:
#   - role:      their job title
#   - goal:      what they are trying to achieve
#   - backstory: their personality / expertise context (guides LLM behavior)
#   - verbose:   print what the agent is thinking/doing

researcher = Agent(
    role="Tech Researcher",
    goal="Find clear, accurate information about a given technology topic",
    backstory=(
        "You are a senior technology researcher with 10 years of experience "
        "breaking down complex topics into simple, digestible explanations. "
        "You love clarity and always back your statements with reasoning."
    ),
    verbose=True,
)

# ── 2. Define a Task ────────────────────────────────────────────────────────
# A Task is the actual work item. It needs:
#   - description:     what to do (the prompt / instructions)
#   - expected_output: what a good result looks like (guides the agent)
#   - agent:           which agent should handle this task

research_task = Task(
    description=(
        "Research the topic: 'What is CrewAI and how does it differ from LangGraph?' "
        "Cover: core concepts, key differences, and when to use each."
    ),
    expected_output=(
        "A clear 3-paragraph summary covering: "
        "1) What CrewAI is, "
        "2) What LangGraph is, "
        "3) Key differences and when to choose one over the other."
    ),
    agent=researcher,
)

# ── 3. Assemble the Crew ────────────────────────────────────────────────────
# A Crew ties agents and tasks together.
#   - agents:   list of all agents (even if assigned in tasks, list them here)
#   - tasks:    list of tasks in execution order
#   - process:  Process.sequential = run tasks one by one (default)
#               Process.hierarchical = manager delegates (covered later)
#   - verbose:  show crew-level logs

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True,
)

# ── 4. Kick it off ──────────────────────────────────────────────────────────
# crew.kickoff() runs the crew and returns the final output.
# inputs={} lets you pass variables into task descriptions using {placeholders}
# (we'll use that in the next lesson — for now no dynamic inputs needed)

print("\n" + "=" * 60)
print("CREW STARTING")
print("=" * 60 + "\n")

result = crew.kickoff()

print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(result)
