"""
03_first_crew.py — Two Agents, Two Tasks, One Crew

Concepts covered:
- Multi-agent collaboration
- How task output flows automatically to the next task as context
- Agents with different roles working on the same goal
- The "assembly line" mental model of sequential process

Real-world analogy:
  Researcher gathers raw info → Writer turns it into polished content
  This is the most common CrewAI pattern you'll see in the wild.
"""

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# ── 1. Define TWO Agents with distinct roles ─────────────────────────────────
# Each agent has a clear specialty. The backstory shapes HOW they approach work.

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover deep insights about {topic} with facts and examples",
    backstory=(
        "You are a meticulous researcher who digs beneath the surface. "
        "You organize findings into clear, structured bullet points. "
        "You always separate facts from opinions."
    ),
    verbose=True,
)

writer = Agent(
    role="Content Strategist",
    goal="Transform research findings into engaging, readable content about {topic}",
    backstory=(
        "You are an experienced tech writer who takes dense research and turns it "
        "into content that a smart non-expert can enjoy. "
        "Your writing is warm, direct, and never uses jargon without explanation."
    ),
    verbose=True,
)

# ── 2. Define TWO Tasks assigned to each agent ───────────────────────────────
# KEY INSIGHT: In sequential process, task 2 automatically receives task 1's
# output as additional context. You don't wire this manually — CrewAI does it.

research_task = Task(
    description=(
        "Research the topic: {topic}\n\n"
        "Produce a structured set of findings:\n"
        "- What it is (2-3 sentences)\n"
        "- Why it matters (3 bullet points)\n"
        "- Real-world examples (2-3 concrete cases)\n"
        "- Common misconceptions (1-2 points)\n"
        "- Future outlook (1 paragraph)"
    ),
    expected_output=(
        "A structured research brief with sections: Definition, Why It Matters, "
        "Real-World Examples, Misconceptions, and Future Outlook. "
        "Bullet points preferred. No fluff."
    ),
    agent=researcher,
)

writing_task = Task(
    description=(
        "Using the research findings provided, write a compelling blog post about {topic}.\n\n"
        "Requirements:\n"
        "- Catchy title\n"
        "- Short intro hook (2-3 sentences)\n"
        "- 3-4 sections with subheadings\n"
        "- Concrete examples woven into the narrative\n"
        "- Punchy conclusion with a call to action\n"
        "- Tone: professional but conversational\n"
        "- Length: ~400-500 words"
    ),
    expected_output=(
        "A complete, publish-ready blog post in markdown format. "
        "Should feel human-written, not like a listicle. "
        "Includes title, intro, body sections, and conclusion."
    ),
    agent=writer,
    output_file="blog_post.md",
)

# ── 3. Crew with both agents and tasks ───────────────────────────────────────
# Order of tasks list = order of execution.
# research_task runs first → its output is passed to writing_task automatically.

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],  # order matters!
    process=Process.sequential,
    verbose=True,
)

# ── 4. Kickoff ───────────────────────────────────────────────────────────────
topic = "AI Agents in software development — how they are changing the way we build software"

print("\n" + "=" * 60)
print(f"CREW STARTING — Topic: {topic}")
print("=" * 60 + "\n")

result = crew.kickoff(inputs={"topic": topic})

print("\n" + "=" * 60)
print("FINAL OUTPUT (Writer's blog post)")
print("=" * 60)
print(result)
print("\n→ Full blog post saved to blog_post.md")
