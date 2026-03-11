"""
09_human_in_the_loop.py — Pausing for Human Input Mid-Execution

Concepts covered:
- human_input=True on a Task: pauses the crew and waits for your input
- How CrewAI passes agent output to you and resumes with your feedback
- Practical pattern: agent drafts → human reviews → agent refines
- Multiple human gates in a single pipeline
- When to use human-in-the-loop vs full automation

Real-world analogy:
  An AI drafts a contract clause. Before it goes to the next agent for
  legal review, YOU read it and say "too aggressive — soften the tone."
  The agent incorporates your feedback and continues. This is HITL.

LangGraph comparison:
  LangGraph: interrupt_before=["node_name"] pauses at a specific node.
             You resume with Command(resume=value) and pass data back.
  CrewAI:    human_input=True on a Task — agent completes the task,
             shows you the result, waits for your text feedback,
             then incorporates it into the next iteration automatically.
"""

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# ── Agents ────────────────────────────────────────────────────────────────────

copywriter = Agent(
    role="Senior Copywriter",
    goal="Write compelling marketing copy for {product} that converts",
    backstory=(
        "You are a seasoned copywriter with 15 years of experience writing "
        "for SaaS products. You write punchy, benefit-driven copy that speaks "
        "directly to the target audience's pain points. "
        "You take feedback seriously and always incorporate it fully."
    ),
    verbose=True,
)

editor = Agent(
    role="Brand Voice Editor",
    goal="Ensure all copy for {product} matches brand guidelines and is publish-ready",
    backstory=(
        "You are a meticulous brand editor who checks copy for consistency, "
        "tone, grammar, and alignment with brand voice. "
        "You only approve copy that is truly ready to publish."
    ),
    verbose=True,
)

# ── Tasks with human_input=True ───────────────────────────────────────────────
# When human_input=True:
#   1. Agent completes the task normally
#   2. CrewAI prints the result and prompts: "Provide feedback (or press Enter to accept):"
#   3. If you type feedback → agent receives it and revises
#   4. If you press Enter → output accepted as-is, crew continues
#
# This gives you a review gate at any point in the pipeline.

draft_task = Task(
    description=(
        "Write a landing page hero section for: {product}\n\n"
        "Include:\n"
        "- A headline (max 8 words, benefit-focused)\n"
        "- A subheadline (1-2 sentences, addresses the key pain point)\n"
        "- 3 bullet points of key benefits\n"
        "- A CTA button label (3-5 words)\n\n"
        "Target audience: {target_audience}\n"
        "Tone: {tone}"
    ),
    expected_output=(
        "A complete hero section with: Headline, Subheadline, "
        "3 Benefit Bullets, and CTA label. Ready for human review."
    ),
    agent=copywriter,
    human_input=True,   # ← PAUSE HERE: you review the draft and give feedback
)

# This task only runs after you've approved (or given feedback on) the draft
refinement_task = Task(
    description=(
        "Take the approved hero copy and produce the complete above-the-fold "
        "landing page section for {product}.\n\n"
        "Expand into:\n"
        "- Final polished headline and subheadline\n"
        "- 3 benefit bullets with icons suggested (e.g. ⚡ Speed: ...)\n"
        "- CTA button with surrounding urgency text (e.g. 'Join 10,000+ teams')\n"
        "- One social proof line (testimonial format)\n\n"
        "Incorporate all human feedback from the draft review."
    ),
    expected_output=(
        "A publish-ready above-the-fold landing page section in markdown. "
        "All feedback from the human review must be fully incorporated."
    ),
    agent=copywriter,
    context=[draft_task],
)

editorial_review_task = Task(
    description=(
        "Perform a final editorial review of the landing page copy for {product}.\n\n"
        "Check:\n"
        "1. Tone matches: {tone}\n"
        "2. Target audience fit: {target_audience}\n"
        "3. No grammar or spelling errors\n"
        "4. CTA is compelling and action-oriented\n"
        "5. Headline follows the 8-word limit\n\n"
        "If anything fails a check, rewrite that element. "
        "Provide a final APPROVED or NEEDS REVISION verdict."
    ),
    expected_output=(
        "Editorial report with: Pass/Fail for each check, "
        "any rewrites made, and final verdict (APPROVED or NEEDS REVISION)."
    ),
    agent=editor,
    context=[refinement_task],
    human_input=True,   # ← SECOND GATE: you review the final before it's 'published'
    output_file="landing_page_copy.md",
)

# ── Crew ──────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[copywriter, editor],
    tasks=[draft_task, refinement_task, editorial_review_task],
    process=Process.sequential,
    verbose=True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HUMAN-IN-THE-LOOP COPYWRITING PIPELINE")
print("=" * 60)
print("\nThis pipeline has TWO human review gates:")
print("  Gate 1: After the initial draft (you give feedback or press Enter)")
print("  Gate 2: After editorial review (final approval before 'publishing')")
print("\nTips for feedback at Gate 1:")
print("  - 'Make the headline more urgent'")
print("  - 'Too formal — use casual language'")
print("  - 'Focus more on time-saving, less on features'")
print("  - Press Enter with no input to accept as-is\n")
print("=" * 60 + "\n")

result = crew.kickoff(inputs={
    "product": "TaskFlow — an AI-powered project management tool for remote teams",
    "target_audience": "Engineering managers at startups with 10-50 person teams",
    "tone": "Professional but approachable, confident, no corporate jargon",
})

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — FINAL APPROVED COPY")
print("=" * 60)
print(result)
print("\n→ Final copy saved to landing_page_copy.md")
