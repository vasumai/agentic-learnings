"""
12_real_world_pipeline.py — Real-World Tech Intelligence Pipeline (Capstone)

Every concept from lessons 01-11, combined into one production-grade system:
  ✓ Flow orchestration (@start, @listen, @router)        — lesson 10-11
  ✓ Multiple Crews as pipeline stages                    — lesson 11
  ✓ Custom tools with REAL free API calls                — lesson 06
  ✓ Structured Pydantic output                           — lesson 07
  ✓ Memory across runs                                   — lesson 08
  ✓ Task dependencies with context=[]                    — lesson 04
  ✓ Dynamic inputs via {placeholders}                    — lesson 02

What it builds:
  A Tech Intelligence Analyst that researches any company or technology,
  pulls LIVE data from Wikipedia, HackerNews, and GitHub (all free, no keys),
  produces a structured investment/adoption brief, and remembers past research.

Real APIs used (100% free, no API key required):
  - Wikipedia REST API    → company/tech summary
  - HackerNews Algolia   → latest community discussion & sentiment
  - GitHub Search API    → open-source ecosystem health (stars, repos, activity)

Run multiple times with different topics to watch memory build up:
  python 12_real_world_pipeline.py "Anthropic"
  python 12_real_world_pipeline.py "LangChain"
  python 12_real_world_pipeline.py "Rust programming language"
"""

import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, listen, start, router
from crewai.tools import BaseTool, tool
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# REAL API TOOLS — no mocks, no stubs, live data
# ══════════════════════════════════════════════════════════════════════════════

@tool("Wikipedia Summary")
def wikipedia_tool(topic: str) -> str:
    """
    Fetches a real-time summary of any topic from Wikipedia.
    Use this to get factual background on companies, technologies, or concepts.
    Input: the topic or company name to search.
    """
    try:
        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "TechIntelBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        title = data.get("title", topic)
        extract = data.get("extract", "No summary available.")
        return f"Wikipedia — {title}:\n{extract[:1500]}"
    except Exception as e:
        return f"Wikipedia lookup failed for '{topic}': {str(e)}"


@tool("HackerNews Pulse")
def hackernews_tool(query: str) -> str:
    """
    Searches HackerNews for the latest community discussions about a topic.
    Use this to gauge developer sentiment, adoption buzz, and recent news.
    Input: the technology or company name to search.
    Returns: top 5 recent stories with titles, scores, and comment counts.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://hn.algolia.com/api/v1/search?query={encoded}&tags=story&hitsPerPage=5"
        req = urllib.request.Request(url, headers={"User-Agent": "TechIntelBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        hits = data.get("hits", [])
        if not hits:
            return f"No HackerNews stories found for '{query}'"
        lines = [f"HackerNews top stories for '{query}':"]
        for i, h in enumerate(hits, 1):
            title = h.get("title", "No title")
            score = h.get("points", 0)
            comments = h.get("num_comments", 0)
            date = h.get("created_at", "")[:10]
            lines.append(f"  {i}. [{date}] {title} (score:{score}, comments:{comments})")
        return "\n".join(lines)
    except Exception as e:
        return f"HackerNews search failed for '{query}': {str(e)}"


@tool("GitHub Ecosystem")
def github_tool(query: str) -> str:
    """
    Searches GitHub for open-source repositories related to a topic.
    Use this to assess ecosystem health: how many projects, stars, and activity.
    Input: the technology, framework, or company name to search.
    Returns: top repositories with stars, forks, and last update.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&per_page=5"
        req = urllib.request.Request(url, headers={
            "User-Agent": "TechIntelBot/1.0",
            "Accept": "application/vnd.github.v3+json"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        repos = data.get("items", [])
        total = data.get("total_count", 0)
        if not repos:
            return f"No GitHub repos found for '{query}'"
        lines = [f"GitHub ecosystem for '{query}' ({total:,} total repos):"]
        for r in repos:
            name = r.get("full_name", "")
            stars = r.get("stargazers_count", 0)
            forks = r.get("forks_count", 0)
            updated = r.get("updated_at", "")[:10]
            desc = (r.get("description") or "")[:80]
            lines.append(f"  ★{stars:,}  {name}  [{updated}]  {desc}")
        return "\n".join(lines)
    except Exception as e:
        return f"GitHub search failed for '{query}': {str(e)}"


@tool("Get Report Timestamp")
def timestamp_tool(unused: str = "") -> str:
    """Returns the current date and time. Use when timestamping reports."""
    return datetime.now().strftime("%B %d, %Y at %H:%M UTC")


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC OUTPUT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TechIntelBrief(BaseModel):
    """Structured intelligence brief for a technology or company."""
    subject: str = Field(description="The company or technology being analyzed")
    category: str = Field(description="Category: AI/ML, DevTools, Cloud, Language, Framework, etc.")
    one_liner: str = Field(description="What it is in one sentence")
    maturity: str = Field(description="Maturity stage: Emerging / Growing / Mature / Declining")
    community_sentiment: str = Field(description="HackerNews/developer sentiment: Positive / Neutral / Mixed / Negative")
    ecosystem_strength: str = Field(description="GitHub ecosystem: Strong / Moderate / Weak")
    top_strengths: List[str] = Field(description="Top 3 strengths")
    top_risks: List[str] = Field(description="Top 2 risks or concerns")
    adoption_recommendation: str = Field(description="Adopt / Evaluate / Hold / Avoid")
    confidence: int = Field(description="Analyst confidence score 1-10")
    summary: str = Field(description="Executive summary in 3 sentences")


# ══════════════════════════════════════════════════════════════════════════════
# FLOW STATE
# ══════════════════════════════════════════════════════════════════════════════

class IntelState(BaseModel):
    subject: str = ""
    raw_research: str = ""
    structured_brief: Optional[TechIntelBrief] = None
    final_report: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# FLOW — three stages, each a focused Crew
# ══════════════════════════════════════════════════════════════════════════════

class TechIntelFlow(Flow[IntelState]):

    # ── Stage 1: Data Gathering Crew — hits all three real APIs ──────────────
    @start()
    def gather_data(self):
        print(f"\n[FLOW] Stage 1: Gathering live data on '{self.state.subject}'...")

        scout = Agent(
            role="Tech Intelligence Scout",
            goal=f"Gather comprehensive live data about {self.state.subject} from all available sources",
            backstory=(
                "You are a relentless data gatherer who never makes assumptions. "
                "You use every tool available to pull real, live information. "
                "You always search Wikipedia, HackerNews, AND GitHub before summarizing."
            ),
            tools=[wikipedia_tool, hackernews_tool, github_tool, timestamp_tool],
            verbose=True,
        )

        gather_task = Task(
            description=(
                f"Research '{self.state.subject}' using ALL available tools.\n\n"
                "You MUST use all three data tools:\n"
                f"1. Wikipedia: get factual background on '{self.state.subject}'\n"
                f"2. HackerNews: get developer community sentiment on '{self.state.subject}'\n"
                f"3. GitHub: assess open-source ecosystem for '{self.state.subject}'\n"
                "4. Get the current timestamp for the report\n\n"
                "Compile everything into a raw research dossier — facts only, no editorializing."
            ),
            expected_output=(
                "A raw research dossier with four sections:\n"
                "BACKGROUND (from Wikipedia), COMMUNITY PULSE (from HackerNews), "
                "ECOSYSTEM (from GitHub), TIMESTAMP."
            ),
            agent=scout,
        )

        result = Crew(
            agents=[scout],
            tasks=[gather_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        self.state.raw_research = str(result)
        print(f"[FLOW] Data gathered — {len(self.state.raw_research)} chars of raw intelligence")

    # ── Stage 2: Analysis Crew — structured Pydantic output ──────────────────
    @listen(gather_data)
    def analyze_and_structure(self):
        print(f"\n[FLOW] Stage 2: Analyzing and structuring intelligence...")

        analyst = Agent(
            role="Senior Technology Analyst",
            goal=f"Produce a precise, structured intelligence brief on {self.state.subject}",
            backstory=(
                "You are a senior analyst at a top technology research firm. "
                "You read raw data and extract signal from noise. "
                "You are opinionated — you give clear recommendations, not wishy-washy hedging."
            ),
            verbose=True,
        )

        analysis_task = Task(
            description=(
                f"Analyze this raw research dossier on '{self.state.subject}':\n\n"
                f"{self.state.raw_research}\n\n"
                "Produce a complete structured intelligence brief. "
                "Be decisive — pick clear values for every field. "
                "adoption_recommendation must be exactly one of: Adopt / Evaluate / Hold / Avoid\n"
                "maturity must be exactly: Emerging / Growing / Mature / Declining\n"
                "community_sentiment: Positive / Neutral / Mixed / Negative\n"
                "ecosystem_strength: Strong / Moderate / Weak"
            ),
            expected_output=(
                "A complete intelligence brief with all fields populated precisely. "
                "Confidence score reflects how much real data supported the analysis."
            ),
            agent=analyst,
            output_pydantic=TechIntelBrief,
        )

        result = Crew(
            agents=[analyst],
            tasks=[analysis_task],
            process=Process.sequential,
            memory=True,   # agent remembers past analyses — gets smarter each run
            verbose=False,
        ).kickoff()

        self.state.structured_brief = analysis_task.output.pydantic
        print(f"[FLOW] Analysis complete — Recommendation: {self.state.structured_brief.adoption_recommendation}")

    # ── Stage 3: Report Crew — narrative final deliverable ───────────────────
    @listen(analyze_and_structure)
    def write_final_report(self):
        print(f"\n[FLOW] Stage 3: Writing final intelligence report...")

        brief = self.state.structured_brief

        writer = Agent(
            role="Intelligence Report Writer",
            goal="Transform structured analysis into a polished, readable intelligence report",
            backstory=(
                "You write intelligence reports for CTOs and engineering leaders. "
                "Your reports are crisp, opinionated, and immediately actionable. "
                "You never bury the lede — verdict goes first, detail follows."
            ),
            verbose=True,
        )

        report_task = Task(
            description=(
                f"Write a polished intelligence report for '{brief.subject if brief else self.state.subject}'.\n\n"
                f"Structured brief data:\n"
                f"  Category: {brief.category if brief else 'N/A'}\n"
                f"  One-liner: {brief.one_liner if brief else 'N/A'}\n"
                f"  Maturity: {brief.maturity if brief else 'N/A'}\n"
                f"  Community: {brief.community_sentiment if brief else 'N/A'}\n"
                f"  Ecosystem: {brief.ecosystem_strength if brief else 'N/A'}\n"
                f"  Strengths: {brief.top_strengths if brief else []}\n"
                f"  Risks: {brief.top_risks if brief else []}\n"
                f"  Recommendation: {brief.adoption_recommendation if brief else 'N/A'}\n"
                f"  Confidence: {brief.confidence if brief else 'N/A'}/10\n"
                f"  Summary: {brief.summary if brief else 'N/A'}\n\n"
                "Write a professional markdown report with:\n"
                "1. Title with subject and date\n"
                "2. VERDICT box (recommendation + confidence) — lead with this\n"
                "3. What It Is (2-3 sentences)\n"
                "4. Why It Matters (strengths as bullets)\n"
                "5. Watch Out For (risks as bullets)\n"
                "6. Data Signals (maturity, community, ecosystem as a mini-table)\n"
                "7. Bottom Line (1 punchy paragraph)\n"
            ),
            expected_output=(
                "A professional markdown intelligence report, verdict-first, "
                "crisp and actionable. Suitable for a CTO briefing."
            ),
            agent=writer,
            output_file="intel_report.md",
        )

        result = Crew(
            agents=[writer],
            tasks=[report_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        self.state.final_report = str(result)
        print("[FLOW] Final report written")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

subject = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "CrewAI"

print("\n" + "=" * 60)
print("TECH INTELLIGENCE PIPELINE — CAPSTONE")
print("=" * 60)
print(f"\nSubject:  {subject}")
print("Live APIs: Wikipedia + HackerNews + GitHub")
print("Output:    Structured brief + narrative report + memory")
print("\nPipeline:")
print("  [gather_data]           → 3 real API calls")
print("  [analyze_and_structure] → Pydantic TechIntelBrief")
print("  [write_final_report]    → markdown report saved to disk")
print()

flow = TechIntelFlow()
flow.kickoff(inputs={"subject": subject})

# ── Print structured brief fields ────────────────────────────────────────────
brief = flow.state.structured_brief
if brief:
    print("\n" + "=" * 60)
    print("STRUCTURED BRIEF (Pydantic fields)")
    print("=" * 60)
    print(f"  Subject:         {brief.subject}")
    print(f"  Category:        {brief.category}")
    print(f"  One-liner:       {brief.one_liner}")
    print(f"  Maturity:        {brief.maturity}")
    print(f"  Community:       {brief.community_sentiment}")
    print(f"  Ecosystem:       {brief.ecosystem_strength}")
    print(f"  Strengths:       {' | '.join(brief.top_strengths)}")
    print(f"  Risks:           {' | '.join(brief.top_risks)}")
    print(f"  Recommendation:  {brief.adoption_recommendation}")
    print(f"  Confidence:      {brief.confidence}/10")

print("\n" + "=" * 60)
print("FINAL REPORT (excerpt)")
print("=" * 60)
print(flow.state.final_report[:1500])
print("\n→ Full report saved to intel_report.md")
print("\nTip: Run again with a different subject to build memory:")
print("  python 12_real_world_pipeline.py 'LangChain'")
print("  python 12_real_world_pipeline.py 'OpenAI'")
print("  python 12_real_world_pipeline.py 'Rust programming language'")
