"""
06_custom_tools.py — Building Custom Tools with @tool

Concepts covered:
- @tool decorator: turn any Python function into an agent-usable tool
- How CrewAI describes tools to the LLM (docstring = tool description)
- BaseTool class: for tools that need state or config (OOP style)
- Agents autonomously decide WHEN and HOW to call your tools
- Combining custom tools with built-in tools

Real-world analogy:
  In lesson 02 we gave agents a calculator they didn't build.
  In this lesson WE build the calculator — and anything else we want.
  Your tool = your business logic that agents can now call.
"""

import json
import urllib.request
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool, tool
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# WAY 1: @tool decorator — simplest, for standalone functions
# The docstring is CRITICAL — it's what the LLM reads to decide when to use it
# ══════════════════════════════════════════════════════════════════════════════

@tool("Get Current DateTime")
def get_datetime_tool(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time.
    Use this when the user asks about the current date, time, or needs
    a timestamp for a report. Accepts an optional format string.
    """
    return datetime.now().strftime(format)


@tool("Calculate")
def calculator_tool(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this for any arithmetic, percentage calculations, or financial math.
    Input should be a valid Python math expression like '150 * 0.15' or '(500 + 300) / 4'.
    Do NOT use for complex code — only simple math expressions.
    """
    try:
        # Safe eval: only allow math operations, no builtins
        allowed = {k: v for k, v in __builtins__.items()
                   if k in ('abs', 'round', 'min', 'max', 'sum')} if isinstance(__builtins__, dict) else {}
        result = eval(expression, {"__builtins__": allowed}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool("Get Crypto Price")
def crypto_price_tool(coin_id: str) -> str:
    """
    Fetches the current price of a cryptocurrency in USD using the free CoinGecko API.
    Use this when asked about crypto prices or market values.
    Input should be the coin ID (e.g., 'bitcoin', 'ethereum', 'solana').
    No API key required.
    """
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        if coin_id not in data:
            return f"Coin '{coin_id}' not found. Try 'bitcoin', 'ethereum', or 'solana'."
        price = data[coin_id]['usd']
        change = data[coin_id].get('usd_24h_change', 0)
        direction = "▲" if change >= 0 else "▼"
        return f"{coin_id.capitalize()}: ${price:,.2f} USD  {direction} {abs(change):.2f}% (24h)"
    except Exception as e:
        return f"Error fetching price for '{coin_id}': {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# WAY 2: BaseTool class — for tools that need config, state, or complex logic
# Inherit from BaseTool, set name + description, implement _run()
# ══════════════════════════════════════════════════════════════════════════════

class WordCountTool(BaseTool):
    name: str = "Word Counter"
    description: str = (
        "Counts words, characters, sentences, and reading time for a given text. "
        "Use this when asked to analyze text length or estimate reading time. "
        "Input should be the text to analyze."
    )

    def _run(self, text: str) -> str:
        words = len(text.split())
        chars = len(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        reading_time = max(1, round(words / 200))  # avg 200 wpm
        return (
            f"Text Analysis:\n"
            f"  Words: {words}\n"
            f"  Characters: {chars}\n"
            f"  Sentences: {sentences}\n"
            f"  Estimated reading time: {reading_time} min"
        )


# ── Instantiate class-based tools ────────────────────────────────────────────
word_counter = WordCountTool()

# ── 1. Agent with ALL custom tools ───────────────────────────────────────────
analyst = Agent(
    role="Financial & Content Analyst",
    goal="Answer questions using real data from tools — never guess numbers",
    backstory=(
        "You are a precise analyst who always uses available tools to get real data. "
        "You never make up numbers or prices. If you have a tool for something, use it. "
        "You present findings clearly with context and calculations shown."
    ),
    tools=[
        get_datetime_tool,
        calculator_tool,
        crypto_price_tool,
        word_counter,
    ],
    verbose=True,
)

# ── 2. Task that forces the agent to use multiple tools ───────────────────────
analysis_task = Task(
    description=(
        "Produce a mini financial snapshot report. Do ALL of the following:\n\n"
        "1. Get the current date and time (use the datetime tool)\n"
        "2. Fetch current prices for bitcoin and ethereum (use crypto tool)\n"
        "3. Calculate: if someone bought 0.5 BTC and 2 ETH at today's prices, "
        "   what is the total portfolio value in USD? (use calculator tool)\n"
        "4. Write a 2-paragraph summary of your findings\n"
        "5. Use the word counter tool on your summary to report its stats"
    ),
    expected_output=(
        "A mini report with: timestamp, BTC price, ETH price, "
        "portfolio calculation (showing the math), 2-paragraph summary, "
        "and word count stats of the summary."
    ),
    agent=analyst,
    output_file="crypto_snapshot.md",
)

# ── 3. Crew ───────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True,
)

# ── 4. Run ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CUSTOM TOOLS DEMO")
print("Agent has: datetime, calculator, crypto prices, word counter")
print("=" * 60 + "\n")

result = crew.kickoff()

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(result)
print("\n→ Saved to crypto_snapshot.md")
