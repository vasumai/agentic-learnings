"""
Lesson 05 — Structured Output
==============================
Covers:
  - output_type=PydanticModel on Agent — forces the LLM to return JSON
  - result.final_output_as(Model) — get typed output from RunResult
  - Nested Pydantic models as output
  - Structured output combined with tools
  - Using structured output for data extraction

Key concept:
  By default, Agent.final_output is a plain string.
  Set output_type=SomePydanticModel and the SDK instructs the LLM
  to respond in valid JSON matching your model's schema.
  result.final_output becomes a validated instance of that model.

  This is how you turn a conversational agent into a reliable
  data pipeline — extract, classify, score, transform — all typed.
"""

import asyncio
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, RunContextWrapper

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Simple structured output — sentiment classification
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0–1")
    reasoning: str = Field(description="One sentence explaining the classification")


async def sentiment_demo():
    print("=" * 50)
    print("PART 1: Sentiment classification")
    print("=" * 50)

    agent = Agent(
        name="SentimentAnalyzer",
        instructions=(
            "Analyze the sentiment of the given text. "
            "Return your classification, confidence, and a one-sentence reasoning."
        ),
        model="gpt-4o-mini",
        output_type=SentimentResult,
    )

    reviews = [
        "This product exceeded all my expectations. Absolutely love it!",
        "Arrived broken. Customer support was unhelpful. Total waste of money.",
        "It's okay. Does what it says, nothing special.",
    ]

    for review in reviews:
        print(f"\nText   : {review}")
        result = await Runner.run(agent, input=review)
        output: SentimentResult = result.final_output
        print(f"  sentiment  : {output.sentiment}")
        print(f"  confidence : {output.confidence:.2f}")
        print(f"  reasoning  : {output.reasoning}")


# ---------------------------------------------------------------------------
# 2. Nested models — structured data extraction
# ---------------------------------------------------------------------------

class ContactInfo(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    company: str | None = None


class SupportTicket(BaseModel):
    title: str = Field(description="Short summary of the issue (max 10 words)")
    category: Literal["billing", "technical", "account", "general"]
    priority: Literal["low", "medium", "high", "urgent"]
    contact: ContactInfo
    description: str = Field(description="Clean restatement of the issue")
    suggested_action: str = Field(description="Recommended next step for support team")


async def extraction_demo():
    print("\n" + "=" * 50)
    print("PART 2: Nested model — support ticket extraction")
    print("=" * 50)

    agent = Agent(
        name="TicketExtractor",
        instructions=(
            "Extract a structured support ticket from the user's message. "
            "Infer priority from urgency cues in the text."
        ),
        model="gpt-4o-mini",
        output_type=SupportTicket,
    )

    messages = [
        (
            "Hi, I'm Sarah from Acme Corp (sarah@acme.com, 555-1234). "
            "Our entire team has been locked out of the dashboard since this morning. "
            "We have a client demo in 2 hours and this is absolutely critical!"
        ),
        (
            "Hey, I think I was billed twice in March. My name is James, "
            "email james@example.com. Not super urgent, just wanted to flag it."
        ),
    ]

    for msg in messages:
        print(f"\nRaw message: {msg[:80]}...")
        result = await Runner.run(agent, input=msg)
        ticket: SupportTicket = result.final_output
        print(f"  title     : {ticket.title}")
        print(f"  category  : {ticket.category}")
        print(f"  priority  : {ticket.priority}")
        print(f"  contact   : {ticket.contact.name} <{ticket.contact.email}>")
        print(f"  action    : {ticket.suggested_action}")


# ---------------------------------------------------------------------------
# 3. Structured output + tools — agent uses tools, then returns typed result
# ---------------------------------------------------------------------------

class ProductRecommendation(BaseModel):
    product_name: str
    reason: str
    price_range: str
    rating: float = Field(ge=1.0, le=5.0)


class RecommendationList(BaseModel):
    query: str
    recommendations: list[ProductRecommendation]
    disclaimer: str


MOCK_CATALOG = {
    "laptop": [
        {"name": "ProBook 15", "price": "$999–$1,299", "rating": 4.6, "tags": ["portable", "fast"]},
        {"name": "UltraSlim X1", "price": "$799–$999", "rating": 4.3, "tags": ["thin", "battery"]},
    ],
    "headphones": [
        {"name": "SoundMax Pro", "price": "$199–$299", "rating": 4.7, "tags": ["noise-cancel", "wireless"]},
        {"name": "BassBoom 3", "price": "$79–$129", "rating": 4.1, "tags": ["bass", "wired"]},
    ],
}


@function_tool
def search_catalog(category: str) -> str:
    """Search the product catalog for a given category.

    Args:
        category: Product category to search (e.g. 'laptop', 'headphones').
    """
    category = category.lower()
    products = MOCK_CATALOG.get(category)
    if not products:
        return f"No products found for category '{category}'."
    lines = []
    for p in products:
        lines.append(f"- {p['name']} | {p['price']} | rating {p['rating']} | tags: {p['tags']}")
    return "\n".join(lines)


async def tools_plus_structured_output_demo():
    print("\n" + "=" * 50)
    print("PART 3: Tools + structured output")
    print("=" * 50)

    agent = Agent(
        name="ShopAdvisor",
        instructions=(
            "You are a shopping advisor. Use the search_catalog tool to find products, "
            "then return a structured list of recommendations based on the user's needs."
        ),
        model="gpt-4o-mini",
        tools=[search_catalog],
        output_type=RecommendationList,
    )

    query = "I need a laptop for travel — lightweight and good battery life."
    print(f"User: {query}")
    result = await Runner.run(agent, input=query)
    recs: RecommendationList = result.final_output

    print(f"\nQuery: {recs.query}")
    for i, r in enumerate(recs.recommendations, 1):
        print(f"\n  {i}. {r.product_name} ({r.price_range}) ★{r.rating}")
        print(f"     {r.reason}")
    print(f"\nDisclaimer: {recs.disclaimer}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await sentiment_demo()
    await extraction_demo()
    await tools_plus_structured_output_demo()


if __name__ == "__main__":
    asyncio.run(main())
