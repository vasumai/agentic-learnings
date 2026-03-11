"""
12_real_world_apis.py
---------------------
All previous examples used mocked data. This file uses REAL free APIs
with no API keys required (except OpenAI which you already have).

Free APIs used:
  - open-meteo.com          → live weather (temp, wind, humidity, condition)
  - air-quality-api.open-meteo.com → UV index, PM2.5, PM10
  - geocoding-api.open-meteo.com   → city name → lat/lon (used by weather+AQ)
  - restcountries.com       → country info (population, capital, currency)
  - en.wikipedia.org        → factual summaries of any topic
  - api.coingecko.com       → live crypto prices (BTC, ETH, etc.)

LLM: OpenAI GPT-4o-mini (cheaper, fast, great for tool calling)

Architecture: ReAct loop (same as file 10) but with live data.
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
import requests

load_dotenv()

# ── Shared HTTP session (connection reuse, timeouts) ─────────────────────────
session = requests.Session()
session.headers.update({"User-Agent": "agentic-learnings/1.0"})
TIMEOUT = 10


# ── Helper: city → (lat, lon, display_name) ──────────────────────────────────
def geocode(city: str) -> tuple[float, float, str]:
    """Convert city name to coordinates using Open-Meteo's free geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = session.get(url, params={"name": city, "count": 1, "language": "en"}, timeout=TIMEOUT)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}")
    r = results[0]
    return r["latitude"], r["longitude"], f"{r['name']}, {r.get('country', '')}"


# ── WMO weather code → human readable ────────────────────────────────────────
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
    85: "Snow showers", 95: "Thunderstorm", 99: "Thunderstorm with hail",
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — real API calls
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for any city worldwide.
    Returns temperature, feels-like, humidity, wind speed, and conditions.
    """
    try:
        lat, lon, display = geocode(city)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weathercode",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        resp = session.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        c = resp.json()["current"]
        condition = WMO_CODES.get(c["weathercode"], f"Code {c['weathercode']}")
        return (
            f"Weather in {display}:\n"
            f"  Condition:   {condition}\n"
            f"  Temperature: {c['temperature_2m']}°C (feels like {c['apparent_temperature']}°C)\n"
            f"  Humidity:    {c['relative_humidity_2m']}%\n"
            f"  Wind speed:  {c['wind_speed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather lookup failed for '{city}': {e}"


@tool
def get_air_quality(city: str) -> str:
    """
    Get air quality data for any city: UV index, PM2.5, PM10, and European AQI.
    UV index: 0-2 low, 3-5 moderate, 6-7 high, 8-10 very high, 11+ extreme.
    """
    try:
        lat, lon, display = geocode(city)
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "uv_index,pm2_5,pm10,european_aqi",
            "timezone": "auto",
        }
        resp = session.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        c = resp.json()["current"]

        uv = c.get("uv_index", 0)
        uv_risk = (
            "Low" if uv <= 2 else "Moderate" if uv <= 5 else
            "High" if uv <= 7 else "Very High" if uv <= 10 else "Extreme"
        )
        aqi = c.get("european_aqi", 0)
        aqi_label = (
            "Good" if aqi <= 20 else "Fair" if aqi <= 40 else
            "Moderate" if aqi <= 60 else "Poor" if aqi <= 80 else "Very Poor"
        )
        return (
            f"Air quality in {display}:\n"
            f"  UV Index:    {uv} ({uv_risk})\n"
            f"  PM2.5:       {c.get('pm2_5', 'N/A')} µg/m³\n"
            f"  PM10:        {c.get('pm10', 'N/A')} µg/m³\n"
            f"  European AQI:{aqi} ({aqi_label})"
        )
    except Exception as e:
        return f"Air quality lookup failed for '{city}': {e}"


@tool
def get_country_info(country: str) -> str:
    """
    Get information about a country: capital, population, currency,
    region, languages, and area.
    """
    try:
        url = f"https://restcountries.com/v3.1/name/{country}"
        resp = session.get(url, params={"fullText": "false"}, timeout=TIMEOUT)
        resp.raise_for_status()
        d = resp.json()[0]

        capital = d.get("capital", ["N/A"])[0]
        population = f"{d.get('population', 0):,}"
        area = f"{d.get('area', 0):,.0f} km²"
        region = f"{d.get('region', 'N/A')} / {d.get('subregion', 'N/A')}"
        currencies = ", ".join(
            f"{v['name']} ({v.get('symbol', '')})"
            for v in d.get("currencies", {}).values()
        )
        languages = ", ".join(d.get("languages", {}).values())
        return (
            f"Country: {d.get('name', {}).get('common', country)}\n"
            f"  Capital:    {capital}\n"
            f"  Population: {population}\n"
            f"  Area:       {area}\n"
            f"  Region:     {region}\n"
            f"  Currency:   {currencies}\n"
            f"  Languages:  {languages}"
        )
    except Exception as e:
        return f"Country info lookup failed for '{country}': {e}"


@tool
def search_wikipedia(topic: str) -> str:
    """
    Search Wikipedia for a factual summary of any topic, person, place, or concept.
    Returns the first 3 sentences of the Wikipedia article.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code == 404:
            # Try search API to find the right title
            search_resp = session.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
                timeout=TIMEOUT
            )
            titles = search_resp.json()[1]
            if not titles:
                return f"No Wikipedia article found for '{topic}'"
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{titles[0].replace(' ', '_')}"
            resp = session.get(url, timeout=TIMEOUT)

        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "No summary available.")
        # Return first 3 sentences
        sentences = extract.split(". ")
        summary = ". ".join(sentences[:3]) + ("." if len(sentences) > 1 else "")
        return f"Wikipedia — {data.get('title', topic)}:\n{summary}"
    except Exception as e:
        return f"Wikipedia search failed for '{topic}': {e}"


@tool
def get_crypto_price(coin: str) -> str:
    """
    Get the current price of a cryptocurrency in USD.
    Supported: bitcoin, ethereum, solana, cardano, dogecoin, ripple, polkadot, etc.
    """
    try:
        # Normalize common names
        aliases = {
            "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
            "ada": "cardano", "doge": "dogecoin", "xrp": "ripple",
            "dot": "polkadot", "bnb": "binancecoin",
        }
        coin_id = aliases.get(coin.lower(), coin.lower())
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        }
        resp = session.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if coin_id not in data:
            return f"Coin '{coin}' not found. Try: bitcoin, ethereum, solana, dogecoin"

        d = data[coin_id]
        price = d.get("usd", 0)
        change = d.get("usd_24h_change", 0)
        mcap = d.get("usd_market_cap", 0)
        arrow = "▲" if change >= 0 else "▼"
        return (
            f"{coin_id.capitalize()} price:\n"
            f"  Price:      ${price:,.2f} USD\n"
            f"  24h Change: {arrow} {abs(change):.2f}%\n"
            f"  Market Cap: ${mcap:,.0f} USD"
        )
    except Exception as e:
        return f"Crypto price lookup failed for '{coin}': {e}"


# ══════════════════════════════════════════════════════════════════════════════
# AGENT — ReAct loop with OpenAI
# ══════════════════════════════════════════════════════════════════════════════
tools = [get_weather, get_air_quality, get_country_info, search_wikipedia, get_crypto_price]
tool_by_name = {t.name: t for t in tools}

SYSTEM_PROMPT = """You are a helpful assistant with access to real-time data tools.
Use them to answer questions accurately with live data.
For weather questions use get_weather. For air quality or UV index use get_air_quality.
For country facts use get_country_info. For general knowledge use search_wikipedia.
For crypto prices use get_crypto_price. Combine tools freely when needed."""

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int


def reason(state: State) -> dict:
    iteration = state.get("iterations", 0) + 1
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        print(f"  → calling: {[tc['name'] for tc in tool_calls]}")
    return {"messages": [response], "iterations": iteration}


def act(state: State) -> dict:
    last = state["messages"][-1]
    tool_messages = []
    for tc in last.tool_calls:
        print(f"  → {tc['name']}({tc['args']})")
        try:
            result = tool_by_name[tc["name"]].invoke(tc["args"])
        except Exception as e:
            result = f"Error: {e}"
        tool_messages.append(ToolMessage(
            content=str(result), tool_call_id=tc["id"], name=tc["name"]
        ))
    return {"messages": tool_messages}


def should_continue(state: State) -> Literal["act", "__end__"]:
    last = state["messages"][-1]
    if state.get("iterations", 0) >= 8:
        return "__end__"
    if getattr(last, "tool_calls", []):
        return "act"
    return "__end__"


builder = StateGraph(State)
builder.add_node("reason", reason)
builder.add_node("act", act)
builder.add_edge(START, "reason")
builder.add_conditional_edges("reason", should_continue, {"act": "act", "__end__": END})
builder.add_edge("act", "reason")
graph = builder.compile()


def ask(question: str) -> str:
    print(f"\n{'─' * 60}")
    print(f"Q: {question}")
    result = graph.invoke({"messages": [HumanMessage(content=question)], "iterations": 0})
    for msg in reversed(result["messages"]):
        content = msg.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        if len(str(content).strip()) > 10:
            return str(content)
    return "[no response]"


if __name__ == "__main__":
    print("=" * 60)
    print("Real-World API Agent — Live Data, No Mocks")
    print("=" * 60)

    queries = [
        "What is the current weather in Singapore?",
        "What is the UV index and air quality in Delhi right now?",
        "Tell me about Japan — capital, population, currency.",
        "What is Bitcoin's current price and 24h change?",
        "Give me a brief Wikipedia summary of the James Webb Space Telescope.",
        "Compare the weather in Tokyo and London right now, and also tell me the current price of Ethereum.",
    ]

    for q in queries:
        answer = ask(q)
        print(f"A: {answer}\n")
