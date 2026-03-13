"""
Companion MCP server for lesson 11.
Run by MCPServerStdio — do NOT run this directly.

Exposes three tools:
  - get_weather(city)          → mock weather data
  - list_notes()               → list all saved notes
  - add_note(title, content)   → save a note
  - get_note(title)            → retrieve a note by title
"""

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LessonServer")

# In-memory note store
_notes: dict[str, str] = {
    "welcome": "Welcome to the MCP lesson! This note was pre-loaded.",
}

WEATHER_DATA = {
    "london":    {"temp": "13°C", "condition": "Overcast",       "humidity": "82%"},
    "tokyo":     {"temp": "22°C", "condition": "Sunny",          "humidity": "55%"},
    "new york":  {"temp": "8°C",  "condition": "Windy",          "humidity": "60%"},
    "sydney":    {"temp": "27°C", "condition": "Partly cloudy",  "humidity": "65%"},
    "mumbai":    {"temp": "34°C", "condition": "Hot and humid",  "humidity": "88%"},
}


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name (e.g. London, Tokyo, New York).
    """
    data = WEATHER_DATA.get(city.lower())
    if not data:
        return f"No weather data for '{city}'. Available: {', '.join(WEATHER_DATA.keys())}"
    return f"{city}: {data['temp']}, {data['condition']}, humidity {data['humidity']}"


@mcp.tool()
def list_notes() -> str:
    """List all saved note titles."""
    if not _notes:
        return "No notes saved yet."
    return "Saved notes: " + ", ".join(f"'{t}'" for t in _notes.keys())


@mcp.tool()
def add_note(title: str, content: str) -> str:
    """Save a new note.

    Args:
        title: The note title (used as key).
        content: The note content.
    """
    _notes[title.lower()] = content
    return f"Note '{title}' saved successfully."


@mcp.tool()
def get_note(title: str) -> str:
    """Retrieve a note by title.

    Args:
        title: The title of the note to retrieve.
    """
    content = _notes.get(title.lower())
    if content is None:
        return f"No note found with title '{title}'."
    return f"Note '{title}':\n{content}"


if __name__ == "__main__":
    mcp.run()
