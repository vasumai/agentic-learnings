"""
MCP Server for Lesson 11 — Text Processing Tools
==================================================
This is a standalone MCP server that exposes three text-analysis tools
via the Model Context Protocol (MCP) over stdio.

Run by 11_mcp_integration.py as a subprocess — do NOT run directly
(unless you want to test it in isolation with an MCP client).

Tools exposed:
  - count_words(text)        : word count, char count
  - get_text_stats(text)     : sentences, paragraphs, avg word length
  - find_keywords(text, min_length) : top unique words by length
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="TextProcessor",
    instructions="Analyses text: word counts, statistics, and keyword extraction.",
)


@mcp.tool()
def count_words(text: str) -> dict:
    """Counts words and characters in a block of text.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "char_no_spaces": len(text.replace(" ", "")),
    }


@mcp.tool()
def get_text_stats(text: str) -> dict:
    """Returns detailed statistics about text: sentences, paragraphs, avg word length.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    # Split on sentence-ending punctuation
    raw_sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    avg_word_len = (
        round(sum(len(w) for w in words) / len(words), 1) if words else 0.0
    )
    return {
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": max(len(paragraphs), 1),
        "avg_word_length": avg_word_len,
    }


@mcp.tool()
def find_keywords(text: str, min_length: int = 5) -> list:
    """Finds the top 10 unique keywords (words longer than min_length), sorted longest-first.

    Args:
        text: The text to scan.
        min_length: Minimum character length for a word to count as a keyword.
    """
    # Strip punctuation, lowercase, deduplicate
    cleaned = set(
        w.strip(".,!?\"'()[]{}:;").lower()
        for w in text.split()
    )
    keywords = sorted(
        [w for w in cleaned if len(w) >= min_length],
        key=len,
        reverse=True,
    )
    return keywords[:10]


if __name__ == "__main__":
    mcp.run(transport="stdio")
