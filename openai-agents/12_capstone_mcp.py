"""
Companion MCP server for lesson 12 capstone.
Launched automatically by the capstone — do NOT run directly.

Exposes:
  - search_kb(query)                        → knowledge base search
  - create_ticket(customer_id, summary, priority) → create support ticket
  - get_customer_history(customer_id)       → past interactions
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SupportKB")

KB_ARTICLES = {
    "api key":        "API keys are managed in Settings → API. Rotate keys via the dashboard. Keys expire after 90 days.",
    "billing cycle":  "Billing cycles run on the 1st of each month. Invoices are emailed within 48 hours.",
    "refund policy":  "Refunds are available within 30 days of charge. Pro-rated refunds apply for plan downgrades.",
    "rate limit":     "Free: 60 req/min. Pro: 1000 req/min. Enterprise: unlimited. Limits reset every minute.",
    "webhook":        "Webhooks can be configured in Settings → Integrations. We support HTTPS endpoints only.",
    "mfa":            "MFA can be enabled in Account → Security. We support TOTP apps (Authy, Google Authenticator).",
    "data export":    "Export account data via Settings → Data. Exports include all logs, usage, and billing history.",
    "sla":            "Pro SLA: 99.9% uptime. Enterprise SLA: 99.99% uptime with dedicated support.",
}

TICKET_COUNTER = {"n": 1000}

CUSTOMER_HISTORY = {
    "C-001": ["2026-01-15: Billing dispute — resolved", "2026-02-20: API rate limit question — resolved"],
    "C-002": ["2026-03-01: Account locked — resolved"],
    "C-003": [],
}


@mcp.tool()
def search_kb(query: str) -> str:
    """Search the support knowledge base for articles matching a query.

    Args:
        query: The search query (e.g. 'refund policy', 'api rate limit').
    """
    query_lower = query.lower()
    matches = []
    for keyword, article in KB_ARTICLES.items():
        if any(word in query_lower for word in keyword.split()):
            matches.append(f"[{keyword}] {article}")
    if not matches:
        return f"No articles found for '{query}'. Try: {', '.join(KB_ARTICLES.keys())}"
    return "\n".join(matches)


@mcp.tool()
def create_ticket(customer_id: str, summary: str, priority: str) -> str:
    """Create a support ticket for a customer issue.

    Args:
        customer_id: The customer identifier.
        summary: Brief description of the issue.
        priority: One of: low, medium, high, urgent.
    """
    TICKET_COUNTER["n"] += 1
    ticket_id = f"TKT-{TICKET_COUNTER['n']}"
    return f"Ticket {ticket_id} created for {customer_id}. Priority: {priority}. Summary: {summary}"


@mcp.tool()
def get_customer_history(customer_id: str) -> str:
    """Get the support interaction history for a customer.

    Args:
        customer_id: The customer identifier (e.g. C-001).
    """
    history = CUSTOMER_HISTORY.get(customer_id, [])
    if not history:
        return f"No prior interactions found for {customer_id}."
    return f"History for {customer_id}:\n" + "\n".join(f"  - {h}" for h in history)


if __name__ == "__main__":
    mcp.run()
