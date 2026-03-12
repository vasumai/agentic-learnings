# API Keys Reference

All API keys used across this learning repository — where to get them,
where they live, and whether to use free or paid tier.

---

## Anthropic API Key (`ANTHROPIC_API_KEY`)

**Used by:** LangGraph, CrewAI, Semantic Kernel (all lessons)

| | |
|---|---|
| **Get it at** | https://console.anthropic.com/settings/keys |
| **Model used** | `claude-sonnet-4-6` (via `anthropic/claude-sonnet-4-6`) |
| **Free tier** | Yes — limited monthly credits for new accounts |
| **Paid tier** | Pay-as-you-go after free credits exhaust. ~$3/MTok input, $15/MTok output for Sonnet |
| **Recommendation** | Add a credit card. Free credits run out quickly during active learning. $5–10 covers all 36 lessons comfortably. |
| **Set in `.env`** | `ANTHROPIC_API_KEY=sk-ant-...` |
| **Frameworks** | `lang-graph/.env`, `crew-ai/.env`, `semantic-kernel/.env` |

---

## Google API Key (`GOOGLE_API_KEY`)

**Used by:** Google ADK (all lessons)

| | |
|---|---|
| **Get it at** | https://aistudio.google.com/app/apikey |
| **Model used** | `gemini-2.5-flash` |
| **Free tier** | Yes — but **hard cap of 20 requests/day**. You will hit this in a single session. |
| **Paid tier** | Enable billing on your Google Cloud project at https://console.cloud.google.com/billing — the key itself stays the same, limits lift immediately |
| **Recommendation** | **Enable billing before starting Google ADK lessons.** The free cap is too low for active development. Cost is negligible — a full 12-lesson run costs under $1. |
| **Set in `.env`** | `GOOGLE_API_KEY=AIza...` |
| **Also required** | `GOOGLE_GENAI_USE_VERTEXAI=FALSE` (tells ADK to use AI Studio, not Vertex AI) |
| **Framework** | `google-adk/.env` |

### How to enable billing (one-time, 2 minutes)

1. Go to https://console.cloud.google.com/billing
2. Create or select a billing account and link it to your project
3. Return to https://aistudio.google.com — your existing API key now has higher limits
4. No code changes needed — the `.env` key stays the same

---

## OpenAI API Key (`OPENAI_API_KEY`)

**Used by:** LangGraph lesson 12, Semantic Kernel lesson 09 and 12

| | |
|---|---|
| **Get it at** | https://platform.openai.com/api-keys |
| **Model used** | `gpt-4o` (SK AgentGroupChat), `text-embedding-3-small` (SK memory embeddings) |
| **Free tier** | No — requires a paid account with credits loaded |
| **Paid tier** | Pay-as-you-go. Load $5 to start — more than enough for all lessons that use it |
| **Recommendation** | Only needed for 3 specific lessons. Skip those lessons if you don't have a key — the rest of LangGraph and Semantic Kernel works without it. |
| **Set in `.env`** | `OPENAI_API_KEY=sk-...` |
| **Frameworks** | `lang-graph/.env`, `semantic-kernel/.env` |

---

## Serper API Key (`SERPER_API_KEY`)

**Used by:** CrewAI lesson 02, CrewAI lesson 12 (Google Search tool)

| | |
|---|---|
| **Get it at** | https://serper.dev |
| **Free tier** | Yes — 2,500 free searches on signup, no credit card required |
| **Paid tier** | $50/month for 50,000 searches (not needed for learning) |
| **Recommendation** | Free tier is more than enough. Sign up, copy the key, done. |
| **Set in `.env`** | `SERPER_API_KEY=...` |
| **Framework** | `crew-ai/.env` |

---

## Free APIs (no key needed)

These are used throughout lessons without any API key or account:

| API | Used in | Purpose |
|-----|---------|---------|
| Open-Meteo | LangGraph 12 | Live weather data |
| CoinGecko | LangGraph 12 | Live crypto prices |
| Wikipedia | LangGraph 12 | Article summaries |
| REST Countries | LangGraph 12 | Country metadata |
| HackerNews | LangGraph 12 | Top stories |
| GitHub Search | LangGraph 12 | Repository search |

---

## Quick Reference

| Key | Framework | Get it | Free? | Paid recommended? |
|-----|-----------|--------|-------|-------------------|
| `ANTHROPIC_API_KEY` | LangGraph, CrewAI, Semantic Kernel | console.anthropic.com | Yes (limited) | Yes |
| `GOOGLE_API_KEY` | Google ADK | aistudio.google.com | Yes (20 req/day cap) | **Yes — enable billing** |
| `OPENAI_API_KEY` | LangGraph 12, SK 09, SK 12 | platform.openai.com | No | Load $5 credits |
| `SERPER_API_KEY` | CrewAI 02, CrewAI 12 | serper.dev | Yes (2500 searches) | No |
