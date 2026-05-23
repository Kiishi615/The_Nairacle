# Nairacle Research Agent

A Nigerian financial research agent powered by LangChain, Anthropic Claude, Pinecone, and Tavily. Provides an intelligent Telegram Mini App interface for researching Nigerian markets, banking, economy, stock prices, and policy.

## Architecture

```
research_agent/
├── agent.py          # LangGraph agent with 9 tools
├── api.py            # FastAPI REST + SSE streaming endpoints
├── auth.py           # Telegram Mini App HMAC auth
├── config.py         # Centralised configuration
├── main.py           # FastAPI entry point + Telegram webhook
├── reformulator.py   # Query reformulation chain
├── reranker.py       # Cross-encoder reranker
├── logging_setup.py  # Structured logging
├── nigerian_stocks.json
├── configs/
│   ├── dev.cfg
│   └── prod.cfg
├── sessions/
│   ├── database.py   # SQLite async helpers
│   ├── manager.py    # Session CRUD + summaries
│   └── models.py     # Dataclasses
└── app/              # Telegram Mini App frontend
    ├── index.html
    ├── style.css
    ├── app.js
    └── fonts/
```

## Deploy to Railway

### 1. Create Railway Project

1. Go to [railway.app](https://railway.app) and create a new project
2. Connect your GitHub repo (push this folder first) **OR** deploy directly via Railway CLI

### 2. Set Environment Variables

In the Railway dashboard, add these variables:

| Variable | Description |
|----------|-------------|
| `APP_ENV` | `prod` |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name |
| `TELEGRAM_API_KEY` | Telegram Bot token |
| `TAVILY_API_KEY` | Tavily API key (at least one) |
| `TAVILY_API_KEY_2` … `_6` | Optional additional Tavily keys |
| `DATABASE_URL` | PostgreSQL connection string (Supabase/Neon) |
| `WEBHOOK_SECRET` | Random string for Telegram webhook auth |
| `LANGSMITH_API_KEY` | *(optional)* LangSmith tracing key |
| `LANGCHAIN_TRACING_V2` | *(optional)* `true` to enable tracing |
| `LANGCHAIN_PROJECT` | *(optional)* LangSmith project name |

> **Note:** Railway automatically provides `PORT`. Do NOT set it manually.

### 3. Deploy

Railway will automatically detect the `Dockerfile` and build. The `railway.toml` configures:
- Dockerfile-based builds
- Health check on `/health`
- Auto-restart on failure

### 4. Set Telegram Webhook

After deployment, set the webhook URL in your bot:

```
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook?url=https://<your-railway-url>/webhook&secret_token=<WEBHOOK_SECRET>
```

## Local Development

```bash
# Install dependencies
pip install -r research_agent/requirements.txt

# Create .env with your API keys
cp .env.example .env

# Run locally
APP_ENV=dev uvicorn research_agent.main:app --reload --port 8000
```

## Docker (Local Testing)

```bash
# Build
docker build -t nairacle-agent .

# Run
docker run -p 8000:8000 --env-file .env nairacle-agent
```
