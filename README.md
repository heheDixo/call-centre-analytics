# Call Center Compliance API

## Description

An AI-powered API that processes call centre audio recordings in Hindi (Hinglish) and Tamil (Tanglish). It performs multi-stage analysis — transcription, SOP compliance validation, payment/sentiment classification, and keyword extraction — returning structured JSON with compliance scores and business intelligence. Transcripts are indexed in a vector store for semantic search.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend Framework | FastAPI (Python 3.11) |
| Speech-to-Text | Groq Whisper (`whisper-large-v3`) |
| LLM / NLP Analysis | Groq LLaMA 3.3 70B (structured output via tool calling) |
| Vector Storage | ChromaDB (persistent, cosine similarity) |
| Task Queue | Celery 5.4 (eager mode or Redis worker) |
| Authentication | API key via `x-api-key` header |
| Deployment | Render.com (web + worker) + Redis Cloud |

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/call-centre-analytics.git
cd call-centre-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) |
| `CALL_ANALYTICS_API_KEY` | Secret key clients send in `x-api-key` header |
| `CELERY_TASK_ALWAYS_EAGER` | `true` = run tasks in-process (no Redis needed) |
| `REDIS_URL` | Redis connection string (only needed if eager=false) |

### 4. Run the application

```bash
uvicorn src.main:app --reload --port 8000
```

The API docs are available at `http://localhost:8000/docs` and the web UI at `http://localhost:8000/`.

## Approach

1. **Audio Decoding** — Base64-encoded MP3 is decoded and written to a temp file.
2. **Speech-to-Text** — Groq Whisper (`whisper-large-v3`) transcribes the audio with explicit language codes (`hi` for Hindi/Hinglish, `ta` for Tamil/Tanglish) for best accuracy.
3. **Structured AI Analysis** — A single Groq LLaMA 3.3 70B API call using tool calling (function calling) extracts: summary, SOP compliance booleans, payment preference, rejection reason, sentiment, and keywords. Tool use enforces exact enum values and boolean types — no fragile JSON parsing.
4. **SOP Scoring** — Compliance score and adherence status are computed deterministically in Python (not by the LLM), ensuring consistency. All 5 SOP steps must be present for "FOLLOWED" status.
5. **Vector Indexing** — Each processed transcript is stored in ChromaDB with metadata (language, sentiment, adherence status, summary) for semantic search via the `/api/search` endpoint.
6. **Async Processing** — Celery handles task execution. In eager mode, tasks run in-process (no Redis needed). In production, a separate Celery worker processes tasks via Redis.

## API Usage

### Analyze a Call

```
POST /api/call-analytics
Header: x-api-key: your-secret-key
```

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded audio>"
}
```

### Semantic Search

```
POST /api/search
Header: x-api-key: your-secret-key
```

```json
{
  "query": "EMI payment discussion",
  "top_k": 5
}
```
