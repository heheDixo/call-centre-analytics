# Call Centre Analytics API

An intelligent call centre analytics system that processes voice recordings in Hindi (Hinglish) and Tamil (Tanglish), extracts transcripts using OpenAI Whisper, and performs AI-powered analysis using Claude claude-sonnet-4-6.

## Approach & Strategy

1. **Speech-to-Text** — OpenAI Whisper (`whisper-1`) transcribes the audio. Language codes `hi` (Hindi/Hinglish) and `ta` (Tamil/Tanglish) are passed explicitly for best accuracy.
2. **Structured AI Analysis** — A single Claude `claude-sonnet-4-6` API call using tool use (function calling) extracts: summary, SOP compliance booleans, payment preference, rejection reason, sentiment, and keywords. Tool use enforces exact enum values and boolean types — no fragile JSON parsing.
3. **SOP Scoring** — Compliance score and adherence status are computed deterministically in Python (not by the AI), ensuring consistency.
4. **Async Processing** — Celery with Redis handles audio processing asynchronously. The API endpoint dispatches the task and waits for the result, returning a synchronous response.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI (Python 3.11) |
| Async task queue | Celery 5.4 + Redis |
| Speech-to-Text | OpenAI Whisper (`whisper-1`) |
| NLP / AI | Anthropic Claude `claude-sonnet-4-6` |
| Auth | API key via `x-api-key` header |
| Deployment | Render.com (web + worker) + Redis Cloud |

## Setup

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
# Edit .env and fill in your API keys
```

Required variables:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude) |
| `OPENAI_API_KEY` | OpenAI API key (for Whisper STT) |
| `CALL_ANALYTICS_API_KEY` | Secret key clients must send in `x-api-key` header |
| `REDIS_URL` | Redis connection string (broker + result backend) |

### 4. Start Redis (local development)

```bash
docker run -d -p 6379:6379 redis:7
```

Or set `CELERY_TASK_ALWAYS_EAGER=true` in `.env` to skip Redis entirely (runs Celery tasks in-process, good for quick testing).

### 5. Run the application

**Terminal 1 — FastAPI web server:**
```bash
uvicorn src.main:app --reload --port 8000
```

**Terminal 2 — Celery worker:**
```bash
celery -A src.main.celery_app worker --loglevel=info
```

The API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Usage

### Endpoint

```
POST /api/call-analytics
```

### Headers

```
x-api-key: your-secret-key
Content-Type: application/json
```

### Request Body

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded audio>"
}
```

**Supported languages:** `Hindi`, `Hinglish`, `Tamil`, `Tanglish`

**Supported formats:** `mp3`, `wav`, `m4a`, `webm`, `ogg`, `flac`

### Example (curl)

```bash
# Encode audio file
BASE64=$(base64 -i call.mp3 | tr -d '\n')

# Call the API
curl -X POST https://your-domain.com/api/call-analytics \
  -H "x-api-key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d "{\"language\":\"Tamil\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

### Response

```json
{
  "status": "success",
  "language": "Tamil",
  "transcript": "வணக்கம், நான் Guvi இல் இருந்து பேசுகிறேன்...",
  "summary": "Agent called customer to discuss a Data Science course. Customer showed interest in EMI options.",
  "sop_validation": {
    "greeting": true,
    "identification": true,
    "problemStatement": true,
    "solutionOffering": true,
    "closing": false,
    "complianceScore": 0.8,
    "adherenceStatus": "FOLLOWED",
    "explanation": "Agent followed most SOP steps but did not formally close the call."
  },
  "analytics": {
    "paymentPreference": "EMI",
    "rejectionReason": "NONE",
    "sentiment": "Positive"
  },
  "keywords": ["Data Science", "EMI", "course", "Guvi", "enrollment", "payment"]
}
```

## Deployment

### Render.com

1. Push this repository to GitHub.
2. Create a free account at [render.com](https://render.com).
3. Create **Web Service** from your GitHub repo:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
4. Create **Background Worker** from the same repo:
   - Start command: `celery -A src.main.celery_app worker --loglevel=info --concurrency=2`
5. Set environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CALL_ANALYTICS_API_KEY`, `REDIS_URL`) on both services.
6. Use [Redis Cloud free tier](https://redis.com/try-free/) for `REDIS_URL`.

The `render.yaml` file in this repo configures both services automatically.

## Error Codes

| Status | Meaning |
|--------|---------|
| 401 | Missing or invalid `x-api-key` |
| 422 | Invalid request (bad language, format, or base64) |
| 500 | Internal processing error |
| 502 | Upstream API error (Whisper or Claude) |
