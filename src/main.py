"""
Call Centre Analytics API
=========================
Processes Hindi (Hinglish) and Tamil (Tanglish) voice recordings.

Pipeline:
  base64 audio → Groq Whisper STT (free) → Groq LLaMA 3.3 70B (free) → structured JSON

Free tier: Sign up at https://console.groq.com — no credit card required.
"""

import os
import base64
import json
import tempfile
import logging
from typing import Optional

from groq import Groq
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from celery import Celery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
GROQ_API_KEY           = os.environ.get("GROQ_API_KEY", "")
CALL_ANALYTICS_API_KEY = os.environ.get("CALL_ANALYTICS_API_KEY", "changeme")
REDIS_URL              = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# When True, Celery tasks run in-process (no Redis / worker needed).
# Great for simple deployments — set to "false" to use a real Celery worker.
CELERY_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "true").lower() == "true"

# ── Celery ────────────────────────────────────────────────────────────────────
celery_app = Celery("call_analytics")
celery_app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    task_always_eager=CELERY_EAGER,   # True = synchronous in-process (no Redis needed)
    task_eager_propagates=True,
)

# ── Language mapping (Whisper language codes) ─────────────────────────────────
LANGUAGE_MAP: dict[str, str] = {
    "hindi":    "hi",
    "hinglish": "hi",
    "tamil":    "ta",
    "tanglish": "ta",
}

SUPPORTED_LANGUAGES = {"Hindi", "Hinglish", "Tamil", "Tanglish"}
SUPPORTED_FORMATS   = {"mp3", "wav", "m4a", "webm", "ogg", "flac"}

# ── Pydantic models ───────────────────────────────────────────────────────────
class CallAnalyticsRequest(BaseModel):
    language:    str = Field(..., description="Hindi, Hinglish, Tamil, or Tanglish")
    audioFormat: str = Field(..., description="mp3, wav, etc.")
    audioBase64: str = Field(..., description="Base64-encoded audio file content")


class SOPValidation(BaseModel):
    greeting:         bool
    identification:   bool
    problemStatement: bool
    solutionOffering: bool
    closing:          bool
    complianceScore:  float
    adherenceStatus:  str   # "FOLLOWED" | "NOT_FOLLOWED"
    explanation:      str


class Analytics(BaseModel):
    paymentPreference: str  # EMI | FULL_PAYMENT | PARTIAL_PAYMENT | DOWN_PAYMENT
    rejectionReason:   str  # HIGH_INTEREST | BUDGET_CONSTRAINTS | ALREADY_PAID | NOT_INTERESTED | NONE
    sentiment:         str  # Positive | Neutral | Negative


class CallAnalyticsResponse(BaseModel):
    status:         str
    language:       str
    transcript:     str
    summary:        str
    sop_validation: SOPValidation
    analytics:      Analytics
    keywords:       list[str]


# ── Auth ──────────────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    valid_keys = {k.strip() for k in os.environ.get("VALID_API_KEYS", "").split(",") if k.strip()}
    valid_keys.add(CALL_ANALYTICS_API_KEY)
    if api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# ── SOP scoring (deterministic Python logic) ──────────────────────────────────
def compute_sop_scores(sop_booleans: dict) -> dict:
    """
    Add complianceScore and adherenceStatus to the 5 booleans returned by the LLM.
    Threshold: >= 0.6 → FOLLOWED (i.e., 3 or more of 5 steps present).
    """
    steps = ["greeting", "identification", "problemStatement", "solutionOffering", "closing"]
    true_count = sum(1 for s in steps if sop_booleans.get(s, False))
    score = round(true_count / 5, 2)
    return {
        **sop_booleans,
        "complianceScore": score,
        "adherenceStatus": "FOLLOWED" if score >= 0.6 else "NOT_FOLLOWED",
    }


# ── Groq tool schema (OpenAI-compatible format) ───────────────────────────────
ANALYZE_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_call",
        "description": (
            "Extract structured analytics from a call centre transcript. "
            "Return all fields exactly as specified — booleans as true/false, "
            "strings as exact enum values listed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2-3 sentence English summary of the key points discussed in the call."
                },
                "sop_validation": {
                    "type": "object",
                    "description": "SOP compliance check. Standard flow: Greeting → Identification → Problem Statement → Solution Offering → Closing.",
                    "properties": {
                        "greeting": {
                            "type": "boolean",
                            "description": "true if the agent greeted the customer at the start of the call"
                        },
                        "identification": {
                            "type": "boolean",
                            "description": "true if the agent introduced themselves or verified the customer's identity"
                        },
                        "problemStatement": {
                            "type": "boolean",
                            "description": "true if the purpose or issue of the call was clearly stated"
                        },
                        "solutionOffering": {
                            "type": "boolean",
                            "description": "true if the agent offered a solution, plan, or discussed options with the customer"
                        },
                        "closing": {
                            "type": "boolean",
                            "description": "true if the call ended with a proper closing, confirmation, or farewell"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One sentence summarising overall SOP compliance of this call"
                        }
                    },
                    "required": ["greeting", "identification", "problemStatement", "solutionOffering", "closing", "explanation"]
                },
                "analytics": {
                    "type": "object",
                    "properties": {
                        "paymentPreference": {
                            "type": "string",
                            "enum": ["EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT"],
                            "description": (
                                "Payment mode discussed or preferred by the customer. "
                                "EMI=installments/monthly payments, FULL_PAYMENT=full amount at once, "
                                "PARTIAL_PAYMENT=paying part of the amount, DOWN_PAYMENT=initial deposit. "
                                "IMPORTANT: You MUST return one of these four values. "
                                "If no payment was discussed, return EMI as the default. "
                                "Never return null, NONE, or any other value."
                            )
                        },
                        "rejectionReason": {
                            "type": "string",
                            "enum": ["HIGH_INTEREST", "BUDGET_CONSTRAINTS", "ALREADY_PAID", "NOT_INTERESTED", "NONE"],
                            "description": "Reason customer declined. Use NONE if the customer did not reject."
                        },
                        "sentiment": {
                            "type": "string",
                            "enum": ["Positive", "Neutral", "Negative"],
                            "description": "Overall customer sentiment during the call"
                        }
                    },
                    "required": ["paymentPreference", "rejectionReason", "sentiment"]
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "5-10 meaningful keywords or topics from the call (no filler words)"
                }
            },
            "required": ["summary", "sop_validation", "analytics", "keywords"]
        }
    }
}

SYSTEM_PROMPT = """You are an expert call centre analytics system specializing in Indian language calls.
You analyze transcripts in Hindi, Hinglish (Hindi mixed with English), Tamil, and Tanglish (Tamil mixed with English).

Standard SOP (Standard Operating Procedure) for a compliant call:
1. Greeting       — agent warmly greets the customer
2. Identification — agent introduces themselves and/or verifies the customer
3. Problem Statement — the purpose or issue is clearly stated
4. Solution Offering — agent proposes a solution or discusses options
5. Closing        — call ends with confirmation, next steps, or polite farewell

Analyze carefully and return accurate results based only on what is in the transcript."""


def build_user_message(transcript: str, language: str) -> str:
    return (
        f"Analyze this {language} call centre transcript and extract structured analytics.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Call the analyze_call function with your analysis."
    )


# ── Celery Task ───────────────────────────────────────────────────────────────
@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
def process_call_analytics(self, language: str, audio_format: str, audio_base64: str) -> dict:
    """
    Core processing pipeline:
      1. Decode base64 audio → temp file
      2. Groq Whisper (whisper-large-v3) → transcript
      3. Groq LLaMA 3.3 70B with tool calling → structured analysis
      4. Compute SOP scores in Python
    """
    tmp_path = None
    try:
        # ── 1. Decode base64 audio ────────────────────────────────────────
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 audio data: {exc}") from exc

        suffix = f".{audio_format.lower()}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        logger.info("Audio decoded: %d bytes → %s", len(audio_bytes), tmp_path)

        # ── 2. Groq Whisper STT ───────────────────────────────────────────
        whisper_lang = LANGUAGE_MAP.get(language.lower(), "hi")
        client = Groq(api_key=GROQ_API_KEY)

        logger.info("Transcribing audio with Groq Whisper (lang=%s)…", whisper_lang)
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(f"audio{suffix}", audio_file.read()),
                model="whisper-large-v3",
                language=whisper_lang,
                response_format="text",
            )

        # Groq returns the text directly when response_format="text"
        transcript = str(transcription).strip()
        if not transcript:
            raise ValueError("Whisper returned an empty transcript — audio may be silent or corrupted")
        logger.info("Transcript (%d chars): %s…", len(transcript), transcript[:120])

        # ── 3. Groq LLaMA — structured analysis via tool calling ──────────
        logger.info("Analyzing transcript with Groq LLaMA…")
        chat_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2048,
            temperature=0.1,   # low temp for consistent structured output
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_message(transcript, language)},
            ],
            tools=[ANALYZE_CALL_TOOL],
            tool_choice={"type": "function", "function": {"name": "analyze_call"}},
        )

        # Extract tool call result
        tool_calls = chat_response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("LLM did not return a tool call — cannot extract structured analysis")

        analysis = json.loads(tool_calls[0].function.arguments)
        logger.info("Analysis complete: sentiment=%s payment=%s",
                    analysis["analytics"]["sentiment"],
                    analysis["analytics"]["paymentPreference"])

        # ── 4. Compute SOP scores (deterministic) ────────────────────────
        sop_with_scores = compute_sop_scores(analysis["sop_validation"])

        return {
            "status":         "success",
            "language":       language,
            "transcript":     transcript,
            "summary":        analysis["summary"],
            "sop_validation": sop_with_scores,
            "analytics":      analysis["analytics"],
            "keywords":       analysis["keywords"],
        }

    except ValueError as exc:
        # Client-side errors — do not retry
        logger.error("Client error (no retry): %s", exc)
        raise

    except Exception as exc:
        logger.error("Task error (will retry): %s", exc, exc_info=True)
        raise self.retry(exc=exc) from exc

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Call Centre Analytics API",
    description=(
        "Analyzes Hindi/Hinglish and Tamil/Tanglish call recordings. "
        "Returns transcript, summary, SOP compliance, payment analytics, and keywords."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend UI
# abspath ensures correct resolution whether run from repo root or src/
_static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
_index_html = os.path.join(_static_dir, "index.html")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    logger.info("Serving static files from %s", _static_dir)


@app.get("/", include_in_schema=False)
def frontend():
    if os.path.isfile(_index_html):
        return FileResponse(_index_html)
    # Fallback: redirect to interactive API docs
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get(
    "/api/call-analytics",
    include_in_schema=False,
)
def call_analytics_info():
    """Friendly response for browsers hitting the POST endpoint via GET."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs#/Analytics/Analyze_a_call_recording_api_call_analytics_post")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post(
    "/api/call-analytics",
    response_model=CallAnalyticsResponse,
    tags=["Analytics"],
    summary="Analyze a call recording",
)
def call_analytics(
    request: CallAnalyticsRequest,
    _: str = Depends(verify_api_key),
):
    """
    Accepts a base64-encoded audio recording and returns structured call analytics.

    **Auth**: Send your API key in the `x-api-key` header.
    """
    # Validate inputs
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{request.language}'. Must be one of: {sorted(SUPPORTED_LANGUAGES)}",
        )
    if request.audioFormat.lower() not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audioFormat '{request.audioFormat}'. Must be one of: {sorted(SUPPORTED_FORMATS)}",
        )
    if not request.audioBase64 or len(request.audioBase64) < 100:
        raise HTTPException(status_code=422, detail="audioBase64 is empty or too short")

    logger.info("Dispatching task — language=%s format=%s", request.language, request.audioFormat)

    # Dispatch to Celery (eager mode = runs inline; real mode = async worker)
    task = process_call_analytics.delay(
        request.language,
        request.audioFormat,
        request.audioBase64,
    )

    try:
        result = task.get(timeout=180, propagate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        msg = str(exc)
        logger.error("Task failed: %s", msg)
        raise HTTPException(status_code=500, detail=f"Processing failed: {msg}") from exc

    return CallAnalyticsResponse(**result)
