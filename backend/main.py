from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import StartupIdea
from app.services.idea_extraction import ServiceError
from app.services.orchestration import ValidationReportService
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).with_name(".env"))
app = FastAPI(title="AI Startup Validator API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home():
    return {"message": "Startup Validator backend is running"}


@app.post("/api/ideas")
def submit_idea(idea: StartupIdea):
    exclude = set() if "estimated_budget_currency" in idea.model_fields_set else {"estimated_budget_currency"}
    return {"message": "Startup idea received successfully", "idea": idea.model_dump(exclude=exclude)}


@app.post("/api/ideas/analyze")
async def analyze_idea(idea: StartupIdea):
    try:
        return await ValidationReportService().analyze(idea)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@app.get("/api/health/providers")
async def provider_health():
    """Non-production diagnostic: probe each configured provider with a minimal
    request and report only reachability, HTTP status, and sanitized errors.
    Never exposes API keys, prompts, or sensitive response values.
    Call this endpoint manually from a shell or browser; it is NOT called by
    the frontend and does NOT run automatically during app startup."""
    import asyncio
    import re

    import httpx

    results: dict = {}

    # ── Gemini ──────────────────────────────────────────────────────────────
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    if not gemini_api_key:
        results["gemini"] = {
            "configured": False,
            "status": "skipped",
            "reason": "GEMINI_API_KEY not set",
        }
    else:
        try:
            from app.services.provider_runner import _create_gemini_client

            client = _create_gemini_client(gemini_api_key)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=gemini_model,
                    contents="Reply with exactly: ok",
                    config={"response_mime_type": "text/plain"},
                ),
                timeout=15.0,
            )
            text = (response.text or "").strip()[:20]
            results["gemini"] = {
                "configured": True,
                "status": "ok",
                "model": gemini_model,
                "response_preview": text,
            }
        except asyncio.TimeoutError:
            results["gemini"] = {
                "configured": True,
                "status": "timeout",
                "model": gemini_model,
            }
        except Exception as exc:
            code = getattr(exc, "code", None)
            try:
                code = int(code)
            except (TypeError, ValueError):
                code = None
            safe_msg = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", str(exc))
            safe_msg = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", safe_msg)
            safe_msg = re.sub(r"\bgsk_[A-Za-z0-9_-]+\b", "[REDACTED]", safe_msg)
            safe_msg = " ".join(safe_msg.split())[:200]
            results["gemini"] = {
                "configured": True,
                "status": "error",
                "model": gemini_model,
                "http_code": code,
                "error": safe_msg,
            }

    # ── Groq ────────────────────────────────────────────────────────────────
    groq_api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    groq_base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    if not groq_api_key:
        results["groq"] = {
            "configured": False,
            "status": "skipped",
            "reason": "LLM_API_KEY not set",
        }
    else:
        try:
            from app.services.provider_runner import _trust_env

            async with httpx.AsyncClient(timeout=15.0, trust_env=_trust_env()) as http_client:
                resp = await http_client.post(
                    f"{groq_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {groq_api_key}"},
                    json={
                        "model": groq_model,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                        "max_tokens": 5,
                    },
                )
            if resp.status_code < 400:
                choices = resp.json().get("choices", [])
                text = (
                    (choices[0].get("message", {}).get("content", "") if choices else "").strip()[:20]
                )
                results["groq"] = {
                    "configured": True,
                    "status": "ok",
                    "model": groq_model,
                    "http_status": resp.status_code,
                    "response_preview": text,
                }
            else:
                try:
                    err_body = resp.json().get("error", {})
                    safe_msg = str(err_body.get("message", f"HTTP {resp.status_code}"))[:200]
                except Exception:
                    safe_msg = f"HTTP {resp.status_code}"
                safe_msg = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", safe_msg)
                safe_msg = re.sub(r"\bgsk_[A-Za-z0-9_-]+\b", "[REDACTED]", safe_msg)
                results["groq"] = {
                    "configured": True,
                    "status": "error",
                    "model": groq_model,
                    "http_status": resp.status_code,
                    "error": safe_msg,
                }
        except asyncio.TimeoutError:
            results["groq"] = {"configured": True, "status": "timeout", "model": groq_model}
        except (httpx.ConnectError, httpx.RequestError) as exc:
            results["groq"] = {
                "configured": True,
                "status": "connection_error",
                "model": groq_model,
                "error": type(exc).__name__,
            }

    return results
