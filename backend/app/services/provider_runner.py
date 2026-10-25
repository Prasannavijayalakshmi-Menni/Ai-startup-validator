"""Shared LLM provider runner implementing Gemini with Groq fallback."""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from pydantic import ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BLOCKED_MODELS = {"openai/gpt-oss-20b"}

MAX_GROQ_429_RETRIES = 2
MAX_SINGLE_RETRY_DELAY = 6.0
MAX_TOTAL_RETRY_DELAY = 12.0

# Groq timeout configuration.
# connect: time to establish the TCP connection to Groq's API endpoint.
# read:    time to receive the complete response body after the request is sent.
#          Large structured-JSON responses (market/competitor/SWOT analysis) can
#          take 45–75 s on openai/gpt-oss-120b under moderate load.  The previous
#          scalar timeout=30.0 was applied to EACH phase (connect AND read), so a
#          response that arrived 31 s after the request was sent raised
#          httpx.ReadTimeout → HTTP 504 even though Groq was still working.
# write:   time to upload the request body (prompts can be ~8 KB).
# pool:    time to acquire a connection from the pool (not relevant here since we
#          create a fresh AsyncClient per call, but set conservatively).
_GROQ_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=15.0, pool=5.0)


def _parse_duration_string(val: str | None) -> float | None:
    if not val:
        return None
    val = val.strip().lower()
    try:
        return float(val)
    except ValueError:
        pass
    if val.endswith("ms"):
        try:
            return float(val[:-2]) / 1000.0
        except ValueError:
            pass
    if val.endswith("s") and not val.endswith("ms"):
        try:
            return float(val[:-1])
        except ValueError:
            pass
    m = re.match(r"^(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?$", val)
    if m and (m.group(1) or m.group(2)):
        mins = float(m.group(1) or 0)
        secs = float(m.group(2) or 0)
        return mins * 60.0 + secs
    return None


def _parse_http_date(val: str | None) -> float | None:
    if not val:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(val.strip())
        now = datetime.now(timezone.utc)
        diff = (dt - now).total_seconds()
        return max(0.0, diff)
    except Exception:
        return None


def _parse_retry_after(response: httpx.Response, attempt: int) -> tuple[float, str | None]:
    raw_header = response.headers.get("retry-after")
    if raw_header:
        raw_header = raw_header.strip()
        dur = _parse_duration_string(raw_header)
        if dur is not None:
            return max(0.1, dur), raw_header
        date_dur = _parse_http_date(raw_header)
        if date_dur is not None:
            return max(0.1, date_dur), raw_header

    reset_val = response.headers.get("x-ratelimit-reset-tokens") or response.headers.get("x-ratelimit-reset-requests")
    if reset_val:
        reset_val = reset_val.strip()
        dur = _parse_duration_string(reset_val)
        if dur is not None:
            return max(0.1, dur), reset_val

    try:
        body = response.json()
        msg = str(body.get("error", {}).get("message", "") if isinstance(body, dict) else "")
    except Exception:
        msg = ""

    if msg:
        m = re.search(r"try again in (\d+(?:\.\d+)?\s*(?:s|ms|m)?)", msg, re.I)
        if m:
            dur = _parse_duration_string(m.group(1))
            if dur is not None:
                return max(0.1, dur), m.group(1).strip()

    backoff = 1.0 * (2 ** attempt)
    return max(0.1, backoff), None


def _trust_env() -> bool:
    return os.getenv("HTTPX_TRUST_ENV", "true").strip().lower() not in {"0", "false", "no"}


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.message, self.status_code = message, status_code
        super().__init__(message)


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _text_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_values(item)]
    return []


def _safe_gemini_error_message(exc: Exception, api_key: str, sensitive_values: list[str]) -> str:
    """Return bounded provider diagnostics with known secrets and prompt data removed."""
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = str(exc)
    if not message.strip():
        return "provider did not include an error message"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    for value in sensitive_values:
        if value and len(value) >= 4:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", message)
    message = re.sub(r"\bgsk_[A-Za-z0-9_-]+\b", "[REDACTED]", message)
    message = re.sub(r"\btvly-[A-Za-z0-9_-]+\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key|token|secret|authorization)\s*[:=]\s*[^,;\s]+",
        r"\1=[REDACTED]",
        message,
    )
    message = re.sub(r"https?://\S+", "[REDACTED_URL]", message)
    return " ".join(message.split())[:400] or "provider did not include an error message"


def _safe_gemini_request_id(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    for header in ("x-goog-request-id", "x-request-id", "request-id"):
        value = headers.get(header) if hasattr(headers, "get") else None
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:/+-]{1,120}", value.strip()):
            return value.strip()
    return None


def _create_gemini_client(api_key: str):
    """Create the official Google Gen AI client, honoring the project's proxy setting."""
    from google import genai

    return genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(
            async_client_args={"trust_env": _trust_env()},
        ),
    )


def _safe_groq_error_message(
    response: httpx.Response | None,
    exc: Exception | None,
    api_key: str,
    sensitive_values: list[str],
    *,
    agent_name: str = "analysis",
    retries_attempted: int = 0,
) -> tuple[int, str]:
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if not isinstance(error, dict):
            error = {}
        code = str(error.get("code") or "").lower()
        msg = str(error.get("message") or "").strip()
        status_code = response.status_code

        if status_code == 401:
            return 503, "Groq rejected the API credential (HTTP 401)."
        if status_code == 403:
            return 503, "The Groq credential does not have permission to use this resource (HTTP 403)."
        if status_code == 429:
            retry_after_hdr = response.headers.get("retry-after")
            if not retry_after_hdr:
                retry_after_hdr = response.headers.get("x-ratelimit-reset-tokens")
            if not retry_after_hdr:
                m = re.search(r"try again in (\d+(?:\.\d+)?\s*(?:s|ms|m)?)", msg, re.I)
                if m:
                    retry_after_hdr = m.group(1).strip()

            err_type = str(error.get("type") or "").lower()
            is_token = "token" in err_type or "token" in msg.lower() or "tpm" in msg.lower()
            rate_label = "token" if is_token else "request"

            parts = [f"Groq {rate_label} rate limit reached during {agent_name} (HTTP 429)."]
            if retry_after_hdr:
                parts.append(f"Provider reported retry-after={retry_after_hdr}.")
            if retries_attempted > 0:
                parts.append(f"Retried {retries_attempted} times; rate limit persisted.")
            else:
                parts.append("No automatic retry was made.")

            sanitized_body = _safe_gemini_error_message(
                RuntimeError(msg or "Rate limit exceeded"), api_key, sensitive_values
            )
            if sanitized_body and sanitized_body not in ("provider did not include an error message", f"HTTP {status_code}"):
                parts.append(sanitized_body)

            return 503, " ".join(parts)

        if status_code == 402:
            sanitized_body = _safe_gemini_error_message(
                RuntimeError(msg or "Quota exceeded"), api_key, sensitive_values
            )
            return 503, f"The Groq account quota or billing limit has been reached (HTTP 402): {sanitized_body}"
        if status_code in (400, 404) and (
            code in {"invalid_model", "model_not_found", "model_decommissioned", "unknown_model"}
            or any(term in msg.lower() for term in ("model", "not found", "does not exist", "unavailable", "blocked"))
        ):
            return 503, "The configured Groq model is invalid or unavailable."

        sanitized = _safe_gemini_error_message(
            RuntimeError(msg or f"HTTP {status_code}"), api_key, sensitive_values
        )
        if status_code == 400:
            return 502, f"Groq rejected the request (HTTP 400): {sanitized}"
        return 502, f"Groq returned HTTP {status_code}: {sanitized}"

    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return 504, f"Groq request timed out ({type(exc).__name__})."
    if isinstance(exc, (httpx.ConnectError, httpx.RequestError)):
        return 502, f"Could not connect to Groq LLM provider ({type(exc).__name__})."

    sanitized = _safe_gemini_error_message(exc, api_key, sensitive_values)
    return 502, f"Groq provider request failed: {sanitized}"


async def call_groq(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[..., T],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    validation_context: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    sensitive_values: list[str] | None = None,
    gemini_error: ServiceError | None = None,
) -> T:
    """Execute fallback call to Groq and return validated model."""
    sensitive_values = sensitive_values or []
    groq_api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        if gemini_error:
            raise gemini_error
        raise ServiceError("LLM_API_KEY is not configured.", 503)

    groq_base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    groq_model = os.getenv("LLM_MODEL", DEFAULT_GROQ_MODEL)
    if groq_model in GROQ_BLOCKED_MODELS:
        groq_model = DEFAULT_GROQ_MODEL

    format_obj = response_format or {"type": "json_object"}
    request_body = {
        "model": groq_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": format_obj,
    }

    groq_started = time.monotonic()
    logger.info(
        "Groq fallback started: agent=%s model=%s prompt_chars=%d",
        agent_name,
        groq_model,
        len(user_prompt),
    )

    retries_attempted = 0
    total_slept = 0.0

    try:
        async with httpx.AsyncClient(
            timeout=_GROQ_TIMEOUT,
            transport=transport,
            trust_env=_trust_env(),
        ) as http_client:
            for attempt in range(MAX_GROQ_429_RETRIES + 1):
                response = await http_client.post(
                    f"{groq_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {groq_api_key}"},
                    json=request_body,
                )

                if response.status_code >= 400:
                    # If org blocked model, try once with default
                    if "blocked at the organization level" in response.text and groq_model != DEFAULT_GROQ_MODEL:
                        request_body["model"] = DEFAULT_GROQ_MODEL
                        response = await http_client.post(
                            f"{groq_base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {groq_api_key}"},
                            json=request_body,
                        )

                if response.status_code == 429 and attempt < MAX_GROQ_429_RETRIES:
                    delay, raw_hdr = _parse_retry_after(response, attempt)
                    if delay <= MAX_SINGLE_RETRY_DELAY and (total_slept + delay) <= MAX_TOTAL_RETRY_DELAY:
                        retries_attempted += 1
                        total_slept += delay
                        logger.warning(
                            "Groq rate limit (HTTP 429) during %s: attempt %d/%d; backing off for %.2fs "
                            "(retry-after=%s, reset_tokens=%s, remaining_tokens=%s)",
                            agent_name,
                            retries_attempted,
                            MAX_GROQ_429_RETRIES,
                            delay,
                            raw_hdr or "unavailable",
                            response.headers.get("x-ratelimit-reset-tokens", "unavailable"),
                            response.headers.get("x-ratelimit-remaining-tokens", "unavailable"),
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning(
                            "Groq rate limit delay (%.2fs) exceeds retry budget (%.2fs); failing fast",
                            delay,
                            MAX_SINGLE_RETRY_DELAY,
                        )

                break

            if response.status_code < 400:
                payload = response.json()
                choices = payload.get("choices", [])
                if not choices or not isinstance(choices, list):
                    raise ServiceError(f"Groq returned an incomplete response for {agent_name}; no choices produced.")
                content = choices[0].get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ServiceError(f"Groq returned an empty response for {agent_name}.")

                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ServiceError(f"Groq returned invalid JSON for {agent_name}.") from exc

                if not isinstance(parsed, dict):
                    raise ServiceError(f"Groq returned non-object JSON for {agent_name}.")

                try:
                    if validation_context is not None:
                        result = validator(parsed, context=validation_context)
                    else:
                        result = validator(parsed)
                    logger.info(
                        "Groq fallback succeeded: agent=%s model=%s elapsed_seconds=%.3f "
                        "http_status=%d retries=%d",
                        agent_name,
                        groq_model,
                        time.monotonic() - groq_started,
                        response.status_code,
                        retries_attempted,
                    )
                    return result
                except ValidationError as exc:
                    raise ServiceError(
                        f"Groq {agent_name} did not match required schema or cited an unsupported source URL."
                    ) from exc
            else:
                status_code, err_msg = _safe_groq_error_message(
                    response,
                    None,
                    groq_api_key,
                    sensitive_values,
                    agent_name=agent_name,
                    retries_attempted=retries_attempted,
                )
                logger.warning(
                    "Groq fallback failed: agent=%s model=%s elapsed_seconds=%.3f "
                    "http_status=%d sanitized_error=%s retries=%d",
                    agent_name,
                    groq_model,
                    time.monotonic() - groq_started,
                    response.status_code,
                    err_msg,
                    retries_attempted,
                )
                final_status = gemini_error.status_code if (gemini_error and gemini_error.status_code in (503, 504)) else status_code
                prefix = f"{gemini_error.message} " if gemini_error else ""
                raise ServiceError(f"{prefix}Groq fallback also failed: {err_msg}", status_code=final_status)

    except ServiceError as exc:
        if "Groq fallback also failed" in exc.message:
            raise
        final_status = gemini_error.status_code if (gemini_error and gemini_error.status_code in (503, 504)) else exc.status_code
        prefix = f"{gemini_error.message} " if gemini_error else ""
        raise ServiceError(
            f"{prefix}Groq fallback also failed: {exc.message}",
            status_code=final_status,
        ) from exc
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        logger.warning(
            "Groq fallback timed out: agent=%s model=%s elapsed_seconds=%.3f exception=%s",
            agent_name,
            groq_model,
            time.monotonic() - groq_started,
            type(exc).__name__,
        )
        prefix = f"{gemini_error.message} " if gemini_error else ""
        raise ServiceError(
            f"{prefix}Groq fallback also timed out ({type(exc).__name__}).",
            504,
        ) from exc
    except (httpx.ConnectError, httpx.RequestError) as exc:
        logger.warning(
            "Groq fallback connection failed: agent=%s model=%s elapsed_seconds=%.3f exception=%s",
            agent_name,
            groq_model,
            time.monotonic() - groq_started,
            type(exc).__name__,
        )
        prefix = f"{gemini_error.message} " if gemini_error else ""
        raise ServiceError(
            f"{prefix}Groq fallback connection failed ({type(exc).__name__}).",
            502,
        ) from exc
    except Exception as exc:
        status_code, err_msg = _safe_groq_error_message(
            None, exc, groq_api_key, sensitive_values
        )
        logger.warning(
            "Groq fallback unexpected error: agent=%s model=%s elapsed_seconds=%.3f "
            "exception=%s sanitized_error=%s",
            agent_name,
            groq_model,
            time.monotonic() - groq_started,
            type(exc).__name__,
            err_msg,
        )
        final_status = gemini_error.status_code if (gemini_error and gemini_error.status_code in (503, 504)) else status_code
        prefix = f"{gemini_error.message} " if gemini_error else ""
        raise ServiceError(
            f"{prefix}Groq fallback also failed: {err_msg}",
            status_code=final_status,
        ) from exc


async def execute_with_fallback(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[..., T],
    *,
    gemini_caller: Callable[[], Awaitable[T]],
    transport: httpx.AsyncBaseTransport | None = None,
    validation_context: dict[str, Any] | None = None,
    groq_response_format: dict[str, Any] | None = None,
    sensitive_values: list[str] | None = None,
) -> T:
    """Execute LLM call using Google Gemini as primary with Groq as fallback."""
    try:
        return await gemini_caller()
    except ServiceError as gemini_err:
        groq_api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise gemini_err

        logger.warning(
            "Gemini %s failed (status=%d); attempting Groq fallback",
            agent_name,
            gemini_err.status_code,
        )
        return await call_groq(
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=validator,
            transport=transport,
            validation_context=validation_context,
            response_format=groq_response_format,
            sensitive_values=sensitive_values,
            gemini_error=gemini_err,
        )

