"""Isolated REST diagnostic for Gemini Market Analysis request serialization.

This module deliberately does not replace the production Gen AI SDK path.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.idea_extraction import ServiceError, _trust_env
from app.services.market_analysis import (
    GEMINI_MODEL,
    MARKET_RESEARCH_CATEGORIES,
    _gemini_response_schema,
    _safe_gemini_error_message,
)

GEMINI_REST_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


def _gemini_rest_response_schema() -> dict[str, Any]:
    """Inline local Pydantic references unsupported by Gemini REST responseSchema."""
    schema = deepcopy(_gemini_response_schema())
    definitions = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, list):
            return [inline(item) for item in node]
        if not isinstance(node, dict):
            return node
        reference = node.get("$ref")
        if reference is not None:
            prefix = "#/$defs/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise ValueError("Unsupported non-local reference in diagnostic schema")
            name = reference[len(prefix):]
            if name not in definitions:
                raise ValueError("Diagnostic schema reference is undefined")
            return inline(deepcopy(definitions[name]))
        return {key: inline(value) for key, value in node.items() if key != "$defs"}

    resolved = inline(schema)
    if not isinstance(resolved, dict):
        raise ValueError("Diagnostic response schema must be an object")
    return resolved


@dataclass
class GeminiMarketRestDiagnosticResult:
    elapsed_seconds: float
    status_code: int | None
    analysis: MarketOpportunityAnalysis | None = None
    error_type: str | None = None
    error_code: str | None = None
    safe_error: str | None = None


def _diagnostic_prompts(
    idea: StructuredIdea,
    sources: list[Source],
) -> tuple[str, str]:
    system_prompt = (
        "Return exactly one JSON object matching the supplied schema. Use only the supplied "
        "research, cite exact supplied URLs, and do not invent facts or URLs."
    )
    data = {
        "startup_idea": idea.model_dump(),
        "research_sources": [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in sources
        ],
    }
    user_prompt = (
        "Analyze market demand and customer segments using only this supplied synthetic "
        "diagnostic data. Return a single JSON object, without Markdown or surrounding text. "
        "Evidence-based claims must cite exact supplied source URLs. Use claim \"Unknown\", "
        "classification \"unknown\", and an empty evidence list when evidence is insufficient.\n"
        + json.dumps(data, ensure_ascii=False)
    )
    return system_prompt, user_prompt


def _safe_provider_error(
    payload: Any,
    *,
    api_key: str,
    sensitive_values: list[str],
) -> tuple[str | None, str]:
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        error = {}
    code = error.get("status") or error.get("code")
    safe_code = str(code)[:80] if isinstance(code, (str, int)) else None
    safe_message = _safe_gemini_error_message(
        type("ProviderMessage", (), {"message": error.get("message")})(),
        api_key,
        sensitive_values,
    )
    return safe_code, safe_message


async def run_market_analysis_rest_diagnostic(
    idea: StructuredIdea,
    research: dict[str, list[Source]],
    *,
    api_key: str | None = None,
    timeout_seconds: float = 65.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GeminiMarketRestDiagnosticResult:
    """Make one direct REST call and return only safe status/validation metadata."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ServiceError("GEMINI_API_KEY is not configured.", 503)

    sources = [
        source
        for category in MARKET_RESEARCH_CATEGORIES
        for source in research.get(category, [])
    ]
    if not sources:
        raise ValueError("The isolated REST diagnostic requires at least one research source")

    allowed_urls = {source.url for source in sources}
    system_prompt, user_prompt = _diagnostic_prompts(idea, sources)
    sensitive_values = [system_prompt, user_prompt]
    sensitive_values.extend(value for source in sources for value in (source.title, source.url, source.content))
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _gemini_rest_response_schema(),
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            trust_env=_trust_env(),
            transport=transport,
        ) as client:
            response = await client.post(
                GEMINI_REST_ENDPOINT,
                headers={"x-goog-api-key": resolved_key, "Content-Type": "application/json"},
                json=body,
            )
    except httpx.TimeoutException as exc:
        return GeminiMarketRestDiagnosticResult(
            elapsed_seconds=time.monotonic() - started,
            status_code=None,
            error_type=type(exc).__name__,
            safe_error="Gemini REST diagnostic request timed out",
        )
    except httpx.RequestError as exc:
        return GeminiMarketRestDiagnosticResult(
            elapsed_seconds=time.monotonic() - started,
            status_code=None,
            error_type=type(exc).__name__,
            safe_error="Gemini REST diagnostic connection failed",
        )

    elapsed = time.monotonic() - started
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = None

    if response.status_code >= 400:
        error_code, safe_error = _safe_provider_error(
            payload,
            api_key=resolved_key,
            sensitive_values=sensitive_values,
        )
        return GeminiMarketRestDiagnosticResult(
            elapsed_seconds=elapsed,
            status_code=response.status_code,
            error_type="GeminiHTTPError",
            error_code=error_code,
            safe_error=safe_error,
        )

    try:
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str))
        if not text.strip():
            raise ValueError("empty response")
        parsed = json.loads(text)
        analysis = MarketOpportunityAnalysis.model_validate(
            parsed,
            context={"allowed_urls": allowed_urls},
        )
    except json.JSONDecodeError:
        return GeminiMarketRestDiagnosticResult(
            elapsed_seconds=elapsed,
            status_code=response.status_code,
            error_type="JSONDecodeError",
            safe_error="Gemini response was not valid JSON",
        )
    except (ValidationError, ValueError, TypeError, IndexError, KeyError):
        return GeminiMarketRestDiagnosticResult(
            elapsed_seconds=elapsed,
            status_code=response.status_code,
            error_type="ResponseValidationError",
            safe_error="Gemini response was empty or failed schema/citation validation",
        )

    return GeminiMarketRestDiagnosticResult(
        elapsed_seconds=elapsed,
        status_code=response.status_code,
        analysis=analysis,
    )
