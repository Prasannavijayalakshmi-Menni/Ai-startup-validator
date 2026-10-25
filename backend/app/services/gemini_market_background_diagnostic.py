"""Isolated Gemini Interactions background diagnostic for Market Analysis."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.gemini_market_rest_diagnostic import (
    _diagnostic_prompts,
    _gemini_rest_response_schema,
)
from app.services.idea_extraction import ServiceError, _trust_env
from app.services.market_analysis import (
    GEMINI_MODEL,
    MARKET_RESEARCH_CATEGORIES,
    _safe_gemini_error_message,
)


@dataclass
class GeminiMarketBackgroundDiagnosticResult:
    elapsed_seconds: float
    status: str
    interaction_id: str | None = None
    http_status: int | None = None
    analysis: MarketOpportunityAnalysis | None = None
    error_type: str | None = None
    error_code: str | None = None
    safe_error: str | None = None


def _create_gemini_client(api_key: str):
    from google import genai

    return genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(
            async_client_args={"trust_env": _trust_env()},
        ),
    )


def _provider_exception_details(exc: Exception, api_key: str, sensitive: list[str]):
    code = getattr(exc, "code", None)
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = None
    provider_code = getattr(exc, "status", None)
    safe_code = str(provider_code)[:80] if isinstance(provider_code, (str, int)) else None
    safe_error = _safe_gemini_error_message(exc, api_key, sensitive)
    return status, safe_code, safe_error


async def run_market_analysis_background_diagnostic(
    idea: StructuredIdea,
    research: dict[str, list[Source]],
    *,
    api_key: str | None = None,
    total_timeout_seconds: float = 90.0,
    poll_interval_seconds: float = 3.0,
    client_factory=None,
) -> GeminiMarketBackgroundDiagnosticResult:
    """Create and poll one background interaction, returning safe validation metadata."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ServiceError("GEMINI_API_KEY is not configured.", 503)

    sources = [
        source
        for category in MARKET_RESEARCH_CATEGORIES
        for source in research.get(category, [])
    ]
    if not sources:
        raise ValueError("The background diagnostic requires at least one research source")

    allowed_urls = {source.url for source in sources}
    system_prompt, input_prompt = _diagnostic_prompts(idea, sources)
    sensitive_values = [system_prompt, input_prompt]
    sensitive_values.extend(
        value for source in sources for value in (source.title, source.url, source.content)
    )
    started = time.monotonic()
    deadline = started + total_timeout_seconds
    factory = client_factory or _create_gemini_client
    client = factory(resolved_key)
    async_client = client.aio

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        interaction = await asyncio.wait_for(
            async_client.interactions.create(
                model=GEMINI_MODEL,
                input=input_prompt,
                system_instruction=system_prompt,
                background=True,
                store=True,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _gemini_rest_response_schema(),
                },
                generation_config={"thinking_level": "low"},
            ),
            timeout=remaining,
        )
        interaction_id = getattr(interaction, "id", None)
        status = getattr(interaction, "status", "unknown")
        if hasattr(status, "value"):
            status = status.value

        while status == "in_progress":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.sleep(min(poll_interval_seconds, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            interaction = await asyncio.wait_for(
                async_client.interactions.get(id=interaction_id),
                timeout=remaining,
            )
            status = getattr(interaction, "status", "unknown")
            if hasattr(status, "value"):
                status = status.value

        elapsed = time.monotonic() - started
        if status != "completed":
            return GeminiMarketBackgroundDiagnosticResult(
                elapsed_seconds=elapsed,
                status=str(status),
                interaction_id=interaction_id,
                error_type="InteractionFailed",
                safe_error="Gemini background interaction did not complete successfully",
            )

        output_text = getattr(interaction, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            return GeminiMarketBackgroundDiagnosticResult(
                elapsed_seconds=elapsed,
                status="completed",
                interaction_id=interaction_id,
                error_type="EmptyOutput",
                safe_error="Gemini background interaction completed without text output",
            )
        try:
            payload: Any = json.loads(output_text)
            analysis = MarketOpportunityAnalysis.model_validate(
                payload,
                context={"allowed_urls": allowed_urls},
            )
        except json.JSONDecodeError:
            return GeminiMarketBackgroundDiagnosticResult(
                elapsed_seconds=elapsed,
                status="completed",
                interaction_id=interaction_id,
                error_type="JSONDecodeError",
                safe_error="Gemini background output was not valid JSON",
            )
        except Exception as exc:
            from pydantic import ValidationError

            if not isinstance(exc, (ValidationError, TypeError, ValueError)):
                raise
            return GeminiMarketBackgroundDiagnosticResult(
                elapsed_seconds=elapsed,
                status="completed",
                interaction_id=interaction_id,
                error_type="ResponseValidationError",
                safe_error="Gemini background output failed schema or citation validation",
            )

        return GeminiMarketBackgroundDiagnosticResult(
            elapsed_seconds=elapsed,
            status="completed",
            interaction_id=interaction_id,
            analysis=analysis,
        )
    except asyncio.TimeoutError:
        return GeminiMarketBackgroundDiagnosticResult(
            elapsed_seconds=time.monotonic() - started,
            status="timeout",
            interaction_id=locals().get("interaction_id"),
            error_type="TimeoutError",
            safe_error="Gemini background diagnostic exceeded its total time limit",
        )
    except Exception as exc:
        status, error_code, safe_error = _provider_exception_details(
            exc,
            resolved_key,
            sensitive_values,
        )
        return GeminiMarketBackgroundDiagnosticResult(
            elapsed_seconds=time.monotonic() - started,
            status="provider_error",
            interaction_id=locals().get("interaction_id"),
            http_status=status,
            error_type=type(exc).__name__,
            error_code=error_code,
            safe_error=safe_error,
        )
    finally:
        close_async = getattr(async_client, "aclose", None)
        if callable(close_async):
            await close_async()
        close_sync = getattr(client, "close", None)
        if callable(close_sync):
            close_sync()
