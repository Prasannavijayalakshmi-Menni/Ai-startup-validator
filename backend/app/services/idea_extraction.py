"""Startup idea extraction using Google Gemini with Groq fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas import StartupIdea, StructuredIdea
from app.services.provider_runner import (
    GEMINI_MODEL,
    ServiceError,
    _create_gemini_client,
    _safe_gemini_error_message,
    _safe_gemini_request_id,
    _trust_env,
    execute_with_fallback,
)

logger = logging.getLogger(__name__)


# Kept for backward compatibility with external/legacy callers.
def _provider_error(response: httpx.Response) -> tuple[str, str, str]:
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        error = {}
    if not isinstance(error, dict):
        error = {}
    code = re.sub(r"[^A-Za-z0-9_.-]", "", str(error.get("code") or ""))[:60].lower()
    error_type = re.sub(r"[^A-Za-z0-9_.-]", "", str(error.get("type") or ""))[:60].lower()
    message = str(error.get("message") or "").strip().lower()
    return code, error_type, message


class IdeaExtractionAgent:
    def __init__(self, client_factory=None, transport=None):
        self.client_factory = client_factory or _create_gemini_client
        self.transport = transport

    async def _extract_gemini(
        self,
        idea: StartupIdea,
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredIdea:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ServiceError("GEMINI_API_KEY is not configured.", 503)

        gemini_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        request_started = time.monotonic()
        client = None
        async_client = None
        try:
            client = self.client_factory(api_key)
            async_client = client.aio
            response = await asyncio.wait_for(
                async_client.models.generate_content(
                    model=gemini_model,
                    contents=user_prompt,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                    },
                ),
                timeout=25.0,
            )
            content = response.text
            if not isinstance(content, str) or not content.strip():
                raise ServiceError("Gemini returned an empty response; no analysis was produced.")
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ServiceError("Gemini returned invalid JSON; no analysis was produced.") from exc
            if not isinstance(payload, dict):
                raise ServiceError(
                    "Gemini returned JSON that does not match the required idea fields; no analysis was produced."
                )
            try:
                return StructuredIdea.model_validate(payload)
            except ValidationError as exc:
                raise ServiceError(
                    "Gemini returned JSON that does not match the required idea fields; no analysis was produced."
                ) from exc
        except ServiceError:
            raise
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Gemini idea extraction request timed out: elapsed_seconds=%.3f exception_type=%s",
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError(f"Gemini idea extraction request timed out ({type(exc).__name__}).", 504) from exc
        except httpx.TimeoutException as exc:
            logger.warning(
                "Gemini idea extraction request timed out: elapsed_seconds=%.3f exception_type=%s",
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError(f"Gemini idea extraction request timed out ({type(exc).__name__}).", 504) from exc
        except httpx.RequestError as exc:
            logger.warning("Gemini idea extraction connection failed: %s", type(exc).__name__)
            raise ServiceError("Could not connect to Gemini for idea extraction; check network or proxy settings.", 502) from exc
        except Exception as exc:
            status = getattr(exc, "code", None)
            try:
                status = int(status)
            except (TypeError, ValueError):
                status = None
            provider_status = getattr(exc, "status", None)
            if not isinstance(provider_status, str):
                provider_status = "unknown"
            provider_status = re.sub(r"[^A-Za-z0-9_.-]", "", provider_status)[:60] or "unknown"
            safe_message = _safe_gemini_error_message(exc, api_key, [user_prompt, system_prompt])
            request_id = _safe_gemini_request_id(exc)
            logger.warning(
                "Gemini idea extraction provider error: code=%s status=%s type=%s "
                "request_id=%s elapsed_seconds=%.3f message=%s",
                status or "unknown",
                provider_status,
                type(exc).__name__,
                request_id or "unavailable",
                time.monotonic() - request_started,
                safe_message,
            )
            if status == 401:
                raise ServiceError("Gemini rejected the configured API credential.", 503) from exc
            if status == 403:
                raise ServiceError("Gemini denied access to idea extraction.", 503) from exc
            if status == 429:
                raise ServiceError("Gemini rate or quota limit reached during idea extraction.", 503) from exc
            if status == 400:
                raise ServiceError(
                    f"Gemini rejected the idea extraction request (HTTP 400): {safe_message}", 502
                ) from exc
            raise ServiceError(
                f"Gemini idea extraction provider request failed: {safe_message}", 502
            ) from exc
        finally:
            if async_client is not None:
                close_async = getattr(async_client, "aclose", None)
                if callable(close_async):
                    try:
                        await close_async()
                    except Exception:
                        logger.debug("Gemini async client close failed")
            if client is not None:
                close_sync = getattr(client, "close", None)
                if callable(close_sync):
                    try:
                        close_sync()
                    except Exception:
                        logger.debug("Gemini client close failed")

    async def extract(self, idea: StartupIdea) -> StructuredIdea:
        system_prompt = (
            "Return exactly one valid JSON object matching the requested fields. "
            "Do not use Markdown fences or add explanatory text."
        )
        user_prompt = (
            "Extract the startup idea into a JSON object with exactly these fields: "
            "problem (string or null), proposed_solution (string or null), "
            "target_customers (array of strings, possibly empty), industry (string or null), "
            "key_features (array of strings, possibly empty), value_proposition (string or null), "
            "assumptions_and_missing_information (array of strings). Use only facts explicitly "
            "supported by the submitted fields; do not add outside knowledge, market statistics, "
            "competitors, dates, or claims. If a field is absent or unsupported, use null or an "
            "empty array and name that exact field in assumptions_and_missing_information. List "
            "any inference that still needs user confirmation there as well. Return only the JSON "
            "object: no Markdown fences, preamble, or explanatory text. Do not transform or "
            "restate the submitted budget; the backend returns it separately with its original "
            "currency. Budget is not part of the extraction task and must not be included in "
            "assumptions or other extracted fields. Input: "
            + idea.model_dump_json(exclude={"estimated_budget", "estimated_budget_currency"})
        )

        return await execute_with_fallback(
            agent_name="idea extraction",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=StructuredIdea.model_validate,
            gemini_caller=lambda: self._extract_gemini(idea, system_prompt, user_prompt),
            transport=self.transport,
            groq_response_format={"type": "json_object"},
            sensitive_values=[user_prompt, system_prompt],
        )
