"""Evidence-grounded market opportunity and customer segmentation analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.provider_runner import (
    DEFAULT_GROQ_MODEL,
    GEMINI_MODEL,
    ServiceError,
    _create_gemini_client,
    _safe_gemini_error_message,
    _safe_gemini_request_id,
    _text_values,
    _trust_env,
    execute_with_fallback,
)

logger = logging.getLogger(__name__)
MARKET_RESEARCH_CATEGORIES = ("market_information", "industry_trends")


def _gemini_response_schema() -> dict:
    """Adapt the existing Pydantic schema to Gemini Developer API's schema subset."""
    schema = MarketOpportunityAnalysis.model_json_schema()

    def remove_unsupported_properties(value: Any):
        if isinstance(value, dict):
            value.pop("additionalProperties", None)
            for nested in value.values():
                remove_unsupported_properties(nested)
        elif isinstance(value, list):
            for nested in value:
                remove_unsupported_properties(nested)

    remove_unsupported_properties(schema)
    return schema


def _unknown_analysis() -> MarketOpportunityAnalysis:
    unknown = {"claim": "Unknown", "classification": "unknown", "evidence": []}
    return MarketOpportunityAnalysis(demand_assessment=unknown)


class MarketOpportunityAnalysisAgent:
    def __init__(self, client_factory=None, transport=None):
        self.client_factory = client_factory or _create_gemini_client
        self.transport = transport

    async def _analyze_gemini(
        self,
        idea: StructuredIdea,
        sources: list[Source],
        prompt: str,
        system_prompt: str,
        source_data: list[dict[str, str]],
        allowed_urls: list[str],
    ) -> MarketOpportunityAnalysis:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ServiceError("GEMINI_API_KEY is not configured.", 503)

        gemini_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        prompt_chars = len(prompt)
        system_chars = len(system_prompt)
        rough_input_tokens = math.ceil((prompt_chars + system_chars) / 4)
        logger.info(
            "Gemini market analysis request metadata: model=%s source_count=%d "
            "source_content_chars=%d max_source_content_chars=%d system_prompt_chars=%d "
            "user_prompt_chars=%d rough_input_tokens=%d max_completion_tokens=provider_default",
            gemini_model,
            len(sources),
            sum(len(source.content) for source in sources),
            max((len(source.content) for source in sources), default=0),
            system_chars,
            prompt_chars,
            rough_input_tokens,
        )

        client = None
        async_client = None
        request_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        request_started = time.monotonic()
        logger.info(
            "Gemini market analysis request started: model=%s request_started_utc=%s",
            gemini_model,
            request_started_utc,
        )
        try:
            client = self.client_factory(api_key)
            async_client = client.aio
            response = await asyncio.wait_for(
                async_client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                    },
                ),
                timeout=25.0,
            )
            logger.info(
                "Gemini market analysis response received: model=%s elapsed_seconds=%.3f",
                gemini_model,
                time.monotonic() - request_started,
            )
            content = response.text
            if not isinstance(content, str) or not content.strip():
                raise ServiceError("Gemini returned an empty market analysis response.")
            return MarketOpportunityAnalysis.model_validate(
                json.loads(content), context={"allowed_urls": allowed_urls}
            )
        except ServiceError:
            raise
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Gemini market analysis request timed out: model=%s elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                gemini_model,
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini market analysis request timed out.", 504) from exc
        except httpx.TimeoutException as exc:
            logger.warning(
                "Gemini market analysis request timed out: model=%s elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                gemini_model,
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini market analysis request timed out.", 504) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "Gemini market analysis connection failed: model=%s elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                gemini_model,
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Could not connect to Gemini for market analysis.", 502) from exc
        except ValidationError as exc:
            logger.warning("Gemini market analysis failed schema or evidence validation")
            raise ServiceError(
                "Gemini market analysis did not match the required schema or cited an unsupported source URL."
            ) from exc
        except json.JSONDecodeError as exc:
            logger.warning("Gemini returned invalid market analysis JSON")
            raise ServiceError("Gemini returned invalid market analysis JSON; no analysis was produced.") from exc
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
            safe_message = _safe_gemini_error_message(
                exc,
                api_key,
                [
                    prompt,
                    system_prompt,
                    *_text_values(source_data),
                    *_text_values(
                        idea.model_dump(
                            exclude={"estimated_budget", "estimated_budget_currency"}
                        )
                    ),
                ],
            )
            request_id = _safe_gemini_request_id(exc)
            logger.warning(
                "Gemini market analysis provider error: code=%s status=%s type=%s "
                "request_id=%s elapsed_seconds=%.3f model=%s message=%s",
                status or "unknown",
                provider_status,
                type(exc).__name__,
                request_id or "unavailable",
                time.monotonic() - request_started,
                gemini_model,
                safe_message,
            )
            if status == 401:
                raise ServiceError("Gemini rejected the configured API credential.", 503) from exc
            if status == 403:
                raise ServiceError("Gemini denied access to market analysis.", 503) from exc
            if status == 429:
                raise ServiceError("Gemini rate or quota limit reached during market analysis.", 503) from exc
            if status == 400:
                raise ServiceError(
                    f"Gemini rejected the market analysis request (HTTP 400): {safe_message}",
                    502,
                ) from exc
            raise ServiceError(
                f"Gemini market analysis provider request failed: {safe_message}", 502
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

    async def analyze(
        self,
        idea: StructuredIdea,
        research: dict[str, list[Source]],
    ) -> MarketOpportunityAnalysis:
        sources = [
            source
            for category in MARKET_RESEARCH_CATEGORIES
            for source in research.get(category, [])
        ]
        if not sources:
            return _unknown_analysis()

        allowed_urls = sorted({source.url for source in sources})
        if not allowed_urls:
            return _unknown_analysis()

        source_data = [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in sources
        ]

        system_prompt = (
            "Return exactly one valid JSON object matching the requested schema. "
            "Use only supplied evidence, never invent claims or URLs."
        )
        prompt = f"""Analyze market demand and customer segments for this startup using only the supplied research.

Startup idea:
{idea.model_dump_json()}

Research sources:
{json.dumps(source_data, ensure_ascii=False)}

Return one valid JSON object only: no Markdown, comments, preamble, or trailing text.
The object must have exactly these keys:
- demand_assessment: one claim object
- customer_segments: array of objects with segment (claim object) and needs (array of claim objects)
- opportunities: array of claim objects
- challenges: array of claim objects

Each claim object must have exactly:
- claim: a non-empty string; use exactly "Unknown" if unsupported
- classification: exactly "evidence_based", "hypothesis", or "unknown"
- evidence: an array containing only exact URLs from the supplied research

An evidence_based claim must cite at least one supplied URL. A hypothesis may have no URLs, and must be clearly framed as something to validate rather than a fact. For unknown details use claim "Unknown", classification "unknown", and evidence []. Do not infer customer demographics, needs, demand, or market size from outside the supplied idea and sources. Do not invent statistics, sources, publication dates, customer data, or factual claims. Do not include or reinterpret the submitted budget. If evidence is insufficient, state Unknown or return empty lists.
"""

        return await execute_with_fallback(
            agent_name="market analysis",
            system_prompt=system_prompt,
            user_prompt=prompt,
            validator=MarketOpportunityAnalysis.model_validate,
            validation_context={"allowed_urls": allowed_urls},
            gemini_caller=lambda: self._analyze_gemini(
                idea, sources, prompt, system_prompt, source_data, allowed_urls
            ),
            transport=self.transport,
            groq_response_format={"type": "json_object"},
            sensitive_values=[
                prompt,
                system_prompt,
                *_text_values(source_data),
                *_text_values(
                    idea.model_dump(
                        exclude={"estimated_budget", "estimated_budget_currency"}
                    )
                ),
            ],
        )
