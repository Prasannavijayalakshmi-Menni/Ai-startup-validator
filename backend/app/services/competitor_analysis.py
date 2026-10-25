"""Evidence-grounded competitor analysis using Google Gemini with Groq fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.schemas import (
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
)
from app.services.provider_runner import (
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


def _groq_response_format_competitor() -> dict[str, Any]:
    """Use plain json_object mode for Groq competitor analysis.

    Groq's strict json_schema mode with enum-constrained evidence arrays causes
    HTTP 400 ("Generated JSON does not match the expected schema") when the model
    cannot find a token that satisfies the URL enum.  We use json_object mode
    instead and validate evidence URLs via Pydantic after parsing.
    """
    return {"type": "json_object"}


class CompetitorAnalysisAgent:
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
    ) -> CompetitorAnalysis:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ServiceError("GEMINI_API_KEY is not configured.", 503)

        gemini_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        request_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        request_started = time.monotonic()
        logger.info(
            "Gemini competitor analysis request started: model=%s request_started_utc=%s "
            "source_count=%d",
            gemini_model,
            request_started_utc,
            len(sources),
        )

        client = None
        async_client = None
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
                "Gemini competitor analysis response received: model=%s elapsed_seconds=%.3f",
                gemini_model,
                time.monotonic() - request_started,
            )
            content = response.text
            if not isinstance(content, str) or not content.strip():
                raise ServiceError("Gemini returned an empty competitor analysis response.")
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ServiceError(
                    "Gemini returned invalid competitor analysis JSON; no analysis was produced."
                ) from exc
            try:
                return CompetitorAnalysis.model_validate(
                    payload, context={"allowed_urls": allowed_urls}
                )
            except ValidationError as exc:
                raise ServiceError(
                    "Gemini competitor analysis did not match the required schema or cited an unsupported source URL."
                ) from exc
        except ServiceError:
            raise
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Gemini competitor analysis request timed out: model=%s elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                gemini_model,
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini competitor analysis request timed out.", 504) from exc
        except Exception as exc:
            import httpx

            if isinstance(exc, httpx.TimeoutException):
                logger.warning(
                    "Gemini competitor analysis request timed out: model=%s elapsed_seconds=%.3f "
                    "exception_type=%s provider_http_status=unavailable",
                    gemini_model,
                    time.monotonic() - request_started,
                    type(exc).__name__,
                )
                raise ServiceError("Gemini competitor analysis request timed out.", 504) from exc
            if isinstance(exc, httpx.RequestError):
                logger.warning(
                    "Gemini competitor analysis connection failed: model=%s elapsed_seconds=%.3f "
                    "exception_type=%s provider_http_status=unavailable",
                    gemini_model,
                    time.monotonic() - request_started,
                    type(exc).__name__,
                )
                raise ServiceError("Could not connect to Gemini for competitor analysis.", 502) from exc

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
                    *_text_values(idea.model_dump(exclude={"estimated_budget", "estimated_budget_currency"})),
                ],
            )
            request_id = _safe_gemini_request_id(exc)
            logger.warning(
                "Gemini competitor analysis provider error: code=%s status=%s type=%s "
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
                raise ServiceError("Gemini denied access to competitor analysis.", 503) from exc
            if status == 429:
                raise ServiceError("Gemini rate or quota limit reached during competitor analysis.", 503) from exc
            if status == 400:
                raise ServiceError(
                    f"Gemini rejected the competitor analysis request (HTTP 400): {safe_message}",
                    502,
                ) from exc
            raise ServiceError(
                f"Gemini competitor analysis provider request failed: {safe_message}", 502
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
        market_analysis: MarketOpportunityAnalysis | None = None,
    ) -> CompetitorAnalysis:
        sources = research.get("competitors_and_alternatives", [])
        if not sources:
            return CompetitorAnalysis()

        allowed_urls = sorted({source.url for source in sources})
        if not allowed_urls:
            return CompetitorAnalysis()

        source_data = [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in sources
        ]
        market_context = (
            market_analysis.model_dump(mode="json")
            if market_analysis is not None
            else {
                "demand_assessment": {
                    "claim": "Unknown",
                    "classification": "unknown",
                    "evidence": [],
                },
                "customer_segments": [],
                "opportunities": [],
                "challenges": [],
            }
        )

        system_prompt = (
            "Return exactly one valid JSON object and nothing else. "
            "Do not use Markdown fences or surrounding text. "
            "Never fabricate facts or source URLs."
        )
        prompt = f"""
Analyze the startup idea and the supplied competitor research.

Startup idea:
{idea.model_dump_json()}

Market opportunity and customer analysis:
{json.dumps(market_context, ensure_ascii=False)}

Research sources:
{json.dumps(source_data, ensure_ascii=False)}

Return exactly one syntactically valid JSON object. Do not include Markdown
fences, comments, a preamble, or any text before or after the JSON object. Use
double quotes for every JSON key and string; do not use trailing commas.

The object must have exactly these top-level keys and value types:
- "competitors": array of competitor objects
- "market_gaps": array of market gap objects

Every competitor object must have exactly these keys and types:
- "name": non-empty string
- "competitor_type": exactly "direct", "indirect", or "unknown"
- "offering": string; use "Unknown" when unsupported
- "target_customers": string; use "Unknown" when unsupported
- "pricing": string; use exactly "Unknown" when pricing is unsupported
- "strengths": array of strings (use [] when unsupported)
- "limitations": array of strings (use [] when unsupported)
- "evidence": array of source URL strings, with at least one supplied URL

Every market gap object must have exactly these keys and types:
- "description": non-empty string
- "rationale": non-empty string
- "gap_type": exactly "evidence_based" or "hypothesis"
- "evidence": array of supplied source URL strings; use [] when the gap is a
  hypothesis without direct supporting evidence. An "evidence_based" gap must
  have at least one evidence URL.

All list fields must be JSON arrays, including empty lists. Do not omit fields
or add keys. If no entries are supported, return this valid empty result:
{{"competitors":[],"market_gaps":[]}}

Rules:
- Use only the supplied sources, startup idea, and market analysis.
- Use the market analysis to compare competitor relevance to its identified customer segments, needs, and opportunities. Treat market-analysis hypotheses as hypotheses, not evidence.
- Classify a competitor as direct only when the supplied evidence supports a similar offering for substantially the same customer need. Classify it as indirect when the evidence supports an alternative approach to the same or a related need. Use "unknown" when the research does not support either classification.
- Never invent competitor names, prices, features, or citations.
- If pricing is not supported by a source, use "Unknown".
- Every evidence URL must exactly match a supplied source URL.
- Distinguish verified source information from your own hypotheses.
- A potential gap is not proof that no competitor offers that feature.
- If evidence is insufficient, return empty lists rather than inventing claims.
- Do not include the submitted budget in this analysis.
"""

        return await execute_with_fallback(
            agent_name="competitor analysis",
            system_prompt=system_prompt,
            user_prompt=prompt,
            validator=CompetitorAnalysis.model_validate,
            validation_context={"allowed_urls": allowed_urls},
            gemini_caller=lambda: self._analyze_gemini(
                idea, sources, prompt, system_prompt, source_data, allowed_urls
            ),
            transport=self.transport,
            groq_response_format=_groq_response_format_competitor(),
            sensitive_values=[
                prompt,
                system_prompt,
                *_text_values(source_data),
                *_text_values(idea.model_dump(exclude={"estimated_budget", "estimated_budget_currency"})),
            ],
        )
