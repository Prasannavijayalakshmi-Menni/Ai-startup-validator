"""Evidence-grounded SWOT and startup execution risk analysis using Gemini with Groq fallback."""

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

from app.schemas import (
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
    SWOTRiskAnalysis,
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


def _groq_response_format_swot() -> dict[str, Any]:
    """Use plain json_object mode for Groq SWOT/risk fallback.

    Groq's strict json_schema mode with enum-constrained evidence arrays causes
    HTTP 400 ("Generated JSON does not match the expected schema") when the model
    cannot produce a token that satisfies the URL enum constraint.  We use
    json_object mode instead and validate evidence URLs via Pydantic after parsing.
    """
    return {"type": "json_object"}


class SWOTRiskAnalysisAgent:
    def __init__(self, client_factory=None, transport=None):
        self.client_factory = client_factory or _create_gemini_client
        self.transport = transport

    async def _analyze_gemini(
        self,
        idea: StructuredIdea,
        market_analysis: MarketOpportunityAnalysis,
        competitor_analysis: CompetitorAnalysis,
        sources: list[Source],
        prompt: str,
        system_prompt: str,
        supplied_research: list[dict[str, str]],
        allowed_urls: list[str],
    ) -> SWOTRiskAnalysis:
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
                    contents=prompt,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                    },
                ),
                timeout=25.0,
            )
            content = response.text
            if not isinstance(content, str) or not content.strip():
                raise ServiceError("Gemini returned an empty SWOT/risk analysis response.")
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ServiceError("Gemini returned invalid SWOT/risk JSON; no analysis was produced.") from exc
            try:
                return SWOTRiskAnalysis.model_validate(
                    payload, context={"allowed_urls": allowed_urls}
                )
            except ValidationError as exc:
                raise ServiceError(
                    "Gemini SWOT/risk analysis did not match the required schema or cited an unsupported source URL."
                ) from exc
        except ServiceError:
            raise
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Gemini SWOT/risk analysis request timed out: elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini SWOT/risk analysis request timed out.", 504) from exc
        except httpx.TimeoutException as exc:
            logger.warning(
                "Gemini SWOT/risk analysis request timed out: elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini SWOT/risk analysis request timed out.", 504) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "Gemini SWOT/risk analysis connection failed: elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Could not connect to Gemini for SWOT/risk analysis.", 502) from exc
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
                [prompt, system_prompt, *_text_values(supplied_research)],
            )
            request_id = _safe_gemini_request_id(exc)
            logger.warning(
                "Gemini SWOT/risk analysis provider error: code=%s status=%s type=%s "
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
                raise ServiceError("Gemini denied access to SWOT/risk analysis.", 503) from exc
            if status == 429:
                raise ServiceError("Gemini rate or quota limit reached during SWOT/risk analysis.", 503) from exc
            if status == 400:
                raise ServiceError(
                    f"Gemini rejected the SWOT/risk analysis request (HTTP 400): {safe_message}", 502
                ) from exc
            raise ServiceError(
                f"Gemini SWOT/risk analysis provider request failed: {safe_message}", 502
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

    @staticmethod
    def _build_prompts(
        idea: StructuredIdea,
        market_analysis: MarketOpportunityAnalysis,
        competitor_analysis: CompetitorAnalysis,
        supplied_research: list[dict[str, str]],
        *,
        source_content_limit: int | None = None,
    ) -> tuple[str, str]:
        """Build system + user prompts. Pass source_content_limit to cap each
        source's content field for token-constrained fallback providers."""

        def _trim_claim(c: dict) -> dict:
            """Keep only claim and classification; drop evidence arrays to save tokens."""
            return {"claim": c.get("claim", ""), "classification": c.get("classification", "unknown")}

        def _compact_market(ma: MarketOpportunityAnalysis) -> dict:
            return {
                "demand_assessment": _trim_claim(ma.demand_assessment.model_dump()),
                "customer_segments": [
                    {
                        "segment": _trim_claim(seg.segment.model_dump()),
                        "needs": [_trim_claim(n.model_dump()) for n in seg.needs],
                    }
                    for seg in ma.customer_segments
                ],
                "opportunities": [_trim_claim(o.model_dump()) for o in ma.opportunities],
                "challenges": [_trim_claim(ch.model_dump()) for ch in ma.challenges],
            }

        def _compact_competitor(ca: CompetitorAnalysis) -> dict:
            return {
                "competitors": [
                    {
                        "name": c.name,
                        "competitor_type": c.competitor_type,
                        "offering": c.offering,
                        "target_customers": c.target_customers,
                        "pricing": c.pricing,
                        "strengths": c.strengths,
                        "limitations": c.limitations,
                    }
                    for c in ca.competitors
                ],
                "market_gaps": [
                    {"description": g.description, "gap_type": g.gap_type}
                    for g in ca.market_gaps
                ],
            }

        research_for_prompt = supplied_research
        if source_content_limit is not None:
            research_for_prompt = [
                {
                    "title": s["title"],
                    "url": s["url"],
                    "content": s["content"][:source_content_limit],
                }
                for s in supplied_research
            ]

        context = {
            "startup_idea": idea.model_dump(mode="json"),
            "market_analysis": _compact_market(market_analysis) if source_content_limit is not None else market_analysis.model_dump(mode="json"),
            "competitor_analysis": _compact_competitor(competitor_analysis) if source_content_limit is not None else competitor_analysis.model_dump(mode="json"),
            "research": research_for_prompt,
        }

        system_prompt = (
            "Return exactly one JSON object matching the requested schema. "
            "Use only supplied context and source URLs; do not invent facts."
        )
        user_prompt = f"""Prepare a SWOT and startup execution risk assessment from the supplied context only.

Context:
{json.dumps(context, ensure_ascii=False)}

Return one valid JSON object with exactly these keys, each an array:
- strengths
- weaknesses
- opportunities
- threats
- execution_risks

Each SWOT list item is a claim object with exactly:
- claim: non-empty string
- classification: exactly "evidence_based", "hypothesis", or "unknown"
- evidence: array of exact URLs from supplied research

Each execution_risks item has exactly:
- risk: claim object
- potential_impact: claim object

Evidence-based claims must cite at least one supplied source URL. Hypotheses must be phrased as possibilities to validate, not established facts, and may have an empty evidence array. When an item cannot be supported, use claim "Unknown", classification "unknown", and evidence []. Empty arrays are acceptable when no supported items can be identified.

Use only the submitted startup idea, supplied market and competitor analyses, and supplied research. Never invent facts, statistics, competitors, customer data, or citations. Every evidence URL must exactly match a URL in supplied research. Do not infer a fact from the absence of a search result. Explain potential impact in each risk item without presenting speculative consequences as fact.
"""
        return system_prompt, user_prompt

    # Groq TPM limit for the fallback model is 8000 tokens (~32000 chars).
    # Cap each source content snippet at 300 chars so the full prompt stays
    # well under that limit even with 15 sources (5 per Tavily category).
    _GROQ_SOURCE_CONTENT_LIMIT = 300

    async def analyze(
        self,
        idea: StructuredIdea,
        market_analysis: MarketOpportunityAnalysis,
        competitor_analysis: CompetitorAnalysis,
        research: dict[str, list[Source]],
    ) -> SWOTRiskAnalysis:
        sources = [source for group in research.values() for source in group]
        allowed_urls = sorted({source.url for source in sources})
        if not allowed_urls:
            return SWOTRiskAnalysis()

        supplied_research = [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in sources
        ]

        # Full-fidelity prompt for Gemini (no truncation).
        system_prompt, prompt = self._build_prompts(
            idea, market_analysis, competitor_analysis, supplied_research
        )
        # Compact prompt for Groq fallback: truncated source content +
        # stripped evidence arrays from market/competitor context.
        _, groq_prompt = self._build_prompts(
            idea, market_analysis, competitor_analysis, supplied_research,
            source_content_limit=self._GROQ_SOURCE_CONTENT_LIMIT,
        )

        return await execute_with_fallback(
            agent_name="SWOT/risk analysis",
            system_prompt=system_prompt,
            user_prompt=groq_prompt,
            validator=SWOTRiskAnalysis.model_validate,
            validation_context={"allowed_urls": allowed_urls},
            gemini_caller=lambda: self._analyze_gemini(
                idea, market_analysis, competitor_analysis, sources, prompt, system_prompt, supplied_research, allowed_urls
            ),
            transport=self.transport,
            groq_response_format=_groq_response_format_swot(),
            sensitive_values=[
                prompt,
                groq_prompt,
                system_prompt,
                *_text_values(supplied_research),
            ],
        )
