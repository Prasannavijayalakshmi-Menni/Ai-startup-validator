"""Evidence-grounded market opportunity and customer segmentation analysis."""

import json
import logging
import math
import os
import re

import httpx
from pydantic import ValidationError

from app.schemas import (
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
)
from app.services.idea_extraction import ServiceError, _trust_env

logger = logging.getLogger(__name__)
MARKET_RESEARCH_CATEGORIES = ("market_information", "industry_trends")


def _market_response_format() -> dict:
    """Ask Groq for JSON syntax; enforce the full schema locally with Pydantic."""
    return {"type": "json_object"}


def _text_values(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _text_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_values(item)]
    return []


def _provider_error_metadata(
    response: httpx.Response,
    api_key: str,
    sensitive_texts: list[str],
) -> tuple[str, str, str, bool, int | None]:
    """Extract provider error metadata without exposing credentials or prompts."""
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        error = {}

    error_type = re.sub(r"[^A-Za-z0-9_.-]", "", str(error.get("type") or ""))[:60]
    code = re.sub(r"[^A-Za-z0-9_.-]", "", str(error.get("code") or ""))[:60]
    message = str(error.get("message") or "").strip()
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    for value in sensitive_texts:
        if value and len(value) >= 8:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bgsk_[A-Za-z0-9_-]+\b|\btvly-[A-Za-z0-9_-]+\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key|token|secret)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        message,
    )
    message = " ".join(message.split())[:400]

    failed_generation = error.get("failed_generation")
    if failed_generation is None:
        failed_generation_length = None
    elif isinstance(failed_generation, str):
        failed_generation_length = len(failed_generation)
    else:
        failed_generation_length = len(json.dumps(failed_generation, ensure_ascii=True))
    return (
        error_type,
        code,
        message,
        failed_generation is not None,
        failed_generation_length,
    )


def _unknown_analysis() -> MarketOpportunityAnalysis:
    unknown = {"claim": "Unknown", "classification": "unknown", "evidence": []}
    return MarketOpportunityAnalysis(demand_assessment=unknown)


class MarketOpportunityAnalysisAgent:
    def __init__(self, transport=None):
        self.transport = transport

    async def analyze(
        self,
        idea: StructuredIdea,
        research: dict[str, list[Source]],
    ) -> MarketOpportunityAnalysis:
        # Competitor alternatives are passed to their dedicated downstream agent.
        # Market demand and trend claims use their corresponding Tavily categories.
        sources = [
            source
            for category in MARKET_RESEARCH_CATEGORIES
            for source in research.get(category, [])
        ]
        if not sources:
            return _unknown_analysis()

        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ServiceError("LLM_API_KEY is not configured.", 503)
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        base_url = os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        source_data = [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in sources
        ]
        allowed_urls = sorted({source.url for source in sources})
        if not allowed_urls:
            return _unknown_analysis()

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

        system_prompt = (
            "Return exactly one valid JSON object matching the requested schema. "
            "Use only supplied evidence, never invent claims or URLs."
        )
        source_content_lengths = [len(source.content) for source in sources]
        prompt_chars = len(prompt)
        system_chars = len(system_prompt)
        # This is a rough planning estimate, not a provider tokenizer count.
        rough_input_tokens = math.ceil((prompt_chars + system_chars) / 4)
        logger.info(
            "Market analysis request metadata: model=%s response_format=json_object "
            "source_count=%d source_content_chars=%d max_source_content_chars=%d "
            "system_prompt_chars=%d user_prompt_chars=%d rough_input_tokens=%d "
            "max_completion_tokens=provider_default",
            model,
            len(sources),
            sum(source_content_lengths),
            max(source_content_lengths, default=0),
            system_chars,
            prompt_chars,
            rough_input_tokens,
        )

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                transport=self.transport,
                trust_env=_trust_env(),
            ) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "temperature": 0,
                        "response_format": _market_response_format(),
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )

            if response.status_code >= 400:
                sensitive_texts = [
                    prompt,
                    *_text_values(source_data),
                    *_text_values(
                        idea.model_dump(
                            exclude={"estimated_budget", "estimated_budget_currency"}
                        )
                    ),
                ]
                error_type, code, message, has_failed_generation, failed_generation_length = (
                    _provider_error_metadata(response, api_key, sensitive_texts)
                )
                logger.warning(
                    "Market analysis provider error: status=%s type=%s code=%s "
                    "message=%s failed_generation_present=%s failed_generation_chars=%s",
                    response.status_code,
                    error_type or "unknown",
                    code or "unknown",
                    message or "unavailable",
                    has_failed_generation,
                    failed_generation_length,
                )
            if response.status_code == 401:
                logger.warning("Market analysis provider authentication failed: HTTP 401")
                raise ServiceError("Groq rejected the API credential during market analysis.", 503)
            if response.status_code == 403:
                logger.warning("Market analysis provider permission denied: HTTP 403")
                raise ServiceError("Groq denied access to market analysis.", 503)
            if response.status_code == 429:
                raise ServiceError("Groq rate limit reached during market analysis.", 503)
            if response.status_code == 402:
                raise ServiceError("Groq account quota or billing limit prevents market analysis.", 503)
            if response.status_code >= 400:
                raise ServiceError(
                    f"Groq rejected the market analysis request (HTTP {response.status_code}).",
                    502,
                )

            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ServiceError("Groq returned an invalid market analysis response.")
            return MarketOpportunityAnalysis.model_validate(
                json.loads(content), context={"allowed_urls": allowed_urls}
            )
        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Market analysis provider timed out: %s", type(exc).__name__)
            raise ServiceError("Market analysis provider request timed out.", 504) from exc
        except httpx.RequestError as exc:
            logger.warning("Market analysis provider connection failed: %s", type(exc).__name__)
            raise ServiceError("Could not connect to the LLM provider for market analysis.", 502) from exc
        except ValidationError as exc:
            logger.warning("Market analysis response failed schema or evidence validation")
            raise ServiceError(
                "Groq market analysis did not match the required schema or cited an unsupported source URL."
            ) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Invalid market analysis response: %s", type(exc).__name__)
            raise ServiceError("Groq returned invalid market analysis JSON; no analysis was produced.") from exc
