"""Evidence-grounded market opportunity and customer segmentation analysis."""

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.idea_extraction import ServiceError, _trust_env

logger = logging.getLogger(__name__)
MARKET_RESEARCH_CATEGORIES = ("market_information", "industry_trends")
GEMINI_MODEL = "gemini-3.8-flash"


def _text_values(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _text_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_values(item)]
    return []


def _safe_gemini_error_message(exc, api_key: str, sensitive_values: list[str]) -> str:
    """Return bounded provider diagnostics with known secrets and prompt data removed."""
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message.strip():
        return "provider did not include an error message"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    for value in sensitive_values:
        if value and len(value) >= 4:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key|token|secret|authorization)\s*[:=]\s*[^,;\s]+",
        r"\1=[REDACTED]",
        message,
    )
    message = re.sub(r"https?://\S+", "[REDACTED_URL]", message)
    return " ".join(message.split())[:400] or "provider did not include an error message"


def _safe_gemini_request_id(exc) -> str | None:
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


def _gemini_response_schema() -> dict:
    """Adapt the existing Pydantic schema to Gemini Developer API's schema subset."""
    schema = MarketOpportunityAnalysis.model_json_schema()

    def remove_unsupported_properties(value):
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
    def __init__(self, client_factory=None):
        self.client_factory = client_factory or _create_gemini_client

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

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ServiceError("GEMINI_API_KEY is not configured.", 503)
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
        prompt_chars = len(prompt)
        system_chars = len(system_prompt)
        rough_input_tokens = math.ceil((prompt_chars + system_chars) / 4)
        logger.info(
            "Gemini market analysis request metadata: model=%s source_count=%d "
            "source_content_chars=%d max_source_content_chars=%d system_prompt_chars=%d "
            "user_prompt_chars=%d rough_input_tokens=%d max_completion_tokens=provider_default",
            GEMINI_MODEL,
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
            GEMINI_MODEL,
            request_started_utc,
        )
        try:
            client = self.client_factory(api_key)
            async_client = client.aio
            response = await asyncio.wait_for(
                async_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "system_instruction": system_prompt,
                        "response_mime_type": "application/json",
                        "response_schema": _gemini_response_schema(),
                    },
                ),
                timeout=30.0,
            )
            logger.info(
                "Gemini market analysis response received: model=%s elapsed_seconds=%.3f",
                GEMINI_MODEL,
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
                GEMINI_MODEL,
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini market analysis request timed out.", 504) from exc
        except httpx.TimeoutException as exc:
            logger.warning(
                "Gemini market analysis request timed out: model=%s elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                GEMINI_MODEL,
                time.monotonic() - request_started,
                type(exc).__name__,
            )
            raise ServiceError("Gemini market analysis request timed out.", 504) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "Gemini market analysis connection failed: model=%s elapsed_seconds=%.3f "
                "exception_type=%s provider_http_status=unavailable",
                GEMINI_MODEL,
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
                GEMINI_MODEL,
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
