import json
import logging
import os
import re

import httpx
from pydantic import ValidationError

from app.schemas import (
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
)
from app.services.idea_extraction import ServiceError, _trust_env

logger = logging.getLogger(__name__)


def _competitor_response_format(source_urls: list[str]) -> dict:
    """Build Groq strict JSON Schema mode's response format for this analysis."""
    url_schema = {"type": "string", "enum": source_urls}
    competitor_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "competitor_type": {
                "type": "string",
                "enum": ["direct", "indirect", "unknown"],
            },
            "offering": {"type": "string"},
            "target_customers": {"type": "string"},
            "pricing": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": url_schema},
        },
        "required": [
            "name",
            "competitor_type",
            "offering",
            "target_customers",
            "pricing",
            "strengths",
            "limitations",
            "evidence",
        ],
        "additionalProperties": False,
    }
    gap_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "rationale": {"type": "string"},
            "gap_type": {
                "type": "string",
                "enum": ["evidence_based", "hypothesis"],
            },
            "evidence": {"type": "array", "items": url_schema},
        },
        "required": ["description", "rationale", "gap_type", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "competitor_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "competitors": {
                        "type": "array",
                        "items": competitor_schema,
                    },
                    "market_gaps": {"type": "array", "items": gap_schema},
                },
                "required": ["competitors", "market_gaps"],
                "additionalProperties": False,
            },
        },
    }


def _safe_provider_error(response: httpx.Response, api_key: str) -> str:
    """Return only the structured provider error, with credential-like text redacted."""
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return "provider response body was not valid JSON"

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        return "provider returned an unstructured error body"

    fields = []
    for name in ("type", "code", "message"):
        value = error.get(name)
        if not isinstance(value, (str, int, float)):
            continue
        text = str(value)
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
        text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
        text = re.sub(r"\bgsk_[A-Za-z0-9_-]+\b", "[REDACTED]", text)
        text = re.sub(r"\btvly-[A-Za-z0-9_-]+\b", "[REDACTED]", text)
        text = re.sub(
            r"(?i)(api[_ -]?key|token|secret)\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            text,
        )
        text = " ".join(text.split())[:500]
        if text:
            fields.append(f"{name}={text}")

    # Groq may include the rejected generation with json_validate_failed. It can
    # contain submitted idea/research text, so diagnose its format without
    # logging or returning the generation itself.
    if error.get("code") == "json_validate_failed":
        failed_generation = error.get("failed_generation")
        # Some provider errors use a structured diagnostic object. Only inspect
        # an attempted JSON string; never emit its content in application logs.
        if isinstance(failed_generation, dict):
            generated_text = failed_generation.get("attempted_arguments")
            detail_available = bool(failed_generation)
        else:
            generated_text = failed_generation
            detail_available = failed_generation is not None

        if isinstance(generated_text, str):
            if re.search(r"```", generated_text):
                diagnosis = "markdown_fences"
            else:
                try:
                    generated_value = json.loads(generated_text)
                except json.JSONDecodeError as exc:
                    diagnosis = (
                        f"invalid_json line={exc.lineno} column={exc.colno}"
                    )
                else:
                    root_type = (
                        "object" if isinstance(generated_value, dict)
                        else type(generated_value).__name__
                    )
                    diagnosis = f"valid_json_root={root_type}"
        else:
            diagnosis = "details_available_unparsed" if detail_available else "unavailable"
        fields.append(f"failed_generation_diagnosis={diagnosis}")
    return "; ".join(fields) or "provider returned no structured error details"


class CompetitorAnalysisAgent:
    def __init__(self, transport=None):
        self.transport = transport

    async def analyze(
        self,
        idea: StructuredIdea,
        research: dict[str, list[Source]],
        market_analysis: MarketOpportunityAnalysis | None = None,
    ) -> CompetitorAnalysis:
        sources = research.get("competitors_and_alternatives", [])

        # No competitor search results: do not make another LLM request.
        if not sources:
            return CompetitorAnalysis()

        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ServiceError("LLM_API_KEY is not configured.", 503)

        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        base_url = os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")

        source_data = [
            {
                "title": source.title,
                "url": source.url,
                "content": source.content,
            }
            for source in sources
        ]
        allowed_urls = sorted({source.url for source in sources})
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
                        "response_format": _competitor_response_format(allowed_urls),
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Return exactly one valid JSON object and nothing else. "
                                    "Do not use Markdown fences or surrounding text. "
                                    "Never fabricate facts or source URLs."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )

            if response.status_code == 401:
                logger.warning("Competitor analysis provider authentication failed: HTTP 401")
                raise ServiceError("Groq rejected the API credential during competitor analysis.", 503)
            if response.status_code == 403:
                logger.warning("Competitor analysis provider permission denied: HTTP 403")
                raise ServiceError("Groq denied access to competitor analysis.", 503)
            if response.status_code == 429:
                raise ServiceError("Groq rate limit reached.", 503)
            if response.status_code == 402:
                raise ServiceError("Groq account quota or billing limit prevents competitor analysis.", 503)
            if response.status_code == 400:
                provider_error = _safe_provider_error(response, api_key)
                logger.warning(
                    "Competitor analysis provider rejected request: HTTP 400; %s",
                    provider_error,
                )
                raise ServiceError("Groq rejected the competitor analysis request; check the model and request configuration.", 502)
            if response.status_code >= 400:
                logger.warning(
                    "Competitor analysis provider error: HTTP %s",
                    response.status_code,
                )
                raise ServiceError(
                    f"Groq returned HTTP {response.status_code} "
                    "during competitor analysis."
                )

            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ServiceError(
                    "Groq returned an invalid competitor analysis."
                )

            return CompetitorAnalysis.model_validate(
                json.loads(content), context={"allowed_urls": allowed_urls}
            )

        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise ServiceError(
                "Competitor analysis timed out.", 504
            ) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "Competitor analysis request failed: %s",
                type(exc).__name__,
            )
            raise ServiceError(
                "Could not connect to the LLM provider "
                "for competitor analysis."
            ) from exc
        except ValidationError as exc:
            logger.warning("Competitor analysis response failed schema or evidence validation")
            raise ServiceError(
                "Groq competitor analysis did not match the required schema or cited an unsupported source URL."
            ) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Invalid competitor analysis response: %s", type(exc).__name__)
            raise ServiceError(
                "Groq returned an invalid competitor analysis."
            ) from exc
