"""Evidence-grounded SWOT and startup execution risk analysis."""

import json
import logging
import os

import httpx
from pydantic import ValidationError

from app.schemas import (
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
    SWOTRiskAnalysis,
)
from app.services.idea_extraction import ServiceError, _trust_env

logger = logging.getLogger(__name__)


def _claim_schema(source_urls: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "classification": {
                "type": "string",
                "enum": ["evidence_based", "hypothesis", "unknown"],
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string", "enum": source_urls},
            },
        },
        "required": ["claim", "classification", "evidence"],
        "additionalProperties": False,
    }


def _swot_response_format(source_urls: list[str]) -> dict:
    claim = _claim_schema(source_urls)
    risk = {
        "type": "object",
        "properties": {
            "risk": claim,
            "potential_impact": claim,
        },
        "required": ["risk", "potential_impact"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "strengths": {"type": "array", "items": claim},
            "weaknesses": {"type": "array", "items": claim},
            "opportunities": {"type": "array", "items": claim},
            "threats": {"type": "array", "items": claim},
            "execution_risks": {"type": "array", "items": risk},
        },
        "required": [
            "strengths",
            "weaknesses",
            "opportunities",
            "threats",
            "execution_risks",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "swot_risk_analysis",
            "strict": True,
            "schema": schema,
        },
    }


class SWOTRiskAnalysisAgent:
    def __init__(self, transport=None):
        self.transport = transport

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

        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ServiceError("LLM_API_KEY is not configured.", 503)
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        base_url = os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")

        supplied_research = [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in sources
        ]
        context = {
            "startup_idea": idea.model_dump(mode="json"),
            "market_analysis": market_analysis.model_dump(mode="json"),
            "competitor_analysis": competitor_analysis.model_dump(mode="json"),
            "research": supplied_research,
        }
        prompt = f"""Prepare a SWOT and startup execution risk assessment from the supplied context only.

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
                        "response_format": _swot_response_format(allowed_urls),
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Return exactly one JSON object matching the requested schema. "
                                    "Use only supplied context and source URLs; do not invent facts."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )

            if response.status_code == 401:
                logger.warning("SWOT/risk provider authentication failed: HTTP 401")
                raise ServiceError("Groq rejected the API credential during SWOT/risk analysis.", 503)
            if response.status_code == 403:
                logger.warning("SWOT/risk provider permission denied: HTTP 403")
                raise ServiceError("Groq denied access to SWOT/risk analysis.", 503)
            if response.status_code == 429:
                raise ServiceError("Groq rate limit reached during SWOT/risk analysis.", 503)
            if response.status_code == 402:
                raise ServiceError("Groq account quota or billing limit prevents SWOT/risk analysis.", 503)
            if response.status_code >= 400:
                logger.warning("SWOT/risk provider returned HTTP %s", response.status_code)
                raise ServiceError(
                    f"Groq rejected the SWOT/risk analysis request (HTTP {response.status_code}).",
                    502,
                )

            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ServiceError("Groq returned an invalid SWOT/risk analysis response.")
            return SWOTRiskAnalysis.model_validate(
                json.loads(content), context={"allowed_urls": allowed_urls}
            )
        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("SWOT/risk provider timed out: %s", type(exc).__name__)
            raise ServiceError("SWOT/risk analysis provider request timed out.", 504) from exc
        except httpx.RequestError as exc:
            logger.warning("SWOT/risk provider connection failed: %s", type(exc).__name__)
            raise ServiceError("Could not connect to the LLM provider for SWOT/risk analysis.", 502) from exc
        except ValidationError as exc:
            logger.warning("SWOT/risk response failed schema or evidence validation")
            raise ServiceError(
                "Groq SWOT/risk analysis did not match the required schema or cited an unsupported source URL."
            ) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Invalid SWOT/risk response: %s", type(exc).__name__)
            raise ServiceError("Groq returned invalid SWOT/risk JSON; no analysis was produced.") from exc
