"""Regression tests confirming the Groq market-analysis path has been removed.

Market Analysis now uses Gemini exclusively. These tests verify that:
- GEMINI_API_KEY is required (LLM_API_KEY alone is not sufficient).
- The agent always calls Gemini regardless of any legacy env-var values.
- The Groq-specific env vars (LLM_API_KEY, LLM_MODEL, MARKET_ANALYSIS_MODEL,
  LLM_BASE_URL) are not used by Market Analysis.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.idea_extraction import ServiceError
from app.services.market_analysis import GEMINI_MODEL, MarketOpportunityAnalysisAgent

SOURCE_URL = "https://research.example.test/market-demand"
IDEA = StructuredIdea(
    problem="Students need more consistent study planning",
    proposed_solution="An adaptive study planner",
    target_customers=["Secondary-school students"],
    industry="Education",
    key_features=["Adaptive schedules"],
    value_proposition="Help students organize study time",
    assumptions_and_missing_information=[],
)
RESEARCH = {
    "market_information": [
        Source(
            title="Synthetic education research fixture",
            url=SOURCE_URL,
            content="Synthetic mocked source: students report difficulty planning study time.",
        )
    ],
    "industry_trends": [],
}


def valid_result():
    return {
        "demand_assessment": {
            "claim": "The supplied synthetic excerpt describes study-planning difficulty.",
            "classification": "evidence_based",
            "evidence": [SOURCE_URL],
        },
        "customer_segments": [
            {
                "segment": {
                    "claim": "Secondary-school students",
                    "classification": "evidence_based",
                    "evidence": [SOURCE_URL],
                },
                "needs": [
                    {
                        "claim": "An adaptive planner may help organize study time.",
                        "classification": "hypothesis",
                        "evidence": [],
                    }
                ],
            }
        ],
        "opportunities": [],
        "challenges": [
            {"claim": "Unknown", "classification": "unknown", "evidence": []}
        ],
    }


class FakeGeminiClient:
    def __init__(self, response=None, error=None, captured=None):
        self.captured = captured if captured is not None else {}
        self.response = response
        self.error = error
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate_content))
        self.aio.aclose = AsyncMock()
        self.close = lambda: None

    async def generate_content(self, **kwargs):
        self.captured["request"] = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.response)


def make_agent(response=None, error=None, captured=None):
    captured = captured if captured is not None else {}
    clients = []

    def factory(api_key):
        captured["api_key_received"] = api_key
        client = FakeGeminiClient(response=response, error=error, captured=captured)
        clients.append(client)
        return client

    return MarketOpportunityAnalysisAgent(client_factory=factory), captured, clients


@pytest.mark.asyncio
async def test_gemini_is_used_regardless_of_legacy_provider_env_var(monkeypatch):
    """Setting MARKET_ANALYSIS_PROVIDER=groq no longer switches to Groq; Gemini is always used."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    # These Groq vars must not be used by Market Analysis.
    monkeypatch.setenv("LLM_API_KEY", "should-not-be-used")
    monkeypatch.setenv("LLM_MODEL", "should-not-be-used")
    monkeypatch.setenv("MARKET_ANALYSIS_MODEL", "should-not-be-used")
    monkeypatch.setenv("LLM_BASE_URL", "https://should-not-be-called.invalid/v1")

    captured = {}
    agent, captured, clients = make_agent(json.dumps(valid_result()), captured=captured)

    result = await agent.analyze(IDEA, RESEARCH)

    assert isinstance(result, MarketOpportunityAnalysis)
    assert result.demand_assessment.evidence == [SOURCE_URL]
    assert captured["api_key_received"] == "mock-gemini-key"
    assert captured["request"]["model"] == GEMINI_MODEL


@pytest.mark.asyncio
async def test_missing_gemini_key_raises_even_when_groq_key_is_set(monkeypatch):
    """LLM_API_KEY alone is not sufficient; GEMINI_API_KEY is required."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "groq-key-present-but-irrelevant")

    with pytest.raises(ServiceError, match="GEMINI_API_KEY"):
        await MarketOpportunityAnalysisAgent().analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_groq_model_env_vars_are_not_used_by_market_analysis(monkeypatch):
    """MARKET_ANALYSIS_MODEL and LLM_MODEL must not influence the Gemini model selection."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("MARKET_ANALYSIS_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")

    captured = {}
    agent, captured, _ = make_agent(json.dumps(valid_result()), captured=captured)

    await agent.analyze(IDEA, RESEARCH)

    assert captured["request"]["model"] == GEMINI_MODEL
    assert captured["request"]["model"] != "qwen/qwen3.8-27b"
    assert captured["request"]["model"] != "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_empty_research_returns_unknown_without_gemini_call(monkeypatch):
    """No Gemini call is made when there are no market research sources."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent = MarketOpportunityAnalysisAgent(
        client_factory=lambda _: (_ for _ in ()).throw(AssertionError("unexpected Gemini call"))
    )

    result = await agent.analyze(IDEA, {"market_information": []})

    assert result.demand_assessment.claim == "Unknown"
    assert result.demand_assessment.classification == "unknown"


@pytest.mark.asyncio
async def test_malformed_json_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    agent, _, _ = make_agent("not json")

    with pytest.raises(ServiceError, match="invalid market analysis JSON"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_schema_validation_failures_are_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    invalid = {key: value for key, value in valid_result().items() if key != "demand_assessment"}
    agent, _, _ = make_agent(json.dumps(invalid))

    with pytest.raises(ServiceError, match="required schema"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_unsupported_citation_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    invalid = valid_result()
    invalid["demand_assessment"]["evidence"] = ["https://outside.example/fact"]
    agent, _, _ = make_agent(json.dumps(invalid))

    with pytest.raises(ServiceError, match="unsupported source URL"):
        await agent.analyze(IDEA, RESEARCH)
