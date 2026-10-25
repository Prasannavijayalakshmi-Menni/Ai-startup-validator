"""Tests for SWOTRiskAnalysisAgent using the Gemini SDK mock pattern."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from app.schemas import (
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
    SWOTRiskAnalysis,
)
from app.services.idea_extraction import ServiceError
from app.services.market_analysis import GEMINI_MODEL
from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent

SOURCE_URL = "https://research.example/startup"
IDEA = StructuredIdea(
    problem="Small clinics need simpler appointment workflows",
    proposed_solution="A lightweight appointment assistant",
    target_customers=["Small clinics"],
    industry="Healthcare",
    key_features=["Appointment support"],
    value_proposition="Simpler appointment handling",
    assumptions_and_missing_information=[],
)
RESEARCH = {
    "market_information": [
        Source(
            title="Clinic workflow overview",
            url=SOURCE_URL,
            content="The excerpt describes workflow challenges for clinics.",
        )
    ],
    "industry_trends": [],
}
MARKET_ANALYSIS = MarketOpportunityAnalysis(
    demand_assessment={
        "claim": "Unknown",
        "classification": "unknown",
        "evidence": [],
    }
)
COMPETITOR_ANALYSIS = CompetitorAnalysis()


def valid_result():
    evidence = {
        "claim": "The supplied excerpt describes a clinic workflow challenge.",
        "classification": "evidence_based",
        "evidence": [SOURCE_URL],
    }
    hypothesis = {
        "claim": "The team may need to validate willingness to adopt a new tool.",
        "classification": "hypothesis",
        "evidence": [],
    }
    return {
        "strengths": [evidence],
        "weaknesses": [hypothesis],
        "opportunities": [],
        "threats": [],
        "execution_risks": [
            {
                "risk": hypothesis,
                "potential_impact": {
                    "claim": "Adoption delays could slow validation; this is a hypothesis.",
                    "classification": "hypothesis",
                    "evidence": [],
                },
            }
        ],
    }


@pytest.fixture(autouse=True)
def configured_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-swot-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)


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
        c = FakeGeminiClient(response=response, error=error, captured=captured)
        clients.append(c)
        return c

    return SWOTRiskAnalysisAgent(client_factory=factory), captured, clients


@pytest.mark.asyncio
async def test_swot_risk_agent_returns_valid_claims_and_uses_all_context():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(valid_result()), captured=captured)

    analysis = await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)

    assert isinstance(analysis, SWOTRiskAnalysis)
    assert analysis.strengths[0].evidence == [SOURCE_URL]
    assert analysis.weaknesses[0].classification == "hypothesis"
    assert analysis.execution_risks[0].potential_impact.classification == "hypothesis"

    assert captured["api_key_received"] == "mock-swot-key"
    req = captured["request"]
    assert req["model"] == GEMINI_MODEL
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["config"]["system_instruction"]
    prompt = req["contents"]
    assert "competitor_analysis" in prompt
    assert "market_analysis" in prompt
    assert SOURCE_URL in prompt
    assert "budget" not in prompt.lower()
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_research_returns_empty_safe_result_without_gemini_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def unexpected_factory(api_key):
        raise AssertionError("unexpected provider call")

    agent = SWOTRiskAnalysisAgent(client_factory=unexpected_factory)
    result = await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, {})
    assert result.strengths == []
    assert result.execution_risks == []


def test_swot_schema_rejects_missing_or_unsupported_evidence():
    payload = valid_result()
    payload["strengths"][0]["evidence"] = []
    with pytest.raises(ValidationError, match="must cite at least one"):
        SWOTRiskAnalysis.model_validate(payload, context={"allowed_urls": {SOURCE_URL}})

    payload = valid_result()
    payload["execution_risks"][0]["risk"]["evidence"] = [
        "https://outside.example/unsupported"
    ]
    with pytest.raises(ValidationError, match="absent from supplied research"):
        SWOTRiskAnalysis.model_validate(payload, context={"allowed_urls": {SOURCE_URL}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({**valid_result(), "extra": True}),
        json.dumps({**valid_result(), "strengths": "not a list"}),
        json.dumps({**valid_result(), "execution_risks": [{"risk": "missing impact"}]}),
    ],
)
async def test_invalid_swot_responses_fail_without_returning_fake_analysis(content):
    agent, _, _ = make_agent(content)
    with pytest.raises(ServiceError, match="SWOT/risk"):
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)


@pytest.mark.asyncio
async def test_missing_gemini_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ServiceError, match="GEMINI_API_KEY"):
        await SWOTRiskAnalysisAgent().analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected,status_code",
    [
        (401, "credential", 503),
        (403, "denied", 503),
        (429, "rate or quota", 503),
        (400, "HTTP 400", 502),
        (500, "provider request failed", 502),
    ],
)
async def test_swot_provider_errors_are_categorized(status, expected, status_code):
    error = RuntimeError("provider failure mock-swot-key")
    error.code = status
    error.message = "provider failure mock-swot-key"
    agent, _, _ = make_agent(error=error)

    with pytest.raises(ServiceError, match=expected) as raised:
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)

    assert raised.value.status_code == status_code


@pytest.mark.asyncio
async def test_swot_provider_connection_and_timeout_errors_are_distinct():
    agent, _, _ = make_agent(error=httpx.ConnectError("offline"))
    with pytest.raises(ServiceError, match="Could not connect") as exc:
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)
    assert exc.value.status_code == 502

    agent, _, _ = make_agent(error=httpx.ReadTimeout("timeout"))
    with pytest.raises(ServiceError, match="timed out") as exc:
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_swot_asyncio_timeout_returns_504():
    agent, _, _ = make_agent(error=asyncio.TimeoutError())
    with pytest.raises(ServiceError, match="timed out") as exc:
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_swot_empty_response_is_rejected():
    agent, _, _ = make_agent("")
    with pytest.raises(ServiceError, match="empty SWOT/risk"):
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)


@pytest.mark.asyncio
async def test_swot_unsupported_citation_is_rejected():
    invalid = valid_result()
    invalid["strengths"][0]["evidence"] = ["https://outside.example/not-supplied"]
    agent, _, _ = make_agent(json.dumps(invalid))
    with pytest.raises(ServiceError, match="unsupported source URL"):
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)


@pytest.mark.asyncio
async def test_swot_aclose_called_after_success():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(valid_result()), captured=captured)
    await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_swot_credentials_are_not_logged(caplog):
    error = RuntimeError("sensitive message mock-swot-key")
    error.code = 500
    error.message = "sensitive message mock-swot-key"
    agent, _, _ = make_agent(error=error)
    caplog.set_level(logging.WARNING, logger="app.services.swot_risk_analysis")

    with pytest.raises(ServiceError):
        await agent.analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)

    assert "mock-swot-key" not in caplog.text
