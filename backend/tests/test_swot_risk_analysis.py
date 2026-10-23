import json

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
from app.services.swot_risk_analysis import (
    SWOTRiskAnalysisAgent,
    _swot_response_format,
)

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
def configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "mock-swot-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")


def response_transport(payload, status=200):
    return httpx.MockTransport(
        lambda request: httpx.Response(status, json=payload, request=request)
    )


def completion(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_swot_risk_agent_returns_valid_claims_and_uses_all_context():
    result = valid_result()
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(json.dumps(result)), request=request)

    analysis = await SWOTRiskAnalysisAgent(
        transport=httpx.MockTransport(handler)
    ).analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)

    assert isinstance(analysis, SWOTRiskAnalysis)
    assert analysis.strengths[0].evidence == [SOURCE_URL]
    assert analysis.weaknesses[0].classification == "hypothesis"
    assert analysis.execution_risks[0].potential_impact.classification == "hypothesis"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-oss-20b"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["response_format"] == _swot_response_format([SOURCE_URL])
    prompt = captured["body"]["messages"][1]["content"]
    assert "competitor_analysis" in prompt
    assert "market_analysis" in prompt
    assert SOURCE_URL in prompt
    assert "budget" not in prompt.lower()


@pytest.mark.asyncio
async def test_no_research_returns_empty_safe_result_without_provider_call():
    agent = SWOTRiskAnalysisAgent(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("unexpected provider call"))
        )
    )
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
    with pytest.raises(ServiceError, match="SWOT/risk"):
        await SWOTRiskAnalysisAgent(
            transport=response_transport(completion(content))
        ).analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "credential"),
        (403, "denied"),
        (402, "quota or billing"),
        (429, "rate limit"),
        (400, "HTTP 400"),
    ],
)
async def test_swot_provider_errors_are_categorized(status, expected):
    with pytest.raises(ServiceError, match=expected):
        await SWOTRiskAnalysisAgent(
            transport=response_transport({"error": {"message": "provider failure"}}, status)
        ).analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)


@pytest.mark.asyncio
async def test_swot_provider_connection_and_timeout_errors_are_distinct():
    def connection_error(request):
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(ServiceError, match="Could not connect") as exc:
        await SWOTRiskAnalysisAgent(
            transport=httpx.MockTransport(connection_error)
        ).analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)
    assert exc.value.status_code == 502

    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(ServiceError, match="timed out") as exc:
        await SWOTRiskAnalysisAgent(
            transport=httpx.MockTransport(timeout)
        ).analyze(IDEA, MARKET_ANALYSIS, COMPETITOR_ANALYSIS, RESEARCH)
    assert exc.value.status_code == 504
