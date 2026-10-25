import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas import (
    AnalysisResponse,
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    SWOTRiskAnalysis,
    Source,
    StartupIdea,
    StructuredIdea,
)
from app.services.competitor_analysis import CompetitorAnalysisAgent
from app.services.idea_extraction import IdeaExtractionAgent, ServiceError
from app.services.market_analysis import GEMINI_MODEL
from app.services.market_research import MarketResearchAgent
from app.services.orchestration import ValidationReportService
from main import app


client = TestClient(app)
SOURCE_URL = "https://research.example/competitor"
IDEA_DATA = {
    "idea_description": "An AI tutor that creates adaptive study plans for students.",
    "target_customers": "Students",
    "industry": "Education",
    "problem_statement": "Students struggle to create consistent study plans.",
    "estimated_budget": 50000,
    "estimated_budget_currency": "INR",
}
IDEA = StructuredIdea(
    problem="Students struggle to create consistent study plans",
    proposed_solution="Adaptive AI study planning",
    target_customers=["Students"],
    industry="Education",
    key_features=["Adaptive study schedule"],
    value_proposition="Consistent study habits",
    assumptions_and_missing_information=[],
)
SOURCES = [
    Source(
        title="StudyPath product review",
        url=SOURCE_URL,
        content="StudyPath provides study schedules for college students. The result does not provide pricing.",
    )
]
RESEARCH = {
    "market_information": [
        Source(title="Student study research", url="https://research.example/market", content="Students report difficulty maintaining study routines.")
    ],
    "competitors_and_alternatives": SOURCES,
    "industry_trends": [],
}
MARKET_ANALYSIS = MarketOpportunityAnalysis(
    demand_assessment={
        "claim": "Study planning is a reported student need.",
        "classification": "evidence_based",
        "evidence": ["https://research.example/market"],
    },
    customer_segments=[
        {
            "segment": {
                "claim": "College students",
                "classification": "evidence_based",
                "evidence": ["https://research.example/market"],
            },
            "needs": [
                {
                    "claim": "Consistent study routines",
                    "classification": "hypothesis",
                    "evidence": [],
                }
            ],
        }
    ],
)


def valid_payload():
    return {
        "competitors": [
            {
                "name": "StudyPath",
                "competitor_type": "direct",
                "offering": "Study schedule planning",
                "target_customers": "College students",
                "pricing": "Unknown",
                "strengths": ["Provides study schedules"],
                "limitations": [],
                "evidence": [SOURCE_URL],
            },
            {
                "name": "PlannerPlus",
                "competitor_type": "unknown",
                "offering": "Unknown",
                "target_customers": "Unknown",
                "pricing": "Unknown",
                "strengths": [],
                "limitations": [],
                "evidence": [SOURCE_URL],
            },
            {
                "name": "TutorGroup",
                "competitor_type": "indirect",
                "offering": "Peer-led study groups",
                "target_customers": "College students",
                "pricing": "Unknown",
                "strengths": [],
                "limitations": [],
                "evidence": [SOURCE_URL],
            },
        ],
        "market_gaps": [
            {
                "description": "Potential opportunity to offer adaptive schedules",
                "rationale": "The supplied result mentions schedules but does not establish adaptive planning; validate this with customers and further research.",
                "gap_type": "hypothesis",
                "evidence": [SOURCE_URL],
            }
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


@pytest.fixture(autouse=True)
def configured_mock_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)


def make_agent(response=None, error=None, captured=None):
    captured = captured if captured is not None else {}
    clients = []

    def factory(api_key):
        captured["api_key_received"] = api_key
        c = FakeGeminiClient(response=response, error=error, captured=captured)
        clients.append(c)
        return c

    return CompetitorAnalysisAgent(client_factory=factory), captured, clients


@pytest.mark.asyncio
async def test_competitor_agent_returns_typed_multi_competitor_result_and_unknown_pricing():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(valid_payload()), captured=captured)

    result = await agent.analyze(IDEA, RESEARCH, MARKET_ANALYSIS)

    assert isinstance(result, CompetitorAnalysis)
    assert len(result.competitors) == 3
    assert result.competitors[0].evidence == [SOURCE_URL]
    assert result.competitors[0].competitor_type == "direct"
    assert result.competitors[1].pricing == "Unknown"
    assert result.competitors[1].competitor_type == "unknown"
    assert result.competitors[2].competitor_type == "indirect"
    assert result.market_gaps[0].gap_type == "hypothesis"

    assert captured["api_key_received"] == "mock-gemini-key"
    req = captured["request"]
    assert req["model"] == GEMINI_MODEL == "gemini-3.8-flash"
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["config"]["system_instruction"]
    prompt = req["contents"]
    assert SOURCE_URL in prompt
    assert "50000" not in prompt
    assert "estimated_budget" not in prompt
    assert "College students" in prompt
    assert "exactly one syntactically valid JSON object" in prompt
    assert '"pricing": string; use exactly "Unknown"' in prompt
    assert '"competitors":[],"market_gaps":[]' in prompt
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_competitor_sources_returns_empty_without_gemini_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = []

    def factory(api_key):
        calls.append(api_key)
        raise AssertionError("unexpected Gemini call")

    agent = CompetitorAnalysisAgent(client_factory=factory)
    result = await agent.analyze(IDEA, {"competitors_and_alternatives": []})
    assert result.competitors == []
    assert result.market_gaps == []
    assert calls == []


@pytest.mark.asyncio
async def test_missing_market_analysis_is_passed_as_unknown_context():
    captured = {}
    agent, captured, _ = make_agent(json.dumps({"competitors": [], "market_gaps": []}), captured=captured)

    result = await agent.analyze(IDEA, RESEARCH)
    assert result.competitors == []
    prompt = captured["request"]["contents"]
    assert '"claim": "Unknown"' in prompt
    assert '"customer_segments": []' in prompt


@pytest.mark.asyncio
async def test_missing_gemini_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ServiceError, match="GEMINI_API_KEY"):
        await CompetitorAnalysisAgent().analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_malformed_json_is_rejected():
    agent, _, _ = make_agent("not json")
    with pytest.raises(ServiceError, match="invalid competitor analysis JSON"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_empty_response_is_rejected():
    agent, _, _ = make_agent("")
    with pytest.raises(ServiceError, match="empty competitor analysis"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"competitors": [{"name": "Example", "offering": "Tool", "evidence": ["https://evil.invalid"]}], "market_gaps": []}, "unsupported source URL"),
        ({"competitors": [{"name": "Example", "competitor_type": "possible", "evidence": [SOURCE_URL]}], "market_gaps": []}, "required schema"),
        ({"competitors": [], "market_gaps": [{"description": "Gap", "rationale": "Test", "gap_type": "proven", "evidence": [SOURCE_URL]}]}, "required schema"),
        ({"competitors": [{"name": "Incomplete"}], "market_gaps": []}, "required schema"),
        ({"competitors": [], "market_gaps": [], "unrequested_claim": "not accepted"}, "required schema"),
    ],
)
async def test_invalid_schema_or_unsupported_citation_rejected(payload, expected):
    agent, _, _ = make_agent(json.dumps(payload))
    with pytest.raises(ServiceError, match=expected):
        await agent.analyze(IDEA, RESEARCH)


def test_invalid_gap_type_and_unsupported_evidence_are_rejected_by_schema():
    payload = valid_payload()
    payload["market_gaps"][0]["gap_type"] = "proven"
    with pytest.raises(ValidationError):
        CompetitorAnalysis.model_validate(payload, context={"allowed_urls": {SOURCE_URL}})

    payload = valid_payload()
    payload["competitors"][0]["evidence"] = ["https://untrusted.invalid"]
    with pytest.raises(ValidationError, match="absent from supplied research"):
        CompetitorAnalysis.model_validate(payload, context={"allowed_urls": {SOURCE_URL}})

    payload = valid_payload()
    payload["competitors"][0]["competitor_type"] = "potential"
    with pytest.raises(ValidationError):
        CompetitorAnalysis.model_validate(payload, context={"allowed_urls": {SOURCE_URL}})

    payload = valid_payload()
    parsed = CompetitorAnalysis.model_validate(
        payload, context={"allowed_urls": {SOURCE_URL}}
    )
    assert [item.competitor_type for item in parsed.competitors] == [
        "direct", "unknown", "indirect"
    ]

    legacy_payload = valid_payload()
    for item in legacy_payload["competitors"]:
        item.pop("competitor_type")
    legacy_parsed = CompetitorAnalysis.model_validate(
        legacy_payload, context={"allowed_urls": {SOURCE_URL}}
    )
    assert [item.competitor_type for item in legacy_parsed.competitors] == [
        "unknown", "unknown", "unknown"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected,status_code",
    [
        (400, "HTTP 400", 502),
        (401, "credential", 503),
        (403, "denied", 503),
        (429, "rate or quota", 503),
        (500, "provider request failed", 502),
    ],
)
async def test_gemini_api_errors_are_sanitized_and_mapped(status, expected, status_code, caplog):
    error = RuntimeError("sensitive provider message mock-gemini-key")
    error.code = status
    error.message = "sensitive provider message mock-gemini-key"
    agent, _, _ = make_agent(error=error)
    caplog.set_level(logging.WARNING, logger="app.services.competitor_analysis")

    with pytest.raises(ServiceError, match=expected) as raised:
        await agent.analyze(IDEA, RESEARCH)

    assert raised.value.status_code == status_code
    assert "mock-gemini-key" not in caplog.text


@pytest.mark.asyncio
async def test_gemini_timeout_returns_504():
    agent, _, _ = make_agent(error=asyncio.TimeoutError())
    with pytest.raises(ServiceError, match="timed out") as exc:
        await agent.analyze(IDEA, RESEARCH)
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_gemini_connection_failure_returns_502():
    agent, _, _ = make_agent(error=httpx.ConnectError("offline"))
    with pytest.raises(ServiceError, match="Could not connect") as exc:
        await agent.analyze(IDEA, RESEARCH)
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_gemini_httpx_timeout_returns_504():
    agent, _, _ = make_agent(error=httpx.ReadTimeout("timeout"))
    with pytest.raises(ServiceError, match="timed out") as exc:
        await agent.analyze(IDEA, RESEARCH)
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_aclose_is_called_after_success():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(valid_payload()), captured=captured)
    await agent.analyze(IDEA, RESEARCH)
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_is_called_after_error():
    captured = {}
    error = RuntimeError("provider failure")
    error.code = 500
    error.message = "provider failure"
    agent, captured, clients = make_agent(error=error, captured=captured)
    with pytest.raises(ServiceError):
        await agent.analyze(IDEA, RESEARCH)
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_tavily_timeout_and_connection_failures_remain_distinct(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "mock-tavily-key")

    def timeout(request):
        raise httpx.ReadTimeout("mock timeout", request=request)

    with pytest.raises(ServiceError, match="timed out"):
        await MarketResearchAgent(transport=httpx.MockTransport(timeout)).research(IDEA)

    def connection_error(request):
        raise httpx.ConnectError("mock disconnect", request=request)

    with pytest.raises(ServiceError, match="Could not connect to Tavily"):
        await MarketResearchAgent(
            transport=httpx.MockTransport(connection_error)
        ).research(IDEA)


@pytest.mark.asyncio
async def test_orchestration_calls_agents_in_order_and_preserves_data(monkeypatch):
    order = []
    monkeypatch.setenv("LLM_API_KEY", "mock-only")
    class Extractor:
        def __init__(self):
            captured_extraction = {}

            def factory(api_key):
                from types import SimpleNamespace
                from unittest.mock import AsyncMock as _AsyncMock

                def make_response(content):
                    return SimpleNamespace(text=content)

                extraction_response = json.dumps(
                    {
                        "problem": IDEA.problem,
                        "proposed_solution": IDEA.proposed_solution,
                        "target_customers": IDEA.target_customers,
                        "industry": IDEA.industry,
                        "key_features": IDEA.key_features,
                        "value_proposition": IDEA.value_proposition,
                        "assumptions_and_missing_information": [],
                    }
                )

                async def generate_content(**kwargs):
                    prompt = kwargs.get("contents", "")
                    assert "50000" not in prompt
                    assert "estimated_budget" not in prompt
                    assert "estimated_budget_currency" not in prompt
                    return make_response(extraction_response)

                client = SimpleNamespace(
                    aio=SimpleNamespace(
                        models=SimpleNamespace(generate_content=generate_content),
                        aclose=_AsyncMock(),
                    ),
                    close=lambda: None,
                )
                return client

            self.agent = IdeaExtractionAgent(client_factory=factory)

        async def extract(self, submitted):
            order.append("extract")
            return await self.agent.extract(submitted)

    class Researcher:
        async def research(self, idea):
            order.append("research")
            return RESEARCH

    received_market_context = []

    class CompetitorAnalyzer:
        async def analyze(self, idea, research, market_analysis):
            order.append("competitor")
            received_market_context.append(market_analysis)
            return CompetitorAnalysis.model_validate(
                valid_payload(), context={"allowed_urls": {SOURCE_URL}}
            )

    class MarketAnalyzer:
        async def analyze(self, idea, research):
            order.append("market")
            return MARKET_ANALYSIS

    class SwotRiskAnalyzer:
        async def analyze(self, idea, market_analysis, competitor_analysis, research):
            order.append("swot-risk")
            return SWOTRiskAnalysis()

    for currency, amount in [("INR", 50000), ("USD", 1250)]:
        submitted = StartupIdea(**{**IDEA_DATA, "estimated_budget_currency": currency, "estimated_budget": amount})
        report = await ValidationReportService(
            extractor=Extractor(),
            researcher=Researcher(),
            market_analyzer=MarketAnalyzer(),
            competitor_analyzer=CompetitorAnalyzer(),
            swot_risk_analyzer=SwotRiskAnalyzer(),
        ).analyze(submitted)
        assert report.submitted_budget.amount == amount
        assert report.submitted_budget.currency == currency
        assert report.market_research == RESEARCH
        assert report.market_analysis.demand_assessment.claim == "Study planning is a reported student need."
        assert report.swot_risk_analysis.execution_risks == []
        assert [
            item.competitor_type for item in report.competitor_analysis.competitors
        ] == ["direct", "unknown", "indirect"]

    assert order == ["extract", "research", "market", "competitor", "swot-risk"] * 2
    assert received_market_context == [MARKET_ANALYSIS, MARKET_ANALYSIS]


def test_api_legacy_and_analysis_contract_compatibility():
    assert client.get("/").status_code == 200
    legacy = client.post("/api/ideas", json=IDEA_DATA)
    assert legacy.status_code == 200
    assert legacy.json()["idea"]["industry"] == "Education"
    assert client.post("/api/ideas", json={**IDEA_DATA, "estimated_budget": 0}).status_code == 422
    missing_industry = {**IDEA_DATA}
    missing_industry.pop("industry")
    assert client.post("/api/ideas", json=missing_industry).status_code == 422

    response_model = AnalysisResponse(
        idea=IDEA,
        submitted_budget={"amount": 50000, "currency": "INR"},
        market_research=RESEARCH,
        competitor_analysis=valid_payload(),
    )
    assert response_model.model_dump(mode="json")["competitor_analysis"]["competitors"][1]["pricing"] == "Unknown"


def test_analysis_endpoint_serializes_competitor_analysis():
    from unittest.mock import AsyncMock, patch

    report = AnalysisResponse(
        idea=IDEA,
        submitted_budget={"amount": 50000, "currency": "INR"},
        market_research=RESEARCH,
        competitor_analysis=valid_payload(),
    )
    service = type("MockService", (), {})()
    service.analyze = AsyncMock(return_value=report)
    with patch("main.ValidationReportService", return_value=service):
        response = client.post("/api/ideas/analyze", json=IDEA_DATA)

    assert response.status_code == 200
    body = response.json()
    assert body["competitor_analysis"]["competitors"][0]["name"] == "StudyPath"
    assert body["competitor_analysis"]["competitors"][0]["competitor_type"] == "direct"
    assert body["competitor_analysis"]["competitors"][0]["pricing"] == "Unknown"
    assert body["competitor_analysis"]["market_gaps"][0]["gap_type"] == "hypothesis"
    assert body["swot_risk_analysis"]["execution_risks"] == []
    assert body["market_research"]["competitors_and_alternatives"][0]["url"] == SOURCE_URL
