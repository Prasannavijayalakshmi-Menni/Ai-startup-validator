import json
import pytest
import httpx
from fastapi.testclient import TestClient
from main import app
from app.schemas import (
    AnalysisResponse,
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
    StartupIdea,
    SWOTRiskAnalysis,
)
from app.services.idea_extraction import IdeaExtractionAgent, ServiceError
from app.services.market_research import MarketResearchAgent

client = TestClient(app)
VALID = {
    "idea_description": "An AI tutor that creates adaptive study plans for students.",
    "target_customers": "Students",
    "industry": "Education",
    "problem_statement": "Students struggle to create consistent study plans.",
    "estimated_budget": 20000,
}

def test_analysis_endpoint_preserves_submitted_budget_and_currency():
    from unittest.mock import AsyncMock, patch
    structured = StructuredIdea(
        problem="Students lack study plans",proposed_solution="Adaptive study planner",

        target_customers=["Students"],industry="Education",key_features=["Adaptive schedule"],

        value_proposition="Consistent study habits",assumptions_and_missing_information=[],

    )

    analysis=AnalysisResponse(

        idea=structured,

        submitted_budget={"amount":50000,"currency":"INR"},

        market_research={"market_information":[],"competitors_and_alternatives":[],"industry_trends":[]},

    )

    service=type("MockService",(),{})()

    service.analyze=AsyncMock(return_value=analysis)

    with patch("main.ValidationReportService",return_value=service):

        response=client.post("/api/ideas/analyze",json={

            **VALID,"estimated_budget":50000,"estimated_budget_currency":"INR",

        })

    assert response.status_code == 200

    assert response.json()["submitted_budget"] == {"amount":50000.0,"currency":"INR"}


@pytest.mark.parametrize("origin", ["http://127.0.0.1:5174", "http://localhost:5174"])
def test_vite_5174_origins_pass_analysis_cors_preflight(origin):
    response = client.options(
        "/api/ideas/analyze",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_analyze_route_runs_mocked_workflow_and_returns_all_agent_results():
    from unittest.mock import AsyncMock, patch
    from app.services.orchestration import ValidationReportService

    idea = StructuredIdea(
        problem="Students struggle to plan study time",
        proposed_solution="An adaptive study planner",
        target_customers=["Students"],
        industry="Education",
        key_features=["Adaptive schedule"],
        value_proposition="Consistent study habits",
        assumptions_and_missing_information=[],
    )
    source = Source(
        title="Synthetic test research",
        url="https://research.example/study-planning",
        content="Synthetic mocked research snippet; not a real source.",
    )
    research = {
        "market_information": [source],
        "competitors_and_alternatives": [source],
        "industry_trends": [],
    }
    market = MarketOpportunityAnalysis(
        demand_assessment={
            "claim": "Study planning is an identified need in the synthetic test data.",
            "classification": "evidence_based",
            "evidence": [source.url],
        }
    )
    competitors = CompetitorAnalysis()
    swot = SWOTRiskAnalysis()
    events = []

    async def response_for(stage, response, *args):
        events.append((stage, args))
        return response

    def mock_stage(stage, response):
        async def run(*args):
            return await response_for(stage, response, *args)

        return run

    extractor = type("FakeExtractor", (), {})()
    extractor.extract = AsyncMock(side_effect=mock_stage("extract", idea))
    researcher = type("FakeResearcher", (), {})()
    researcher.research = AsyncMock(side_effect=mock_stage("research", research))
    market_agent = type("FakeMarketAgent", (), {})()
    market_agent.analyze = AsyncMock(side_effect=mock_stage("market", market))
    competitor_agent = type("FakeCompetitorAgent", (), {})()
    competitor_agent.analyze = AsyncMock(side_effect=mock_stage("competitor", competitors))
    swot_agent = type("FakeSwotAgent", (), {})()
    swot_agent.analyze = AsyncMock(side_effect=mock_stage("swot", swot))
    service = ValidationReportService(
        extractor=extractor,
        researcher=researcher,
        market_analyzer=market_agent,
        competitor_analyzer=competitor_agent,
        swot_risk_analyzer=swot_agent,
    )

    with patch("main.ValidationReportService", return_value=service):
        response = client.post(
            "/api/ideas/analyze",
            json={
                **VALID,
                "estimated_budget": 50000,
                "estimated_budget_currency": "INR",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["market_analysis"]["demand_assessment"]["evidence"] == [source.url]
    assert payload["competitor_analysis"] == {"competitors": [], "market_gaps": []}
    assert payload["swot_risk_analysis"]["execution_risks"] == []
    assert payload["submitted_budget"] == {"amount": 50000.0, "currency": "INR"}
    assert [stage for stage, _ in events] == [
        "extract", "research", "market", "competitor", "swot"
    ]
    assert events[2][1] == (idea, research)
    assert events[3][1] == (idea, research, market)
    assert events[4][1] == (idea, market, competitors, research)



@pytest.mark.asyncio
async def test_llm_connection_failure_is_distinguished(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    async def raise_connect_error(**kwargs):
        raise httpx.ConnectError("unavailable")

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=raise_connect_error),
            aclose=AsyncMock(),
        ),
        close=lambda: None,
    )
    with pytest.raises(ServiceError, match="Could not connect") as exc:
        await IdeaExtractionAgent(client_factory=lambda _: fake_client).extract(
            __import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_llm_timeout_is_distinguished(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    async def raise_timeout(**kwargs):
        raise httpx.ReadTimeout("timed out")

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=raise_timeout),
            aclose=AsyncMock(),
        ),
        close=lambda: None,
    )
    with pytest.raises(ServiceError, match=r"timed out") as exc:
        await IdeaExtractionAgent(client_factory=lambda _: fake_client).extract(
            __import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))
    assert exc.value.status_code == 504



@pytest.mark.asyncio

async def test_missing_tavily_key(monkeypatch):

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    idea=StructuredIdea(problem="problem",proposed_solution="solution",target_customers=["farmers"],industry="Agriculture",key_features=["alerts"],value_proposition="yield",assumptions_and_missing_information=[])

    with pytest.raises(ServiceError, match="TAVILY_API_KEY"):

        await MarketResearchAgent().research(idea)



@pytest.mark.asyncio

async def test_tavily_provider_failure(monkeypatch):

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    import httpx

    from unittest.mock import patch

    def handler(request):

        return httpx.Response(429, request=request)

    transport=httpx.MockTransport(handler)

    with pytest.raises(ServiceError, match="rate limit"):

        await MarketResearchAgent(transport=transport).research(

            StructuredIdea(problem="problem",proposed_solution="solution",target_customers=["farmers"],industry="Agriculture",key_features=["alerts"],value_proposition="yield",assumptions_and_missing_information=[]))



@pytest.mark.asyncio

async def test_tavily_empty_results_are_reported(monkeypatch):

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    import httpx

    transport=httpx.MockTransport(lambda request: httpx.Response(200,json={"results":[]},request=request))

    idea=StructuredIdea(problem="problem",proposed_solution="solution",target_customers=["farmers"],industry="Agriculture",key_features=["alerts"],value_proposition="yield",assumptions_and_missing_information=[])

    with pytest.raises(ServiceError, match="no market research results"):

        await MarketResearchAgent(transport=transport).research(idea)



@pytest.mark.asyncio

async def test_tavily_preserves_source_data_and_skips_incomplete_sources(monkeypatch):

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    import httpx

    results=[

        {"title":"Source title exactly as returned","url":"https://source.example/item?x=1","content":"Snippet with an unverified claim."},

        {"url":"https://missing-title.example","content":"No title"},

        {"title":"No URL","content":"No source URL"},

    ]

    transport=httpx.MockTransport(lambda request: httpx.Response(200,json={"results":results},request=request))

    idea=StructuredIdea(problem="problem",proposed_solution="solution",target_customers=["farmers"],industry="Agriculture",key_features=["alerts"],value_proposition="yield",assumptions_and_missing_information=[])

    response=await MarketResearchAgent(transport=transport).research(idea)

    source=response["market_information"][0]

    assert source.title == "Source title exactly as returned"
    assert source.url == "https://source.example/item?x=1"
    assert source.content == "Snippet with an unverified claim."
    assert len(response["market_information"]) == 1
    assert set(response) == {"market_information","competitors_and_alternatives","industry_trends"}


@pytest.mark.parametrize("industry", ["Education", "Healthcare", "Agriculture"])
def test_three_industry_inputs(industry):
    body={**VALID,"industry":industry}
    assert client.post("/api/ideas",json=body).status_code == 200
