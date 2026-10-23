import json
import pytest
from fastapi.testclient import TestClient
from main import app
from app.schemas import AnalysisResponse, StructuredIdea, StartupIdea
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



@pytest.mark.asyncio

async def test_llm_connection_failure_is_distinguished(monkeypatch):

    monkeypatch.setenv("LLM_API_KEY", "test-key")

    import httpx

    transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("unavailable", request=request)))

    with pytest.raises(ServiceError, match="Could not connect") as exc:

        await IdeaExtractionAgent(transport=transport).extract(

            __import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))

    assert exc.value.status_code == 502



@pytest.mark.asyncio

async def test_llm_timeout_is_distinguished(monkeypatch):

    monkeypatch.setenv("LLM_API_KEY", "test-key")

    import httpx

    transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timed out", request=request)))

    with pytest.raises(ServiceError, match=r"timed out \(ReadTimeout\)") as exc:

        await IdeaExtractionAgent(transport=transport).extract(

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
