import os
import json
import pytest
from fastapi.testclient import TestClient
from main import app
from app.services.idea_extraction import IdeaExtractionAgent, ServiceError
from app.services.market_research import MarketResearchAgent
from app.schemas import AnalysisResponse, StructuredIdea
from app.services.orchestration import ValidationReportService

client = TestClient(app)
VALID = {"idea_description":"An AI tutor that creates adaptive study plans for students.","target_customers":"Students","industry":"Education","problem_statement":"Students struggle to create consistent study plans.","estimated_budget":20000}

def test_health_and_legacy_submit():
    assert client.get("/").status_code == 200
    response = client.post("/api/ideas", json=VALID)
    assert response.status_code == 200 and response.json()["idea"]["industry"] == "Education"
    assert "estimated_budget_currency" not in response.json()["idea"]

def test_invalid_and_missing_fields_rejected():
    assert client.post("/api/ideas", json={**VALID, "estimated_budget": 0}).status_code == 422
    missing = dict(VALID); missing.pop("industry")
    assert client.post("/api/ideas", json=missing).status_code == 422

@pytest.mark.asyncio
async def test_missing_llm_key_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ServiceError, match="LLM_API_KEY"):
        await IdeaExtractionAgent().extract(__import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))

@pytest.mark.asyncio
async def test_malformed_llm_response(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    class Response:
        status_code=200
        def json(self): return {"choices":[{"message":{"content":"not-json"}}]}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def post(self,*args,**kwargs): return Response()
    import httpx
    from unittest.mock import patch
    with patch.object(httpx, "AsyncClient", return_value=Client()):
        with pytest.raises(ServiceError, match="invalid JSON"):
            await IdeaExtractionAgent().extract(__import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))

@pytest.mark.asyncio
async def test_llm_success_returns_structured_idea(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    import httpx
    structured={"problem":"Students lack study plans","proposed_solution":"Adaptive study planner","target_customers":["Students"],"industry":"Education","key_features":["Adaptive schedule"],"value_proposition":"Consistent study habits","assumptions_and_missing_information":[]}
    def handler(request):
        body=json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["response_format"] == {"type":"json_object"}
        assert "estimated_budget" not in body["messages"][1]["content"]
        assert "estimated_budget_currency" not in body["messages"][1]["content"]
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(structured)}}]},request=request)
    result=await IdeaExtractionAgent(transport=httpx.MockTransport(handler)).extract(
        __import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))
    assert result.industry == "Education"
    assert result.target_customers == ["Students"]

@pytest.mark.asyncio
async def test_llm_accepts_json_in_markdown_fence(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    import httpx
    structured={"problem":"Students lack study plans","proposed_solution":"Adaptive study planner","target_customers":["Students"],"industry":"Education","key_features":["Adaptive schedule"],"value_proposition":"Consistent study habits","assumptions_and_missing_information":[]}
    content="```json\n" + json.dumps(structured) + "\n```"
    transport=httpx.MockTransport(lambda request: httpx.Response(200,json={"choices":[{"message":{"content":content}}]},request=request))
    result=await IdeaExtractionAgent(transport=transport).extract(
        __import__("app.schemas",fromlist=["StartupIdea"]).StartupIdea(**VALID))
    assert result.industry == "Education"

@pytest.mark.asyncio
@pytest.mark.parametrize("payload,match", [
    ({"problem": "ok", "proposed_solution": "ok", "target_customers": "Students", "industry": "Education", "key_features": [], "value_proposition": "ok", "assumptions_and_missing_information": []}, "required idea fields"),
    ({"problem": "ok", "proposed_solution": "ok", "target_customers": ["Students"], "industry": "Education", "key_features": [], "assumptions_and_missing_information": []}, "required idea fields"),
])
async def test_llm_rejects_wrong_types_and_missing_required_fields(monkeypatch,payload,match):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    import httpx
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(payload)}}]},request=request)
    with pytest.raises(ServiceError,match=match):
        await IdeaExtractionAgent(transport=httpx.MockTransport(handler)).extract(
            __import__("app.schemas",fromlist=["StartupIdea"]).StartupIdea(**VALID))
    assert len(calls) == 1

@pytest.mark.asyncio
async def test_llm_quota_error_is_identified(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    class Response:
        status_code=429
        def json(self): return {"error":{"code":"credit_balance_exhausted","type":"insufficient_quota"}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def post(self,*args,**kwargs): return Response()
    import httpx
    from unittest.mock import patch
    with patch.object(httpx, "AsyncClient", return_value=Client()):
        with pytest.raises(ServiceError, match="billing limit"):
            await IdeaExtractionAgent().extract(__import__("app.schemas", fromlist=["StartupIdea"]).StartupIdea(**VALID))

@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload,match", [
    (401,{"error":{"type":"authentication_error","code":"invalid_api_key"}},"rejected the API credential"),
    (400,{"error":{"type":"invalid_request_error","code":"model_not_found","message":"model not found"}},"model is invalid or unavailable"),
    (429,{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded"}},"rate limit reached"),
])
async def test_llm_provider_errors_are_distinguished(monkeypatch,status,payload,match):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    import httpx
    transport=httpx.MockTransport(lambda request: httpx.Response(status,json=payload,request=request))
    with pytest.raises(ServiceError,match=match):
        await IdeaExtractionAgent(transport=transport).extract(
            __import__("app.schemas",fromlist=["StartupIdea"]).StartupIdea(**VALID))

@pytest.mark.asyncio
@pytest.mark.parametrize("amount,currency", [(50000,"INR"),(1250,"USD")])
async def test_analysis_orchestration_preserves_budget_and_combines_mocked_services(monkeypatch,amount,currency):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    import httpx
    structured={"problem":"Students lack study plans","proposed_solution":"Adaptive study planner","target_customers":["Students"],"industry":"Education","key_features":["Adaptive schedule"],"value_proposition":"Consistent study habits","assumptions_and_missing_information":[]}
    llm=httpx.MockTransport(lambda request: httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(structured)}}]},request=request))
    search=httpx.MockTransport(lambda request: httpx.Response(200,json={"results":[{"title":"Example source","url":"https://example.test/source","content":"Example snippet"}]},request=request))
    request_data={**VALID,"estimated_budget":amount,"estimated_budget_currency":currency}
    response=await ValidationReportService(
        extractor=IdeaExtractionAgent(transport=llm),
        researcher=MarketResearchAgent(transport=search),
    ).analyze(__import__("app.schemas",fromlist=["StartupIdea"]).StartupIdea(**request_data))
    assert response.idea.industry == "Education"
    assert response.submitted_budget.amount == amount
    assert response.submitted_budget.currency == currency
    assert all("budget" not in note.lower() for note in response.idea.assumptions_and_missing_information)
    assert len(response.market_research) == 3
    assert response.market_research["market_information"][0].title == "Example source"
    assert response.market_research["market_information"][0].url == "https://example.test/source"
    assert response.market_research["market_information"][0].content == "Example snippet"

@pytest.mark.asyncio
async def test_extraction_marks_missing_fields_instead_of_inventing(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    import httpx
    extracted={"problem":"Students lack study plans","proposed_solution":None,"target_customers":["Students"],"industry":"Education","key_features":[],"value_proposition":None,"assumptions_and_missing_information":["proposed_solution is not specified in the submission","key_features are not specified in the submission","value_proposition is not specified in the submission"]}
    transport=httpx.MockTransport(lambda request: httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(extracted)}}]},request=request))
    result=await IdeaExtractionAgent(transport=transport).extract(
        __import__("app.schemas",fromlist=["StartupIdea"]).StartupIdea(**VALID))
    assert result.proposed_solution is None
    assert result.key_features == []
    assert result.value_proposition is None
    assert len(result.assumptions_and_missing_information) == 3

def test_missing_extracted_fields_must_be_named():
    from pydantic import ValidationError
    with pytest.raises(ValidationError,match="must be named"):
        StructuredIdea.model_validate({
            "problem":None,"proposed_solution":"Adaptive tutor","target_customers":["Students"],
            "industry":"Education","key_features":["Study plans"],"value_proposition":"Better planning",
            "assumptions_and_missing_information":[],
        })

def test_analysis_endpoint_preserves_submitted_budget_and_currency():
    from unittest.mock import AsyncMock, patch
    structured=StructuredIdea(
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
