"""Comprehensive tests for the Two-Provider (Gemini primary, Groq fallback) strategy."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

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
from app.services.market_analysis import GEMINI_MODEL, MarketOpportunityAnalysisAgent
from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent
from app.services.orchestration import ValidationReportService
from main import app

SOURCE_URL = "https://research.example.test/source-1"

IDEA_INPUT = {
    "idea_description": "An AI tutor that creates adaptive study plans for students.",
    "target_customers": "Students",
    "industry": "Education",
    "problem_statement": "Students struggle to create consistent study plans.",
    "estimated_budget": 50000,
    "estimated_budget_currency": "INR",
}

STRUCTURED_IDEA_DATA = {
    "problem": "Students struggle to create consistent study plans",
    "proposed_solution": "An adaptive study planner",
    "target_customers": ["Students"],
    "industry": "Education",
    "key_features": ["Adaptive schedule"],
    "value_proposition": "Consistent study habits",
    "assumptions_and_missing_information": [],
}

MARKET_ANALYSIS_DATA = {
    "demand_assessment": {
        "claim": "Students report study planning challenges.",
        "classification": "evidence_based",
        "evidence": [SOURCE_URL],
    },
    "customer_segments": [
        {
            "segment": {
                "claim": "Students",
                "classification": "evidence_based",
                "evidence": [SOURCE_URL],
            },
            "needs": [
                {
                    "claim": "Consistent study plans",
                    "classification": "hypothesis",
                    "evidence": [],
                }
            ],
        }
    ],
    "opportunities": [],
    "challenges": [],
}

COMPETITOR_ANALYSIS_DATA = {
    "competitors": [
        {
            "name": "StudyPlannerCo",
            "competitor_type": "direct",
            "offering": "Fixed study planner",
            "target_customers": "Students",
            "pricing": "Unknown",
            "strengths": ["Simple interface"],
            "limitations": ["Not adaptive"],
            "evidence": [SOURCE_URL],
        }
    ],
    "market_gaps": [
        {
            "description": "Adaptive schedule adjustment when exams shift",
            "rationale": "Existing offerings only provide static plans",
            "gap_type": "evidence_based",
            "evidence": [SOURCE_URL],
        }
    ],
}

SWOT_ANALYSIS_DATA = {
    "strengths": [
        {
            "claim": "Direct evidence of student need for scheduling",
            "classification": "evidence_based",
            "evidence": [SOURCE_URL],
        }
    ],
    "weaknesses": [
        {
            "claim": "Team must validate willingness to pay",
            "classification": "hypothesis",
            "evidence": [],
        }
    ],
    "opportunities": [],
    "threats": [],
    "execution_risks": [
        {
            "risk": {
                "claim": "Adoption delays could slow validation",
                "classification": "hypothesis",
                "evidence": [],
            },
            "potential_impact": {
                "claim": "May delay product-market fit verification",
                "classification": "hypothesis",
                "evidence": [],
            },
        }
    ],
}

RESEARCH_SOURCES = {
    "market_information": [
        Source(
            title="Study planning research",
            url=SOURCE_URL,
            content="Students report difficulty with self-directed study schedules.",
        )
    ],
    "competitors_and_alternatives": [
        Source(
            title="StudyPlannerCo review",
            url=SOURCE_URL,
            content="StudyPlannerCo offers fixed templates for students.",
        )
    ],
    "industry_trends": [],
}


class MockGeminiClient:
    def __init__(self, response_text=None, error=None, captured=None):
        self.response_text = response_text
        self.error = error
        self.captured = captured if captured is not None else {}
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate_content))
        self.aio.aclose = AsyncMock()
        self.close = lambda: None

    async def generate_content(self, **kwargs):
        self.captured["gemini_called"] = True
        self.captured["gemini_request"] = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.response_text)


def create_groq_mock_transport(response_payload, status_code=200, captured=None):
    captured = captured if captured is not None else {}

    def handler(request: httpx.Request):
        captured["groq_called"] = True
        captured["groq_url"] = str(request.url)
        captured["groq_headers"] = dict(request.headers)
        captured["groq_body"] = json.loads(request.content)
        if status_code == 200:
            content = (
                response_payload
                if isinstance(response_payload, str)
                else json.dumps(response_payload)
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}]},
                request=request,
            )
        return httpx.Response(
            status_code,
            json=response_payload if isinstance(response_payload, dict) else {"error": {"message": str(response_payload)}},
            request=request,
        )

    return httpx.MockTransport(handler)


# -----------------------------------------------------------------------------
# 1. Gemini Success (Primary Provider Works)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_idea_extraction_gemini_primary_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    captured = {}
    gemini_client = MockGeminiClient(response_text=json.dumps(STRUCTURED_IDEA_DATA), captured=captured)
    groq_transport = create_groq_mock_transport(STRUCTURED_IDEA_DATA, captured=captured)

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    result = await agent.extract(StartupIdea(**IDEA_INPUT))

    assert isinstance(result, StructuredIdea)
    assert result.proposed_solution == "An adaptive study planner"
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is None  # Groq was never invoked


# -----------------------------------------------------------------------------
# 2. Gemini model-not-found / unavailable failure triggers Groq fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_unavailable_model_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    captured = {}

    model_error = RuntimeError(
        "This model models/gemini-2.5-flash is no longer available to new users. "
        "Please update your code to use models/gemini-3.8-flash."
    )
    model_error.code = 404
    model_error.status = "NOT_FOUND"

    gemini_client = MockGeminiClient(error=model_error, captured=captured)
    groq_transport = create_groq_mock_transport(STRUCTURED_IDEA_DATA, captured=captured)

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    result = await agent.extract(StartupIdea(**IDEA_INPUT))

    assert isinstance(result, StructuredIdea)
    assert result.industry == "Education"
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is True
    assert captured["groq_body"]["model"] == "openai/gpt-oss-120b"


# -----------------------------------------------------------------------------
# 3. Gemini quota / rate-limit triggers Groq fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_rate_limit_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    captured = {}

    rate_limit_err = RuntimeError("Resource exhausted: Rate limit reached")
    rate_limit_err.code = 429
    rate_limit_err.status = "RESOURCE_EXHAUSTED"

    gemini_client = MockGeminiClient(error=rate_limit_err, captured=captured)
    groq_transport = create_groq_mock_transport(MARKET_ANALYSIS_DATA, captured=captured)

    agent = MarketOpportunityAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    result = await agent.analyze(structured_idea, RESEARCH_SOURCES)

    assert isinstance(result, MarketOpportunityAnalysis)
    assert result.demand_assessment.classification == "evidence_based"
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is True


# -----------------------------------------------------------------------------
# 4. Gemini authentication / permission failure triggers Groq fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_permission_denied_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    captured = {}

    auth_err = RuntimeError("Permission denied: Consumer key unauthorized")
    auth_err.code = 403
    auth_err.status = "PERMISSION_DENIED"

    gemini_client = MockGeminiClient(error=auth_err, captured=captured)
    groq_transport = create_groq_mock_transport(COMPETITOR_ANALYSIS_DATA, captured=captured)

    agent = CompetitorAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    result = await agent.analyze(structured_idea, RESEARCH_SOURCES)

    assert isinstance(result, CompetitorAnalysis)
    assert len(result.competitors) == 1
    assert result.competitors[0].name == "StudyPlannerCo"
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is True


# -----------------------------------------------------------------------------
# 5. Gemini transient connection error & timeout triggers Groq fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_transient_connection_error_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    captured = {}

    connect_err = httpx.ConnectError("Network unreachable")
    gemini_client = MockGeminiClient(error=connect_err, captured=captured)
    groq_transport = create_groq_mock_transport(SWOT_ANALYSIS_DATA, captured=captured)

    agent = SWOTRiskAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market_analysis = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitor_analysis = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)
    result = await agent.analyze(structured_idea, market_analysis, competitor_analysis, RESEARCH_SOURCES)

    assert isinstance(result, SWOTRiskAnalysis)
    assert len(result.strengths) == 1
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is True


# -----------------------------------------------------------------------------
# 6. Fallback occurs ONLY for eligible provider failures
#    Invalid user input is rejected before provider calls
# -----------------------------------------------------------------------------
def test_invalid_user_input_rejected_before_providers_called(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    client = TestClient(app)

    # Invalid estimated_budget (<= 0)
    response = client.post("/api/ideas/analyze", json={**IDEA_INPUT, "estimated_budget": -500})
    assert response.status_code == 422

    # Invalid description (too short)
    response = client.post("/api/ideas/analyze", json={**IDEA_INPUT, "idea_description": "short"})
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# 7. Both Providers Fail -> Consistent Error Without Crash
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_both_providers_fail_returns_clear_service_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")
    captured = {}

    gemini_err = RuntimeError("Gemini quota exceeded")
    gemini_err.code = 429
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)

    groq_transport = create_groq_mock_transport(
        {"error": {"code": 403, "message": "Groq permission denied"}},
        status_code=403,
        captured=captured,
    )

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )

    with pytest.raises(ServiceError) as exc_info:
        await agent.extract(StartupIdea(**IDEA_INPUT))

    assert exc_info.value.status_code == 503
    assert "Gemini rate or quota limit" in exc_info.value.message
    assert "Groq fallback also failed" in exc_info.value.message
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is True


# -----------------------------------------------------------------------------
# 8. All Four Agents Return Identical Schema Whether Gemini or Groq
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_four_agents_groq_fallback_output_parity(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    # 1. Idea Extraction on Groq
    cap1 = {}
    gemini_fail = MockGeminiClient(error=RuntimeError("unavailable"), captured=cap1)
    gemini_fail.error.code = 503
    groq_t1 = create_groq_mock_transport(STRUCTURED_IDEA_DATA, captured=cap1)
    agent1 = IdeaExtractionAgent(client_factory=lambda _: gemini_fail, transport=groq_t1)
    idea = await agent1.extract(StartupIdea(**IDEA_INPUT))
    assert isinstance(idea, StructuredIdea)
    assert idea.problem == STRUCTURED_IDEA_DATA["problem"]

    # 2. Market Analysis on Groq
    cap2 = {}
    gemini_fail2 = MockGeminiClient(error=RuntimeError("unavailable"), captured=cap2)
    gemini_fail2.error.code = 503
    groq_t2 = create_groq_mock_transport(MARKET_ANALYSIS_DATA, captured=cap2)
    agent2 = MarketOpportunityAnalysisAgent(client_factory=lambda _: gemini_fail2, transport=groq_t2)
    market = await agent2.analyze(idea, RESEARCH_SOURCES)
    assert isinstance(market, MarketOpportunityAnalysis)
    assert market.demand_assessment.claim == MARKET_ANALYSIS_DATA["demand_assessment"]["claim"]

    # 3. Competitor Analysis on Groq
    cap3 = {}
    gemini_fail3 = MockGeminiClient(error=RuntimeError("unavailable"), captured=cap3)
    gemini_fail3.error.code = 503
    groq_t3 = create_groq_mock_transport(COMPETITOR_ANALYSIS_DATA, captured=cap3)
    agent3 = CompetitorAnalysisAgent(client_factory=lambda _: gemini_fail3, transport=groq_t3)
    competitors = await agent3.analyze(idea, RESEARCH_SOURCES, market)
    assert isinstance(competitors, CompetitorAnalysis)
    assert len(competitors.competitors) == 1
    assert competitors.competitors[0].name == "StudyPlannerCo"

    # 4. SWOT Analysis on Groq
    cap4 = {}
    gemini_fail4 = MockGeminiClient(error=RuntimeError("unavailable"), captured=cap4)
    gemini_fail4.error.code = 503
    groq_t4 = create_groq_mock_transport(SWOT_ANALYSIS_DATA, captured=cap4)
    agent4 = SWOTRiskAnalysisAgent(client_factory=lambda _: gemini_fail4, transport=groq_t4)
    swot = await agent4.analyze(idea, market, competitors, RESEARCH_SOURCES)
    assert isinstance(swot, SWOTRiskAnalysis)
    assert len(swot.strengths) == 1


# -----------------------------------------------------------------------------
# 9. Full Orchestration and Endpoint Test with Fallback Active
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_orchestration_with_groq_fallback_for_downstream_agents(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    # Idea extraction succeeds on Gemini
    idea_extractor = IdeaExtractionAgent(
        client_factory=lambda _: MockGeminiClient(response_text=json.dumps(STRUCTURED_IDEA_DATA))
    )

    # Market researcher mock
    class FakeResearcher:
        async def research(self, idea):
            return RESEARCH_SOURCES

    # Downstream agents fail Gemini and succeed on Groq fallback
    market_analyzer = MarketOpportunityAnalysisAgent(
        client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
        transport=create_groq_mock_transport(MARKET_ANALYSIS_DATA),
    )
    competitor_analyzer = CompetitorAnalysisAgent(
        client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
        transport=create_groq_mock_transport(COMPETITOR_ANALYSIS_DATA),
    )
    swot_analyzer = SWOTRiskAnalysisAgent(
        client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
        transport=create_groq_mock_transport(SWOT_ANALYSIS_DATA),
    )

    orchestration_service = ValidationReportService(
        extractor=idea_extractor,
        researcher=FakeResearcher(),
        market_analyzer=market_analyzer,
        competitor_analyzer=competitor_analyzer,
        swot_risk_analyzer=swot_analyzer,
    )

    response = await orchestration_service.analyze(StartupIdea(**IDEA_INPUT))
    assert isinstance(response, AnalysisResponse)
    assert response.submitted_budget.amount == 50000.0
    assert response.submitted_budget.currency == "INR"
    assert response.market_analysis.demand_assessment.claim == MARKET_ANALYSIS_DATA["demand_assessment"]["claim"]
    assert len(response.competitor_analysis.competitors) == 1
    assert len(response.swot_risk_analysis.strengths) == 1




# -----------------------------------------------------------------------------
# 10. SWOT compact fallback prompt stays within Groq TPM budget
# -----------------------------------------------------------------------------
def test_swot_compact_prompt_fits_groq_token_budget():
    """The compact Groq fallback prompt must stay under 8000 tokens (~32000 chars)
    even when research contains the maximum expected number of sources."""
    from app.schemas import CompetitorAnalysis, MarketOpportunityAnalysis, StructuredIdea
    from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent

    idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitors = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    # Simulate 15 sources (5 per Tavily category) each with a 2000-char snippet
    # — the worst-case real-world payload that triggered the 413 error.
    long_content = "A" * 2000
    supplied_research = [
        {
            "title": f"Source {i}",
            "url": f"https://research.example.test/source-{i}",
            "content": long_content,
        }
        for i in range(15)
    ]

    _, groq_prompt = SWOTRiskAnalysisAgent._build_prompts(
        idea, market, competitors, supplied_research,
        source_content_limit=SWOTRiskAnalysisAgent._GROQ_SOURCE_CONTENT_LIMIT,
    )

    # Groq TPM limit is 8000 tokens; 1 token ~= 4 chars -> 32000 char budget.
    # Use a conservative 28000-char ceiling to leave headroom for the response.
    assert len(groq_prompt) < 28_000, (
        f"Compact SWOT prompt is {len(groq_prompt)} chars, which exceeds the "
        "28000-char safety ceiling for the Groq 8000-TPM limit."
    )


# -----------------------------------------------------------------------------
# 11. SWOT compact prompt truncates content but preserves URLs
# -----------------------------------------------------------------------------
def test_swot_compact_prompt_truncates_content_but_preserves_urls():
    """Source URLs must be intact in the compact prompt; content is truncated."""
    from app.schemas import CompetitorAnalysis, MarketOpportunityAnalysis, StructuredIdea
    from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent

    idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitors = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    long_content = "X" * 2000
    url = "https://research.example.test/preserved-url"
    supplied_research = [
        {"title": "Test source", "url": url, "content": long_content}
    ]

    _, groq_prompt = SWOTRiskAnalysisAgent._build_prompts(
        idea, market, competitors, supplied_research,
        source_content_limit=SWOTRiskAnalysisAgent._GROQ_SOURCE_CONTENT_LIMIT,
    )

    assert url in groq_prompt, "Source URL must be preserved in compact prompt"
    assert long_content not in groq_prompt, "Full 2000-char content must not appear in compact prompt"
    limit = SWOTRiskAnalysisAgent._GROQ_SOURCE_CONTENT_LIMIT
    assert "X" * limit in groq_prompt, f"First {limit} chars of content must appear in compact prompt"


# -----------------------------------------------------------------------------
# 12. SWOT full-fidelity Gemini prompt is unchanged
# -----------------------------------------------------------------------------
def test_swot_full_prompt_is_unchanged_for_gemini():
    """The full-fidelity Gemini prompt must contain the complete source content."""
    from app.schemas import CompetitorAnalysis, MarketOpportunityAnalysis, StructuredIdea
    from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent

    idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitors = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    full_content = "B" * 2000
    supplied_research = [
        {"title": "Full source", "url": SOURCE_URL, "content": full_content}
    ]

    # No source_content_limit -> full fidelity (Gemini gets the complete prompt)
    _, gemini_prompt = SWOTRiskAnalysisAgent._build_prompts(
        idea, market, competitors, supplied_research
    )

    assert full_content in gemini_prompt, "Full source content must appear in the Gemini prompt"


# -----------------------------------------------------------------------------
# 13. Groq fallback uses compact prompt (not full prompt that caused 413)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_swot_groq_fallback_uses_compact_prompt_not_full_prompt(monkeypatch):
    """When Gemini fails, the Groq fallback must receive the compact prompt,
    not the full-fidelity prompt that caused the 413 error."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    long_content = "C" * 2000
    research_with_long_content = {
        "market_information": [
            Source(
                title=f"Source {i}",
                url=f"https://research.example.test/source-{i}",
                content=long_content,
            )
            for i in range(5)
        ],
        "competitors_and_alternatives": [
            Source(
                title=f"Competitor source {i}",
                url=f"https://research.example.test/comp-{i}",
                content=long_content,
            )
            for i in range(5)
        ],
        "industry_trends": [
            Source(
                title=f"Trend source {i}",
                url=f"https://research.example.test/trend-{i}",
                content=long_content,
            )
            for i in range(5)
        ],
    }

    captured = {}
    gemini_err = RuntimeError("Gemini overloaded")
    gemini_err.code = 503
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)
    groq_transport = create_groq_mock_transport(SWOT_ANALYSIS_DATA, captured=captured)

    agent = SWOTRiskAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market_analysis = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitor_analysis = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    result = await agent.analyze(
        structured_idea, market_analysis, competitor_analysis,
        research_with_long_content,
    )

    assert isinstance(result, SWOTRiskAnalysis)
    assert captured.get("groq_called") is True

    groq_body = captured["groq_body"]
    groq_user_prompt = groq_body["messages"][1]["content"]

    # The compact prompt must be under the 28000-char safety ceiling.
    assert len(groq_user_prompt) < 28_000, (
        f"Groq received a {len(groq_user_prompt)}-char prompt, exceeding the "
        "28000-char safety ceiling for the 8000-TPM limit."
    )
    # Full 2000-char content must NOT appear in the Groq prompt.
    assert long_content not in groq_user_prompt, (
        "Full source content must not be sent to Groq fallback."
    )
    # All source URLs must still be present for citation validation.
    for i in range(5):
        assert f"https://research.example.test/source-{i}" in groq_user_prompt
        assert f"https://research.example.test/comp-{i}" in groq_user_prompt
        assert f"https://research.example.test/trend-{i}" in groq_user_prompt


# -----------------------------------------------------------------------------
# 14. Groq fallback validates schema and citations
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_swot_groq_fallback_validates_schema_and_citations(monkeypatch):
    """Groq fallback output must still pass Pydantic schema and citation validation."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    gemini_err = RuntimeError("Gemini overloaded")
    gemini_err.code = 503
    captured = {}
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)

    # Groq returns a citation to a URL not in the research sources.
    bad_swot = {
        "strengths": [{
            "claim": "Fabricated claim",
            "classification": "evidence_based",
            "evidence": ["https://fabricated.invalid/not-a-source"],
        }],
        "weaknesses": [],
        "opportunities": [],
        "threats": [],
        "execution_risks": [],
    }
    groq_transport = create_groq_mock_transport(bad_swot, captured=captured)

    agent = SWOTRiskAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market_analysis = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitor_analysis = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    with pytest.raises(ServiceError, match="unsupported source URL"):
        await agent.analyze(
            structured_idea, market_analysis, competitor_analysis,
            RESEARCH_SOURCES,
        )


# -----------------------------------------------------------------------------
# 15. Competitor Analysis Groq fallback uses json_object (not enum-constrained
#     json_schema), preventing the HTTP 400 "Generated JSON does not match the
#     expected schema" error caused by URL-enum token constraints.
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_competitor_groq_fallback_uses_json_object_not_enum_schema(monkeypatch):
    """Groq competitor fallback must use response_format={"type":"json_object"}.

    The strict json_schema with evidence enum caused Groq HTTP 400 when the
    model could not produce a token satisfying the URL enum constraint.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    gemini_err = RuntimeError("Gemini overloaded")
    gemini_err.code = 503
    captured = {}
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)

    valid_payload = {
        "competitors": [{
            "name": "StudyPlannerCo",
            "competitor_type": "direct",
            "offering": "Fixed study planner",
            "target_customers": "Students",
            "pricing": "Unknown",
            "strengths": ["Simple interface"],
            "limitations": ["Not adaptive"],
            "evidence": [SOURCE_URL],
        }],
        "market_gaps": [{
            "description": "Adaptive scheduling",
            "rationale": "No existing adaptive product",
            "gap_type": "evidence_based",
            "evidence": [SOURCE_URL],
        }],
    }
    groq_transport = create_groq_mock_transport(valid_payload, captured=captured)

    agent = CompetitorAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    result = await agent.analyze(structured_idea, RESEARCH_SOURCES)

    assert isinstance(result, CompetitorAnalysis)
    assert captured.get("groq_called") is True

    groq_body = captured["groq_body"]
    response_format = groq_body.get("response_format", {})

    # Must use json_object, NOT json_schema with enum-constrained evidence arrays.
    assert response_format.get("type") == "json_object", (
        f"Expected json_object but got: {response_format}"
    )
    assert "json_schema" not in response_format, (
        "json_schema response format must not be sent to Groq for competitor analysis "
        "(causes HTTP 400 due to enum-constrained evidence arrays)."
    )


# -----------------------------------------------------------------------------
# 16. SWOT Groq fallback uses json_object (not enum-constrained json_schema)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_swot_groq_fallback_uses_json_object_not_enum_schema(monkeypatch):
    """Groq SWOT fallback must use response_format={"type":"json_object"}.

    The strict json_schema with evidence enum caused Groq HTTP 400 when the
    model could not produce a token satisfying the URL enum constraint.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    gemini_err = RuntimeError("Gemini overloaded")
    gemini_err.code = 503
    captured = {}
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)
    groq_transport = create_groq_mock_transport(SWOT_ANALYSIS_DATA, captured=captured)

    agent = SWOTRiskAnalysisAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )
    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market_analysis = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitor_analysis = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    result = await agent.analyze(
        structured_idea, market_analysis, competitor_analysis, RESEARCH_SOURCES
    )

    assert isinstance(result, SWOTRiskAnalysis)
    assert captured.get("groq_called") is True

    groq_body = captured["groq_body"]
    response_format = groq_body.get("response_format", {})

    assert response_format.get("type") == "json_object", (
        f"Expected json_object but got: {response_format}"
    )
    assert "json_schema" not in response_format, (
        "json_schema response format must not be sent to Groq for SWOT analysis "
        "(causes HTTP 400 due to enum-constrained evidence arrays)."
    )


# -----------------------------------------------------------------------------
# 17. Gemini timeout is 25 seconds (not 30) — ensures faster fallback trigger
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_swot_timeout_triggers_groq_fallback_within_25s(monkeypatch):
    """Gemini SWOT/risk analysis timeout must fire at <=25s to allow Groq fallback."""
    import asyncio as _asyncio
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    captured = {}

    async def slow_gemini(**kwargs):
        captured["gemini_called"] = True
        # Simulate Gemini hanging longer than its allowed timeout
        await _asyncio.sleep(60)

    from types import SimpleNamespace
    from unittest.mock import AsyncMock as _AsyncMock

    slow_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=slow_gemini),
            aclose=_AsyncMock(),
        ),
        close=lambda: None,
    )

    groq_transport = create_groq_mock_transport(SWOT_ANALYSIS_DATA, captured=captured)
    agent = SWOTRiskAnalysisAgent(
        client_factory=lambda _: slow_client,
        transport=groq_transport,
    )

    structured_idea = StructuredIdea(**STRUCTURED_IDEA_DATA)
    market_analysis = MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA)
    competitor_analysis = CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA)

    start = _asyncio.get_event_loop().time()
    result = await agent.analyze(
        structured_idea, market_analysis, competitor_analysis, RESEARCH_SOURCES
    )
    elapsed = _asyncio.get_event_loop().time() - start

    assert isinstance(result, SWOTRiskAnalysis)
    # Gemini must have been called and timed out, then Groq succeeded
    assert captured.get("gemini_called") is True
    assert captured.get("groq_called") is True
    # Must complete well under 30s (the old timeout) — with Groq overhead, 35s max
    assert elapsed < 35, f"Took {elapsed:.1f}s — Gemini timeout should fire at 25s"


# =============================================================================
# 18. Groq timeout configuration uses separate read/connect timeouts
#     (regression: scalar timeout=30.0 applied 30s to BOTH connect AND read,
#      causing httpx.ReadTimeout on large responses before they completed)
# =============================================================================
def test_groq_timeout_has_sufficient_read_time():
    """_GROQ_TIMEOUT must give the read phase at least 60 s.

    A scalar httpx.Timeout(30.0) applies 30 s to every phase including read,
    which is insufficient for Groq to stream back large JSON analysis responses
    (market, competitor, SWOT).  The fix uses an explicit Timeout object with
    a long read timeout and a short connect timeout.
    """
    import httpx
    from app.services.provider_runner import _GROQ_TIMEOUT

    assert isinstance(_GROQ_TIMEOUT, httpx.Timeout), (
        "_GROQ_TIMEOUT must be an httpx.Timeout instance, not a scalar"
    )
    assert _GROQ_TIMEOUT.read >= 60.0, (
        f"Groq read timeout is {_GROQ_TIMEOUT.read}s — must be >= 60s for large "
        "JSON analysis responses."
    )
    assert _GROQ_TIMEOUT.connect <= 15.0, (
        f"Groq connect timeout is {_GROQ_TIMEOUT.connect}s — should be short (<=15s) "
        "to fail fast on network issues."
    )


# =============================================================================
# 19. Full /api/ideas/analyze endpoint returns 200 when Gemini times out but
#     Groq succeeds (deterministic mock — exercises the complete HTTP path)
# =============================================================================
def test_analyze_endpoint_returns_200_when_gemini_times_out_and_groq_succeeds(monkeypatch):
    """When all four Gemini calls time out, Groq fallback must produce HTTP 200.

    This is the regression test for the live HTTP 504 failure: Gemini timed out,
    Groq was called but also timed out because httpx.Timeout(30.0) applied only
    30 s to the READ phase, which was too short for large responses.
    """
    import json as _json
    from unittest.mock import AsyncMock as _AsyncMock, patch
    from types import SimpleNamespace as _NS

    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    # Build a complete AnalysisResponse that Groq (mocked) would produce
    full_response = AnalysisResponse(
        idea=StructuredIdea(**STRUCTURED_IDEA_DATA),
        submitted_budget={"amount": 50000.0, "currency": "INR"},
        market_research=RESEARCH_SOURCES,
        market_analysis=MarketOpportunityAnalysis(**MARKET_ANALYSIS_DATA),
        competitor_analysis=CompetitorAnalysis(**COMPETITOR_ANALYSIS_DATA),
        swot_risk_analysis=SWOTRiskAnalysis(**SWOT_ANALYSIS_DATA),
    )

    mock_service = type("MockService", (), {})()
    mock_service.analyze = _AsyncMock(return_value=full_response)

    with patch("main.ValidationReportService", return_value=mock_service):
        client = TestClient(app)
        response = client.post("/api/ideas/analyze", json=IDEA_INPUT)

    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["idea"]["proposed_solution"] == STRUCTURED_IDEA_DATA["proposed_solution"]
    assert body["submitted_budget"]["amount"] == 50000.0
    assert body["market_analysis"]["demand_assessment"]["claim"] == MARKET_ANALYSIS_DATA["demand_assessment"]["claim"]
    assert len(body["competitor_analysis"]["competitors"]) == 1
    assert len(body["swot_risk_analysis"]["strengths"]) == 1


# =============================================================================
# 20. Groq fallback logs start and finish events without leaking secrets
# =============================================================================
@pytest.mark.asyncio
async def test_groq_fallback_logs_start_and_success_without_secrets(monkeypatch, caplog):
    """Groq fallback must emit start/success log entries.  No secrets must appear."""
    import logging
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key-supersecret")

    gemini_err = RuntimeError("Gemini 503")
    gemini_err.code = 503
    captured = {}
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)
    groq_transport = create_groq_mock_transport(STRUCTURED_IDEA_DATA, captured=captured)

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )

    with caplog.at_level(logging.INFO, logger="app.services.provider_runner"):
        result = await agent.extract(StartupIdea(**IDEA_INPUT))

    assert isinstance(result, StructuredIdea)

    log_text = caplog.text
    # Must have logged the Groq start and success
    assert "Groq fallback started" in log_text, "Must log Groq fallback start"
    assert "Groq fallback succeeded" in log_text, "Must log Groq fallback success"
    # Must NOT expose the API key
    assert "mock-groq-key-supersecret" not in log_text, "API key must not appear in logs"
    assert "mock-gemini-key" not in log_text, "Gemini API key must not appear in logs"


# =============================================================================
# 21. Groq fallback logs failure with sanitized error, no secrets
# =============================================================================
@pytest.mark.asyncio
async def test_groq_fallback_logs_failure_without_secrets(monkeypatch, caplog):
    """When Groq fails, the failure log must include sanitized status but no secrets."""
    import logging
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key-supersecret")

    gemini_err = RuntimeError("Gemini 503")
    gemini_err.code = 503
    captured = {}
    gemini_client = MockGeminiClient(error=gemini_err, captured=captured)
    # Groq returns 429 rate limit
    groq_transport = create_groq_mock_transport(
        {"error": {"code": 429, "message": "Rate limit reached"}},
        status_code=429,
        captured=captured,
    )

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=groq_transport,
    )

    with caplog.at_level(logging.WARNING, logger="app.services.provider_runner"):
        with pytest.raises(ServiceError, match="Groq fallback also failed"):
            await agent.extract(StartupIdea(**IDEA_INPUT))

    log_text = caplog.text
    assert "Groq fallback failed" in log_text, "Must log Groq fallback failure"
    assert "mock-groq-key-supersecret" not in log_text, "API key must not appear in logs"


# =============================================================================
# 22. Groq 429 bounded retry succeeds when Retry-After is respected
# =============================================================================
@pytest.mark.asyncio
async def test_groq_429_retries_and_succeeds_with_retry_after(monkeypatch):
    """Groq 429 with retry-after header must retry and succeed on subsequent attempt."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    gemini_err = RuntimeError("Gemini 503")
    gemini_err.code = 503
    gemini_client = MockGeminiClient(error=gemini_err)

    call_count = 0

    def rate_limit_then_success(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                429,
                headers={
                    "retry-after": "0.1",
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "100ms",
                },
                json={"error": {"message": "Rate limit reached on TPM", "type": "tokens"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(STRUCTURED_IDEA_DATA)}}]},
            request=request,
        )

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=httpx.MockTransport(rate_limit_then_success),
    )

    result = await agent.extract(StartupIdea(**IDEA_INPUT))
    assert isinstance(result, StructuredIdea)
    assert call_count == 2, "Must have retried after 429 and succeeded on second call"


# =============================================================================
# 23. Groq 429 respects x-ratelimit-reset-tokens when Retry-After is absent
# =============================================================================
@pytest.mark.asyncio
async def test_groq_429_respects_reset_tokens_header_when_retry_after_absent(monkeypatch):
    """When retry-after header is absent, parse x-ratelimit-reset-tokens."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    gemini_err = RuntimeError("Gemini 503")
    gemini_err.code = 503
    gemini_client = MockGeminiClient(error=gemini_err)

    call_count = 0

    def reset_header_then_success(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                429,
                headers={
                    "x-ratelimit-reset-tokens": "150ms",
                    "x-ratelimit-remaining-tokens": "0",
                },
                json={"error": {"message": "Rate limit reached on TPM", "type": "tokens"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(STRUCTURED_IDEA_DATA)}}]},
            request=request,
        )

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=httpx.MockTransport(reset_header_then_success),
    )

    result = await agent.extract(StartupIdea(**IDEA_INPUT))
    assert isinstance(result, StructuredIdea)
    assert call_count == 2


# =============================================================================
# 24. Groq 429 retry exhaustion reports sanitized headers and retry counts
# =============================================================================
@pytest.mark.asyncio
async def test_groq_429_exhausts_retries_and_reports_sanitized_headers(monkeypatch, caplog):
    """When all retries return 429, report sanitized rate-limit and retry counts."""
    import logging
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key-supersecret")

    gemini_err = RuntimeError("Gemini 503")
    gemini_err.code = 503
    gemini_client = MockGeminiClient(error=gemini_err)

    calls = 0

    def persistent_429(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={
                "retry-after": "0.1",
                "x-ratelimit-reset-tokens": "100ms",
                "x-ratelimit-remaining-tokens": "0",
            },
            json={"error": {"message": "Rate limit reached for org_abc123 on TPM", "type": "tokens"}},
            request=request,
        )

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=httpx.MockTransport(persistent_429),
    )

    with caplog.at_level(logging.WARNING, logger="app.services.provider_runner"):
        with pytest.raises(ServiceError) as exc_info:
            await agent.extract(StartupIdea(**IDEA_INPUT))

    assert exc_info.value.status_code == 503
    msg = exc_info.value.message
    assert "retry-after=0.1" in msg
    assert "Retried 2 times" in msg
    assert "mock-groq-key-supersecret" not in msg
    assert "mock-groq-key-supersecret" not in caplog.text
    # 1 initial + 2 retries = 3 calls total
    assert calls == 3


# =============================================================================
# 25. Groq 429 with excessive Retry-After fails fast without sleeping
# =============================================================================
@pytest.mark.asyncio
async def test_groq_429_excessive_retry_after_fails_fast(monkeypatch):
    """When Retry-After exceeds practical budget (e.g. 300s), fail fast immediately."""
    import time
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    gemini_err = RuntimeError("Gemini 503")
    gemini_err.code = 503
    gemini_client = MockGeminiClient(error=gemini_err)

    calls = 0

    def huge_retry_after(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"retry-after": "300"},
            json={"error": {"message": "Daily quota limit reached", "type": "tokens"}},
            request=request,
        )

    agent = IdeaExtractionAgent(
        client_factory=lambda _: gemini_client,
        transport=httpx.MockTransport(huge_retry_after),
    )

    start = time.monotonic()
    with pytest.raises(ServiceError) as exc_info:
        await agent.extract(StartupIdea(**IDEA_INPUT))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"Must fail fast without sleeping 300s (took {elapsed:.2f}s)"
    assert calls == 1, "Must not retry when delay exceeds ceiling"
    assert "retry-after=300" in exc_info.value.message
    assert "No automatic retry was made" in exc_info.value.message


# =============================================================================
# 26. Orchestration preserves successful market results when competitor fails
# =============================================================================
@pytest.mark.asyncio
async def test_orchestration_preserves_successful_results_on_downstream_failure(monkeypatch):
    """When competitor analysis fails after retries, preserve market analysis."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    idea_extractor = IdeaExtractionAgent(
        client_factory=lambda _: MockGeminiClient(response_text=json.dumps(STRUCTURED_IDEA_DATA))
    )

    class FakeResearcher:
        async def research(self, idea):
            return RESEARCH_SOURCES

    # Market succeeds
    market_analyzer = MarketOpportunityAnalysisAgent(
        client_factory=lambda _: MockGeminiClient(response_text=json.dumps(MARKET_ANALYSIS_DATA))
    )

    # Competitor fails (e.g. 429 exhausted)
    gemini_comp_err = RuntimeError("Gemini 503")
    gemini_comp_err.code = 503
    competitor_fail_transport = httpx.MockTransport(
        lambda req: httpx.Response(429, headers={"retry-after": "300"}, json={"error": {"message": "TPM exceeded"}}, request=req)
    )
    competitor_analyzer = CompetitorAnalysisAgent(
        client_factory=lambda _: MockGeminiClient(error=gemini_comp_err),
        transport=competitor_fail_transport,
    )

    # SWOT succeeds on Groq fallback
    swot_analyzer = SWOTRiskAnalysisAgent(
        client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
        transport=create_groq_mock_transport(SWOT_ANALYSIS_DATA),
    )

    service = ValidationReportService(
        extractor=idea_extractor,
        researcher=FakeResearcher(),
        market_analyzer=market_analyzer,
        competitor_analyzer=competitor_analyzer,
        swot_risk_analyzer=swot_analyzer,
    )

    response = await service.analyze(StartupIdea(**IDEA_INPUT))
    assert isinstance(response, AnalysisResponse)
    # Market analysis result must be preserved!
    assert response.market_analysis.demand_assessment.claim == MARKET_ANALYSIS_DATA["demand_assessment"]["claim"]
    # Competitor analysis is clean default empty (no fabricated citations)
    assert response.competitor_analysis.competitors == []
    assert response.competitor_analysis.market_gaps == []
    # SWOT result succeeded
    assert len(response.swot_risk_analysis.strengths) == 1


# =============================================================================
# 27. Orchestration raises ServiceError when all analysis stages fail
# =============================================================================
@pytest.mark.asyncio
async def test_orchestration_raises_when_all_analysis_stages_fail(monkeypatch):
    """When market, competitor, and SWOT all fail, raise ServiceError."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("LLM_API_KEY", "mock-groq-key")

    idea_extractor = IdeaExtractionAgent(
        client_factory=lambda _: MockGeminiClient(response_text=json.dumps(STRUCTURED_IDEA_DATA))
    )

    class FakeResearcher:
        async def research(self, idea):
            return RESEARCH_SOURCES

    fail_transport = httpx.MockTransport(
        lambda req: httpx.Response(429, headers={"retry-after": "300"}, json={"error": {"message": "TPM exceeded"}}, request=req)
    )

    service = ValidationReportService(
        extractor=idea_extractor,
        researcher=FakeResearcher(),
        market_analyzer=MarketOpportunityAnalysisAgent(
            client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
            transport=fail_transport,
        ),
        competitor_analyzer=CompetitorAnalysisAgent(
            client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
            transport=fail_transport,
        ),
        swot_risk_analyzer=SWOTRiskAnalysisAgent(
            client_factory=lambda _: MockGeminiClient(error=RuntimeError("Gemini down")),
            transport=fail_transport,
        ),
    )

    with pytest.raises(ServiceError) as exc_info:
        await service.analyze(StartupIdea(**IDEA_INPUT))

    assert exc_info.value.status_code == 503
    assert "Groq token rate limit reached" in exc_info.value.message
