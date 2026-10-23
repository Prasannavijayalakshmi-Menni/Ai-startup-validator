import json
import os

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
                "strengths": ["Provides study schedules"],
                "limitations": [],
                "evidence": [SOURCE_URL],
            },
            {
                "name": "PlannerPlus",
                "competitor_type": "unknown",
                "offering": "Unknown",
                "target_customers": "Unknown",
                "pricing": None,
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


@pytest.fixture(autouse=True)
def configured_mock_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "mock-llm-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")


def response_transport(payload, status=200):
    return httpx.MockTransport(
        lambda request: httpx.Response(
            status,
            json=payload,
            request=request,
        )
    )


@pytest.mark.asyncio
async def test_competitor_agent_returns_typed_multi_competitor_result_and_unknown_pricing():
    calls = []

    def handler(request):
        assert request.method == "POST"
        configured_base_url = os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")
        assert str(request.url) == f"{configured_base_url}/chat/completions"
        assert request.headers["Authorization"] == "Bearer mock-llm-key"
        body = json.loads(request.content)
        assert set(body) == {
            "model", "temperature", "response_format", "messages"
        }
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["temperature"] == 0
        expected_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "competitor_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "competitors": {
                            "type": "array",
                            "items": {
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
                                    "strengths": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "limitations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": [SOURCE_URL],
                                        },
                                    },
                                },
                                "required": [
                                    "name", "competitor_type", "offering", "target_customers", "pricing",
                                    "strengths", "limitations", "evidence",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "market_gaps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "rationale": {"type": "string"},
                                    "gap_type": {
                                        "type": "string",
                                        "enum": ["evidence_based", "hypothesis"],
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": [SOURCE_URL],
                                        },
                                    },
                                },
                                "required": [
                                    "description", "rationale", "gap_type", "evidence"
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["competitors", "market_gaps"],
                    "additionalProperties": False,
                },
            },
        }
        assert body["response_format"] == expected_schema

        expected_system = (
            "Return exactly one valid JSON object and nothing else. "
            "Do not use Markdown fences or surrounding text. "
            "Never fabricate facts or source URLs."
        )
        expected_research = json.dumps(
            [{"title": source.title, "url": source.url, "content": source.content}
             for source in SOURCES],
            ensure_ascii=False,
        )
        expected_market_analysis = json.dumps(
            MARKET_ANALYSIS.model_dump(mode="json"), ensure_ascii=False
        )
        expected_prompt = f"""
Analyze the startup idea and the supplied competitor research.

Startup idea:
{IDEA.model_dump_json()}

Market opportunity and customer analysis:
{expected_market_analysis}

Research sources:
{expected_research}

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
        assert expected_research in expected_prompt
        expected_messages = [
            {"role": "system", "content": expected_system},
            {"role": "user", "content": expected_prompt},
        ]
        assert body["messages"] == expected_messages
        prompt = body["messages"][1]["content"]
        calls.append(prompt)
        assert SOURCE_URL in prompt
        assert "50000" not in prompt
        assert "estimated_budget" not in prompt
        assert "College students" in prompt
        assert "exactly one syntactically valid JSON object" in prompt
        assert "Do not include Markdown" in prompt
        assert '"strengths": array of strings' in prompt
        assert '"limitations": array of strings' in prompt
        assert '"gap_type": exactly "evidence_based" or "hypothesis"' in prompt
        assert '"pricing": string; use exactly "Unknown"' in prompt
        assert '"competitors":[],"market_gaps":[]' in prompt
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(valid_payload())}}]},
            request=request,
        )

    result = await CompetitorAnalysisAgent(
        transport=httpx.MockTransport(handler)
    ).analyze(IDEA, RESEARCH, MARKET_ANALYSIS)

    assert isinstance(result, CompetitorAnalysis)
    assert len(result.competitors) == 3
    assert result.competitors[0].evidence == [SOURCE_URL]
    assert result.competitors[0].competitor_type == "direct"
    assert result.competitors[1].pricing == "Unknown"
    assert result.competitors[1].competitor_type == "unknown"
    assert result.competitors[2].competitor_type == "indirect"
    assert result.market_gaps[0].gap_type == "hypothesis"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_competitor_sources_returns_empty_without_llm_request():
    calls = []
    agent = CompetitorAnalysisAgent(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    )
    result = await agent.analyze(IDEA, {"competitors_and_alternatives": []})
    assert result.competitors == []
    assert result.market_gaps == []
    assert calls == []


@pytest.mark.asyncio
async def test_missing_market_analysis_is_passed_as_unknown_context():
    captured = {}

    def handler(request):
        captured["prompt"] = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"competitors": [], "market_gaps": []})}}]},
            request=request,
        )

    result = await CompetitorAnalysisAgent(
        transport=httpx.MockTransport(handler)
    ).analyze(IDEA, RESEARCH)
    assert result.competitors == []
    assert '"claim": "Unknown"' in captured["prompt"]
    assert '"customer_segments": []' in captured["prompt"]


@pytest.mark.asyncio
async def test_missing_llm_configuration_is_clear(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(ServiceError, match="LLM_API_KEY"):
        await CompetitorAnalysisAgent().analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_malformed_json_is_rejected():
    transport = response_transport(
        {"choices": [{"message": {"content": "not json"}}]}
    )
    with pytest.raises(ServiceError, match="invalid competitor analysis"):
        await CompetitorAnalysisAgent(transport=transport).analyze(IDEA, RESEARCH)


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
    transport = response_transport(
        {"choices": [{"message": {"content": json.dumps(payload)}}]}
    )
    with pytest.raises(ServiceError, match=expected):
        await CompetitorAnalysisAgent(transport=transport).analyze(IDEA, RESEARCH)


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


@pytest.mark.parametrize(
    "status,detail",
    [(400, "request"), (401, "credential"), (429, "rate limit")],
)
@pytest.mark.asyncio
async def test_provider_http_errors_are_categorized(status, detail):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": "mock error"}, request=request)
    )
    with pytest.raises(ServiceError, match=detail):
        await CompetitorAnalysisAgent(transport=transport).analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_http_400_logs_structured_provider_reason_without_credentials(caplog):
    provider_reason = (
        "response_format is unsupported; Bearer mock-llm-key gsk_abc123 token=secret-value"
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "unsupported_parameter",
                    "message": provider_reason,
                }
            },
            request=request,
        )
    )
    caplog.set_level("WARNING", logger="app.services.competitor_analysis")

    with pytest.raises(ServiceError, match="rejected the competitor analysis request"):
        await CompetitorAnalysisAgent(transport=transport).analyze(IDEA, RESEARCH)

    assert "unsupported_parameter" in caplog.text
    assert "response_format is unsupported" in caplog.text
    assert "mock-llm-key" not in caplog.text
    assert "gsk_abc123" not in caplog.text
    assert "secret-value" not in caplog.text


@pytest.mark.asyncio
async def test_json_validation_error_logs_only_failed_generation_diagnosis(caplog):
    rejected_output = '{"competitors": [oops]}'
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "json_validate_failed",
                    "message": "Failed to validate JSON. Please adjust your prompt.",
                    "failed_generation": rejected_output,
                }
            },
            request=request,
        )
    )
    caplog.set_level("WARNING", logger="app.services.competitor_analysis")

    with pytest.raises(ServiceError, match="rejected the competitor analysis request") as exc:
        await CompetitorAnalysisAgent(transport=transport).analyze(IDEA, RESEARCH)

    assert exc.value.status_code == 502
    assert "json_validate_failed" in caplog.text
    assert "failed_generation_diagnosis=invalid_json line=1" in caplog.text
    assert rejected_output not in caplog.text


@pytest.mark.parametrize(
    "error_type,expected,status",
    [
        (httpx.ReadTimeout, "timed out", 504),
        (httpx.ConnectError, "Could not connect", 502),
    ],
)
@pytest.mark.asyncio
async def test_provider_timeout_and_connection_failures(error_type, expected, status):
    def handler(request):
        raise error_type("mock provider failure", request=request)

    with pytest.raises(ServiceError, match=expected) as exc:
        await CompetitorAnalysisAgent(
            transport=httpx.MockTransport(handler)
        ).analyze(IDEA, RESEARCH)
    assert exc.value.status_code == status


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
    def extraction_handler(request):
        prompt = json.loads(request.content)["messages"][1]["content"]
        assert "50000" not in prompt
        assert "estimated_budget" not in prompt
        assert "estimated_budget_currency" not in prompt
        return httpx.Response(200, json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
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
                    }
                }
            ]
        }, request=request)

    extraction_transport = httpx.MockTransport(extraction_handler)

    class Extractor:
        def __init__(self):
            self.agent = IdeaExtractionAgent(transport=extraction_transport)

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
