"""Offline competitor-agent checks using fictional, explicitly synthetic fixtures."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.competitor_analysis import CompetitorAnalysisAgent
from app.services.idea_extraction import ServiceError


DIRECT_URL = "https://research.example.test/direct-study-planner"
INDIRECT_URL = "https://research.example.test/peer-study-groups"
UNKNOWN_URL = "https://research.example.test/minimal-listing"
SOURCE_URLS = {DIRECT_URL, INDIRECT_URL, UNKNOWN_URL}

IDEA = StructuredIdea(
    problem="High school students struggle to adapt study plans when exams move.",
    proposed_solution="An AI planner that adapts short study sessions.",
    target_customers=["High school students"],
    industry="Education technology",
    key_features=["Adaptive scheduling"],
    value_proposition="A study schedule that adjusts to changing exam dates.",
    assumptions_and_missing_information=[],
)

# Fictional source records for mocked tests only. They are not live research,
# real companies, customer data, or claims about the education market.
SOURCES = [
    Source(
        title="SYNTHETIC TEST SOURCE: fictional ClassMate listing",
        url=DIRECT_URL,
        content=(
            "Synthetic fixture only, not real research. Fictional ClassMate "
            "provides weekly study schedules for high school students. Its test "
            "listing explicitly says schedules are fixed and cannot be adjusted "
            "when exam dates change."
        ),
    ),
    Source(
        title="SYNTHETIC TEST SOURCE: fictional StudyCircle listing",
        url=INDIRECT_URL,
        content=(
            "Synthetic fixture only, not real research. Fictional StudyCircle "
            "offers peer-led group sessions and shared reminders to help "
            "students keep up with study routines."
        ),
    ),
    Source(
        title="SYNTHETIC TEST SOURCE: fictional MinimalPlanner catalog entry",
        url=UNKNOWN_URL,
        content=(
            "Synthetic fixture only, not real research. Fictional MinimalPlanner "
            "is listed by name, but the test entry contains no details about "
            "its offering, customer group, or use case."
        ),
    ),
]

RESEARCH = {"competitors_and_alternatives": SOURCES}

MARKET_ANALYSIS = MarketOpportunityAnalysis(
    demand_assessment={
        "claim": "Synthetic test context: students need adaptable study plans.",
        "classification": "hypothesis",
        "evidence": [],
    },
    customer_segments=[
        {
            "segment": {
                "claim": "High school students",
                "classification": "hypothesis",
                "evidence": [],
            },
            "needs": [
                {
                    "claim": "Adjust study sessions when exam dates change.",
                    "classification": "hypothesis",
                    "evidence": [],
                }
            ],
        }
    ],
    opportunities=[
        {
            "claim": "Test whether adaptable schedules address this need.",
            "classification": "hypothesis",
            "evidence": [],
        }
    ],
)

MOCK_RESULT = {
    "competitors": [
        {
            "name": "ClassMate",
            "competitor_type": "direct",
            "offering": "Fixed weekly study schedules",
            "target_customers": "High school students",
            "pricing": "Unknown",
            "strengths": ["Provides weekly schedules"],
            "limitations": ["The synthetic listing says schedules cannot adjust to exam-date changes."],
            "evidence": [DIRECT_URL],
        },
        {
            "name": "StudyCircle",
            "competitor_type": "indirect",
            "offering": "Peer-led group sessions and reminders",
            "target_customers": "Students",
            "pricing": "Unknown",
            "strengths": ["Offers group study support"],
            "limitations": [],
            "evidence": [INDIRECT_URL],
        },
        {
            "name": "MinimalPlanner",
            "competitor_type": "unknown",
            "offering": "Unknown",
            "target_customers": "Unknown",
            "pricing": "Unknown",
            "strengths": [],
            "limitations": [],
            "evidence": [UNKNOWN_URL],
        },
    ],
    "market_gaps": [
        {
            "description": "The reviewed fictional listing describes schedules that cannot adapt to exam-date changes.",
            "rationale": "That limitation is explicitly stated in the supplied synthetic ClassMate source.",
            "gap_type": "evidence_based",
            "evidence": [DIRECT_URL],
        },
        {
            "description": "Adaptive schedules may better support students when exam dates change.",
            "rationale": "This is a hypothesis from the synthetic idea and context, not a verified market finding.",
            "gap_type": "hypothesis",
            "evidence": [],
        },
    ],
}


@pytest.fixture(autouse=True)
def configure_mock_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
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

    def factory(api_key):
        return FakeGeminiClient(response=response, error=error, captured=captured)

    return CompetitorAnalysisAgent(client_factory=factory), captured


@pytest.mark.asyncio
async def test_synthetic_sources_cover_classification_comparison_gaps_and_market_context():
    captured = {}
    agent, captured = make_agent(json.dumps(MOCK_RESULT), captured=captured)

    result = await agent.analyze(IDEA, RESEARCH, MARKET_ANALYSIS)

    prompt = captured["request"]["contents"]
    assert "SYNTHETIC TEST SOURCE" in prompt
    assert "not real research" in prompt
    assert "High school students" in prompt
    assert "Adjust study sessions when exam dates change." in prompt
    assert all(url in prompt for url in SOURCE_URLS)
    assert "estimated_budget" not in prompt

    assert [competitor.competitor_type for competitor in result.competitors] == [
        "direct",
        "indirect",
        "unknown",
    ]
    assert result.competitors[0].offering == "Fixed weekly study schedules"
    assert result.competitors[1].offering == "Peer-led group sessions and reminders"
    assert result.competitors[2].offering == "Unknown"
    assert result.market_gaps[0].gap_type == "evidence_based"
    assert result.market_gaps[0].evidence == [DIRECT_URL]
    assert result.market_gaps[1].gap_type == "hypothesis"
    citations = {
        url
        for item in [*result.competitors, *result.market_gaps]
        for url in item.evidence
    }
    assert citations <= SOURCE_URLS


@pytest.mark.asyncio
async def test_synthetic_fixture_rejects_citations_outside_supplied_urls():
    invalid_result = {
        "competitors": [
            {
                **MOCK_RESULT["competitors"][0],
                "evidence": ["https://unsupported.example.test/fabricated"],
            }
        ],
        "market_gaps": [],
    }
    agent, _ = make_agent(json.dumps(invalid_result))

    with pytest.raises(ServiceError, match="unsupported source URL"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_empty_synthetic_competitor_research_skips_provider_request(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def unexpected_factory(api_key):
        raise AssertionError("empty competitor research must not call the provider")

    result = await CompetitorAnalysisAgent(
        client_factory=unexpected_factory
    ).analyze(IDEA, {"competitors_and_alternatives": []}, MARKET_ANALYSIS)

    assert result.competitors == []
    assert result.market_gaps == []
