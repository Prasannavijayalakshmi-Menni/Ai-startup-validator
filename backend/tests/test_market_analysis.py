import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError
from google import genai
from google.genai import types

from app.schemas import MarketOpportunityAnalysis, Source, StructuredIdea
from app.services.idea_extraction import ServiceError
from app.services.market_analysis import (
    GEMINI_MODEL,
    MarketOpportunityAnalysisAgent,
    _gemini_response_schema,
)
import app.services.market_analysis as market_analysis_module

SOURCE_URL = "https://research.example/market"
IDEA = StructuredIdea(
    problem="Small farms need reliable crop planning",
    proposed_solution="A crop planning assistant",
    target_customers=["Small farms"],
    industry="Agriculture",
    key_features=["Planning support"],
    value_proposition="Simpler planning",
    assumptions_and_missing_information=[],
)
RESEARCH = {
    "market_information": [
        Source(
            title="Farm planning report",
            url=SOURCE_URL,
            content="The supplied excerpt describes planning challenges reported by farms.",
        )
    ],
    "industry_trends": [],
}


def valid_result():
    return {
        "demand_assessment": {
            "claim": "The supplied excerpt describes crop planning challenges.",
            "classification": "evidence_based",
            "evidence": [SOURCE_URL],
        },
        "customer_segments": [
            {
                "segment": {
                    "claim": "Small farms",
                    "classification": "evidence_based",
                    "evidence": [SOURCE_URL],
                },
                "needs": [
                    {
                        "claim": "Planning support",
                        "classification": "hypothesis",
                        "evidence": [],
                    }
                ],
            }
        ],
        "opportunities": [
            {
                "claim": "A planning assistant may address this challenge.",
                "classification": "hypothesis",
                "evidence": [],
            }
        ],
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


@pytest.fixture(autouse=True)
def configured_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")


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
async def test_gemini_success_uses_teammate_model_and_validates_typed_response():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(valid_result()), captured=captured)

    analysis = await agent.analyze(IDEA, RESEARCH)

    assert isinstance(analysis, MarketOpportunityAnalysis)
    assert analysis.demand_assessment.evidence == [SOURCE_URL]
    assert analysis.customer_segments[0].needs[0].classification == "hypothesis"
    assert analysis.challenges[0].claim == "Unknown"
    assert captured["api_key_received"] == "mock-gemini-key"
    request = captured["request"]
    assert request["model"] == GEMINI_MODEL == "gemini-3.8-flash"
    assert request["config"]["response_mime_type"] == "application/json"
    schema = request["config"]["response_schema"]
    assert schema["title"] == "MarketOpportunityAnalysis"
    assert "demand_assessment" in schema["required"]
    assert "additionalProperties" not in json.dumps(schema)
    assert request["config"]["system_instruction"]
    prompt = request["contents"]
    assert "one valid JSON object only" in prompt
    assert "Do not invent statistics" in prompt
    assert "Do not include or reinterpret the submitted budget" in prompt
    assert "estimated_budget" not in prompt
    assert "50000" not in prompt
    assert SOURCE_URL in prompt
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_sdk_serializes_supported_response_schema_without_extra_properties():
    serialized = {}

    def handler(request):
        serialized["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": json.dumps(valid_result())}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            request=request,
        )

    client = genai.Client(
        api_key="mock-only-key",
        http_options=types.HttpOptions(
            async_client_args={
                "transport": httpx.MockTransport(handler),
                "trust_env": False,
            }
        ),
    )
    agent = MarketOpportunityAnalysisAgent(client_factory=lambda _: client)

    try:
        analysis = await agent.analyze(IDEA, RESEARCH)
    finally:
        await client.aio.aclose()
        client.close()

    assert isinstance(analysis, MarketOpportunityAnalysis)
    request_schema = serialized["body"]["generationConfig"]["responseSchema"]
    assert request_schema["type"] == "OBJECT"
    assert "demand_assessment" in request_schema["required"]
    assert "additionalProperties" not in json.dumps(request_schema)
    assert "additional_properties" not in json.dumps(request_schema)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {key: value for key, value in valid_result().items() if key != "demand_assessment"},
        {**valid_result(), "unexpected": True},
        {**valid_result(), "customer_segments": "not an array"},
        {**valid_result(), "opportunities": [{"claim": "missing required classification"}]},
    ],
)
async def test_gemini_invalid_schema_response_is_rejected_without_fabrication(invalid):
    agent, _, _ = make_agent(json.dumps(invalid))

    with pytest.raises(ServiceError, match="required schema"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_gemini_unsupported_citation_is_rejected():
    invalid = valid_result()
    invalid["demand_assessment"]["evidence"] = ["https://outside.example/claim"]
    agent, _, _ = make_agent(json.dumps(invalid))

    with pytest.raises(ServiceError, match="unsupported source URL"):
        await agent.analyze(IDEA, RESEARCH)


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
    caplog.set_level(logging.WARNING, logger="app.services.market_analysis")

    with pytest.raises(ServiceError, match=expected) as raised:
        await agent.analyze(IDEA, RESEARCH)

    assert raised.value.status_code == status_code
    assert "mock-gemini-key" not in caplog.text
    assert "sensitive provider message" in caplog.text
    assert f"code={status}" in caplog.text


@pytest.mark.asyncio
async def test_gemini_error_diagnostic_redacts_credentials_research_and_logs_request_id(caplog):
    error = RuntimeError("provider error")
    error.code = 400
    error.status = "INVALID_ARGUMENT"
    error.message = (
        f"Invalid response schema for {SOURCE_URL}; "
        f"{RESEARCH['market_information'][0].content}; "
        "api_key=mock-gemini-key Bearer mock-gemini-key"
    )
    error.response = SimpleNamespace(
        headers={"x-goog-request-id": "gemini-test-request-123"}
    )
    agent, _, _ = make_agent(error=error)
    caplog.set_level(logging.WARNING, logger="app.services.market_analysis")

    with pytest.raises(ServiceError, match="Invalid response schema") as raised:
        await agent.analyze(IDEA, RESEARCH)

    assert "gemini-test-request-123" in caplog.text
    assert "INVALID_ARGUMENT" in caplog.text
    assert "mock-gemini-key" not in caplog.text
    assert SOURCE_URL not in caplog.text
    assert RESEARCH["market_information"][0].content not in caplog.text
    assert "[REDACTED]" in str(raised.value)


@pytest.mark.asyncio
async def test_prompt_excludes_competitor_sources_and_request_logs_only_safe_metadata(caplog):
    competitor_only_text = "COMPETITOR_ONLY_TEXT_MUST_NOT_ENTER_MARKET_PROMPT"
    research = {
        **RESEARCH,
        "competitors_and_alternatives": [
            Source(
                title="Alternative listing",
                url="https://research.example/competitors",
                content=competitor_only_text,
            )
        ],
    }
    agent, captured, _ = make_agent(json.dumps(valid_result()))
    caplog.set_level(logging.INFO, logger="app.services.market_analysis")

    await agent.analyze(IDEA, research)

    assert competitor_only_text not in captured["request"]["contents"]
    assert "market_information" not in captured["request"]["contents"]
    assert "source_count=1" in caplog.text
    assert "source_content_chars=" in caplog.text
    assert "system_prompt_chars=" in caplog.text
    assert "user_prompt_chars=" in caplog.text
    assert "rough_input_tokens=" in caplog.text
    assert "max_completion_tokens=provider_default" in caplog.text
    assert competitor_only_text not in caplog.text
    assert SOURCE_URL not in caplog.text


@pytest.mark.asyncio
async def test_missing_research_returns_unknown_without_gemini_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent = MarketOpportunityAnalysisAgent(
        client_factory=lambda _: (_ for _ in ()).throw(AssertionError("unexpected Gemini call"))
    )

    analysis = await agent.analyze(
        IDEA,
        {
            "competitors_and_alternatives": [
                Source(
                    title="Competitor result",
                    url="https://research.example/competitor-only",
                    content="This source is for the downstream competitor agent.",
                )
            ]
        },
    )

    assert analysis.demand_assessment.claim == "Unknown"
    assert analysis.demand_assessment.classification == "unknown"
    assert analysis.customer_segments == []


@pytest.mark.asyncio
async def test_missing_gemini_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ServiceError, match="GEMINI_API_KEY"):
        await MarketOpportunityAnalysisAgent().analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", "", "null"])
async def test_gemini_invalid_json_or_empty_response_fails_clearly(content):
    agent, _, _ = make_agent(content)
    with pytest.raises(ServiceError, match="Gemini"):
        await agent.analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_gemini_timeout_and_connection_failures_are_distinct():
    agent, _, _ = make_agent(error=httpx.ConnectError("offline"))
    with pytest.raises(ServiceError, match="Could not connect") as connection_error:
        await agent.analyze(IDEA, RESEARCH)
    assert connection_error.value.status_code == 502

    agent, _, _ = make_agent(error=httpx.ReadTimeout("timeout"))
    with pytest.raises(ServiceError, match="timed out") as timeout_error:
        await agent.analyze(IDEA, RESEARCH)
    assert timeout_error.value.status_code == 504


@pytest.mark.asyncio
async def test_market_analysis_outer_deadline_returns_504_without_provider_response(monkeypatch):
    client = FakeGeminiClient()

    async def wait_forever(**kwargs):
        await asyncio.Event().wait()

    client.aio.models.generate_content = wait_forever
    original_wait_for = asyncio.wait_for

    async def short_diagnostic_deadline(awaitable, timeout=None):
        return await original_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr(
        market_analysis_module,
        "asyncio",
        SimpleNamespace(
            wait_for=short_diagnostic_deadline,
            TimeoutError=asyncio.TimeoutError,
        ),
    )
    agent = MarketOpportunityAnalysisAgent(client_factory=lambda _: client)

    with pytest.raises(ServiceError, match="Gemini market analysis request timed out") as exc:
        await agent.analyze(IDEA, RESEARCH)

    assert exc.value.status_code == 504
    client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_market_analysis_timeout_logs_safe_timing_metadata(monkeypatch, caplog):
    client = FakeGeminiClient()

    async def wait_forever(**kwargs):
        await asyncio.Event().wait()

    client.aio.models.generate_content = wait_forever
    original_wait_for = asyncio.wait_for

    async def short_diagnostic_deadline(awaitable, timeout=None):
        return await original_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr(
        market_analysis_module,
        "asyncio",
        SimpleNamespace(
            wait_for=short_diagnostic_deadline,
            TimeoutError=asyncio.TimeoutError,
        ),
    )
    caplog.set_level(logging.INFO, logger="app.services.market_analysis")

    with pytest.raises(ServiceError, match="timed out"):
        await MarketOpportunityAnalysisAgent(client_factory=lambda _: client).analyze(IDEA, RESEARCH)

    assert "request_started_utc=" in caplog.text
    assert "elapsed_seconds=" in caplog.text
    assert f"model={GEMINI_MODEL}" in caplog.text
    assert "exception_type=TimeoutError" in caplog.text
    assert "provider_http_status=unavailable" in caplog.text
    assert "mock-gemini-key" not in caplog.text
    assert SOURCE_URL not in caplog.text
    assert RESEARCH["market_information"][0].content not in caplog.text


def test_pydantic_schema_requires_evidence_and_rejects_unprovided_urls():
    payload = valid_result()
    payload["demand_assessment"]["evidence"] = []
    with pytest.raises(ValidationError, match="must cite at least one"):
        MarketOpportunityAnalysis.model_validate(
            payload, context={"allowed_urls": {SOURCE_URL}}
        )

    payload = valid_result()
    payload["demand_assessment"]["evidence"] = ["https://outside.example/claim"]
    with pytest.raises(ValidationError, match="absent from supplied research"):
        MarketOpportunityAnalysis.model_validate(
            payload, context={"allowed_urls": {SOURCE_URL}}
        )
