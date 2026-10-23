import json
import logging

import httpx
import pytest
from pydantic import ValidationError

from app.schemas import (
    MarketOpportunityAnalysis,
    Source,
    StructuredIdea,
)
from app.services.idea_extraction import ServiceError
from app.services.market_analysis import (
    MarketOpportunityAnalysisAgent,
    _market_response_format,
)

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
            {
                "claim": "Unknown",
                "classification": "unknown",
                "evidence": [],
            }
        ],
    }


@pytest.fixture(autouse=True)
def configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "mock-market-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")


def transport_with(payload, status=200):
    return httpx.MockTransport(
        lambda request: httpx.Response(status, json=payload, request=request)
    )


def completion(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_market_analysis_uses_json_object_mode_and_validates_result_locally():
    result = valid_result()
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(json.dumps(result)), request=request)

    analysis = await MarketOpportunityAnalysisAgent(
        transport=httpx.MockTransport(handler)
    ).analyze(IDEA, RESEARCH)

    assert isinstance(analysis, MarketOpportunityAnalysis)
    assert analysis.demand_assessment.evidence == [SOURCE_URL]
    assert analysis.customer_segments[0].needs[0].classification == "hypothesis"
    assert analysis.challenges[0].claim == "Unknown"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["body"]["model"] == "openai/gpt-oss-20b"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == _market_response_format()
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0]["role"] == "system"
    prompt = seen["body"]["messages"][1]["content"]
    assert "Do not invent statistics" in prompt
    assert "Do not include or reinterpret the submitted budget" in prompt
    assert "estimated_budget" not in prompt
    assert "50000" not in prompt
    assert "valid JSON object only" in prompt


@pytest.mark.asyncio
async def test_market_request_metadata_is_safe_and_competitor_sources_are_not_duplicated(caplog):
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
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(json.dumps(valid_result())), request=request)

    caplog.set_level(logging.INFO, logger="app.services.market_analysis")
    await MarketOpportunityAnalysisAgent(
        transport=httpx.MockTransport(handler)
    ).analyze(IDEA, research)

    prompt = seen["body"]["messages"][1]["content"]
    assert competitor_only_text not in prompt
    assert "market_information" not in prompt
    assert "source_count=1" in caplog.text
    assert "source_content_chars=" in caplog.text
    assert "system_prompt_chars=" in caplog.text
    assert "user_prompt_chars=" in caplog.text
    assert "rough_input_tokens=" in caplog.text
    assert "max_completion_tokens=provider_default" in caplog.text
    assert competitor_only_text not in caplog.text
    assert SOURCE_URL not in caplog.text


@pytest.mark.asyncio
async def test_provider_400_logs_redacted_metadata_without_generation_or_prompt(caplog):
    generated_content = "SENSITIVE_GENERATED_CONTENT"
    provider_message = (
        f"JSON validation failed for {RESEARCH['market_information'][0].content}; "
        "api_key=mock-market-key Bearer mock-market-key"
    )

    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "json_validate_failed",
                    "message": provider_message,
                    "failed_generation": generated_content,
                }
            },
            request=request,
        )

    caplog.set_level(logging.WARNING, logger="app.services.market_analysis")
    with pytest.raises(ServiceError, match="HTTP 400"):
        await MarketOpportunityAnalysisAgent(
            transport=httpx.MockTransport(handler)
        ).analyze(IDEA, RESEARCH)

    assert "status=400" in caplog.text
    assert "type=invalid_request_error" in caplog.text
    assert "code=json_validate_failed" in caplog.text
    assert "failed_generation_present=True" in caplog.text
    assert f"failed_generation_chars={len(generated_content)}" in caplog.text
    assert "mock-market-key" not in caplog.text
    assert generated_content not in caplog.text
    assert RESEARCH["market_information"][0].content not in caplog.text
    assert SOURCE_URL not in caplog.text


@pytest.mark.asyncio
async def test_previous_groq_json_validation_error_is_not_retried():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "json_validate_failed",
                    "message": "Failed to validate JSON.",
                    "failed_generation": "",
                }
            },
            request=request,
        )

    with pytest.raises(ServiceError, match="HTTP 400"):
        await MarketOpportunityAnalysisAgent(
            transport=httpx.MockTransport(handler)
        ).analyze(IDEA, RESEARCH)

    assert len(requests) == 1
    assert requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_competitor_only_research_returns_unknown_without_provider_call():
    agent = MarketOpportunityAnalysisAgent(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("unexpected provider call"))
        )
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


def test_market_schema_requires_evidence_and_rejects_unprovided_urls():
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({**valid_result(), "unexpected": True}),
        json.dumps({**valid_result(), "customer_segments": "not an array"}),
        json.dumps({**valid_result(), "opportunities": [{"claim": "missing fields"}]}),
    ],
)
async def test_invalid_provider_outputs_fail_without_fabricating_analysis(content):
    agent = MarketOpportunityAnalysisAgent(
        transport=transport_with(completion(content))
    )
    with pytest.raises(ServiceError, match="market analysis"):
        await agent.analyze(IDEA, RESEARCH)


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
async def test_market_provider_errors_are_reported(status, expected):
    with pytest.raises(ServiceError, match=expected):
        await MarketOpportunityAnalysisAgent(
            transport=transport_with({"error": {"message": "provider failure"}}, status)
        ).analyze(IDEA, RESEARCH)


@pytest.mark.asyncio
async def test_rate_limit_code_is_logged_without_changing_public_error(caplog):
    response = {
        "error": {
            "type": "tokens",
            "code": "rate_limit_exceeded",
            "message": "TPM limit reached; try again in 12 seconds.",
        }
    }
    caplog.set_level(logging.WARNING, logger="app.services.market_analysis")
    with pytest.raises(ServiceError, match="rate limit") as exc:
        await MarketOpportunityAnalysisAgent(
            transport=transport_with(response, status=429)
        ).analyze(IDEA, RESEARCH)

    assert exc.value.status_code == 503
    assert "type=tokens" in caplog.text
    assert "code=rate_limit_exceeded" in caplog.text


@pytest.mark.asyncio
async def test_market_provider_network_and_timeout_errors_are_distinct():
    def connect(request):
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(ServiceError, match="Could not connect") as exc:
        await MarketOpportunityAnalysisAgent(
            transport=httpx.MockTransport(connect)
        ).analyze(IDEA, RESEARCH)
    assert exc.value.status_code == 502

    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(ServiceError, match="timed out") as exc:
        await MarketOpportunityAnalysisAgent(
            transport=httpx.MockTransport(timeout)
        ).analyze(IDEA, RESEARCH)
    assert exc.value.status_code == 504
