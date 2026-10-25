import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from google import genai
from google.genai import types

from app.schemas import Source, StructuredIdea
from app.services.market_analysis import GEMINI_MODEL
from app.services.gemini_market_background_diagnostic import (
    run_market_analysis_background_diagnostic,
)

API_KEY = "background-diagnostic-only-key"
SOURCE_URL = "https://example.invalid/synthetic-background-market"
IDEA = StructuredIdea(
    problem="Synthetic background test problem",
    proposed_solution="Synthetic background test solution",
    target_customers=["Synthetic users"],
    industry="Synthetic sector",
    key_features=["Synthetic feature"],
    value_proposition="Synthetic value",
    assumptions_and_missing_information=[],
)
RESEARCH = {
    "market_information": [
        Source(
            title="Synthetic background source",
            url=SOURCE_URL,
            content="Synthetic evidence for mocked background interaction tests.",
        )
    ]
}


def valid_result(url=SOURCE_URL):
    return {
        "demand_assessment": {
            "claim": "Synthetic demand claim",
            "classification": "evidence_based",
            "evidence": [url],
        },
        "customer_segments": [],
        "opportunities": [],
        "challenges": [],
    }


def interaction(status, output=None):
    steps = []
    if output is not None:
        steps = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(output)}],
            }
        ]
    return {
        "id": "mock-interaction-id",
        "status": status,
        "model": GEMINI_MODEL,
        "created": "2026-10-23T00:00:00Z",
        "updated": "2026-10-23T00:00:00Z",
        "steps": steps,
    }


def sdk_client(handler):
    return genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            async_client_args={
                "transport": httpx.MockTransport(handler),
                "trust_env": False,
            }
        ),
    )


@pytest.mark.asyncio
async def test_background_interaction_serialization_polling_and_validated_output():
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=interaction("in_progress"), request=request)
        if len(requests) == 2:
            return httpx.Response(200, json=interaction("in_progress"), request=request)
        return httpx.Response(200, json=interaction("completed", valid_result()), request=request)

    client = sdk_client(handler)
    result = await run_market_analysis_background_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        total_timeout_seconds=3,
        poll_interval_seconds=0,
        client_factory=lambda _: client,
    )

    request = requests[0]
    body = json.loads(request.content)
    assert request.method == "POST"
    assert request.url.path.endswith("/interactions")
    assert request.headers["x-goog-api-key"] == API_KEY
    assert "authorization" not in request.headers
    assert API_KEY not in str(request.url)
    assert body["model"] == GEMINI_MODEL
    assert body["background"] is True
    assert body["store"] is True
    assert body["generation_config"] == {"thinking_level": "low"}
    assert body["response_format"]["type"] == "text"
    assert body["response_format"]["mime_type"] == "application/json"
    schema_text = json.dumps(body["response_format"]["schema"])
    assert "$defs" not in schema_text
    assert "$ref" not in schema_text
    assert "additionalProperties" not in schema_text
    assert result.interaction_id == "mock-interaction-id"
    assert result.status == "completed"
    assert result.analysis is not None
    assert result.analysis.demand_assessment.evidence == [SOURCE_URL]
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_background_failed_interaction_returns_failure_without_parsing_output():
    client = sdk_client(
        lambda request: httpx.Response(200, json=interaction("failed"), request=request)
    )
    result = await run_market_analysis_background_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        client_factory=lambda _: client,
    )
    assert result.status == "failed"
    assert result.analysis is None
    assert result.error_type == "InteractionFailed"


@pytest.mark.asyncio
async def test_background_total_timeout_is_bounded_during_polling():
    async def slow_get(**kwargs):
        await asyncio.Event().wait()

    client = SimpleNamespace()
    client.aio = SimpleNamespace()
    client.aio.aclose = AsyncMock()
    client.aio.interactions = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="slow-id", status="in_progress")),
        get=AsyncMock(side_effect=slow_get),
        aclose=AsyncMock(),
    )
    client.close = lambda: None

    result = await run_market_analysis_background_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        total_timeout_seconds=0.5,
        poll_interval_seconds=0,
        client_factory=lambda _: client,
    )

    assert result.status == "timeout"
    assert result.error_type == "TimeoutError"
    assert result.analysis is None
    client.aio.interactions.get.assert_awaited_once_with(id="slow-id")
    client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_output_rejects_citations_outside_supplied_research():
    client = sdk_client(
        lambda request: httpx.Response(
            200,
            json=interaction("completed", valid_result("https://outside.invalid/not-supplied")),
            request=request,
        )
    )
    result = await run_market_analysis_background_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        client_factory=lambda _: client,
    )
    assert result.status == "completed"
    assert result.analysis is None
    assert result.error_type == "ResponseValidationError"
    assert "citation validation" in result.safe_error
