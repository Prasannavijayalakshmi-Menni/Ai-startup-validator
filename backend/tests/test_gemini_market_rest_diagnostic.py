import json

import httpx
import pytest

from app.schemas import Source, StructuredIdea
from app.services.gemini_market_rest_diagnostic import run_market_analysis_rest_diagnostic

API_KEY = "diagnostic-only-key"
SOURCE_URL = "https://example.invalid/synthetic-market"
IDEA = StructuredIdea(
    problem="Synthetic diagnostic problem",
    proposed_solution="Synthetic diagnostic solution",
    target_customers=["Synthetic customers"],
    industry="Synthetic industry",
    key_features=["Synthetic feature"],
    value_proposition="Synthetic value",
    assumptions_and_missing_information=[],
)
RESEARCH = {
    "market_information": [
        Source(
            title="Synthetic diagnostic source",
            url=SOURCE_URL,
            content="Synthetic content for request-format validation only.",
        )
    ]
}


def valid_result(evidence_url=SOURCE_URL):
    return {
        "demand_assessment": {
            "claim": "Synthetic demand claim",
            "classification": "evidence_based",
            "evidence": [evidence_url],
        },
        "customer_segments": [],
        "opportunities": [],
        "challenges": [],
    }


def ok_response(request, result=None):
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(result or valid_result())}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        },
        request=request,
    )


@pytest.mark.asyncio
async def test_rest_diagnostic_sends_exact_camel_case_json_and_validates_response(monkeypatch):
    monkeypatch.setenv("HTTPX_TRUST_ENV", "false")
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        return ok_response(request)

    result = await run_market_analysis_rest_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    body = captured["body"]
    generation = body["generationConfig"]
    assert generation["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert set(generation) == {"responseMimeType", "responseSchema", "thinkingConfig"}
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseSchema"]["title"] == "MarketOpportunityAnalysis"
    schema_json = json.dumps(generation["responseSchema"])
    assert "$defs" not in schema_json
    assert "$ref" not in schema_json
    assert "additionalProperties" not in schema_json
    assert generation["responseSchema"]["required"] == ["demand_assessment"]
    assert set(generation["responseSchema"]["properties"]) == {
        "demand_assessment",
        "customer_segments",
        "opportunities",
        "challenges",
    }
    assert body["contents"][0]["role"] == "user"
    assert "apiKey" not in body and "key" not in body
    assert "key=" not in captured["url"]
    assert API_KEY not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == API_KEY
    assert "authorization" not in captured["headers"]
    assert result.status_code == 200
    assert result.error_type is None
    assert result.analysis is not None
    assert result.analysis.demand_assessment.evidence == [SOURCE_URL]


@pytest.mark.asyncio
async def test_rest_diagnostic_rejects_citation_outside_supplied_research():
    def handler(request):
        return ok_response(request, valid_result("https://outside.invalid/unsupported"))

    result = await run_market_analysis_rest_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    assert result.status_code == 200
    assert result.analysis is None
    assert result.error_type == "ResponseValidationError"
    assert "citation validation" in result.safe_error


@pytest.mark.asyncio
async def test_rest_diagnostic_reports_timeout_without_response_metadata():
    def handler(request):
        raise httpx.ReadTimeout("sensitive synthetic request text", request=request)

    result = await run_market_analysis_rest_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    assert result.status_code is None
    assert result.error_type == "ReadTimeout"
    assert result.analysis is None
    assert result.safe_error == "Gemini REST diagnostic request timed out"


@pytest.mark.asyncio
async def test_rest_diagnostic_sanitizes_provider_error_details():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"status": "INVALID_ARGUMENT", "message": f"bad key={API_KEY}"}},
            request=request,
        )

    result = await run_market_analysis_rest_diagnostic(
        IDEA,
        RESEARCH,
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    assert result.status_code == 400
    assert result.error_type == "GeminiHTTPError"
    assert result.error_code == "INVALID_ARGUMENT"
    assert API_KEY not in (result.safe_error or "")
