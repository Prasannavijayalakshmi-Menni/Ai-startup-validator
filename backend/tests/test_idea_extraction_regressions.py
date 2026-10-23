import json

import httpx
import pytest
from pydantic import ValidationError

from app.schemas import StartupIdea, StructuredIdea
from app.services.idea_extraction import IdeaExtractionAgent, ServiceError


IDEA_INPUT = {
    "idea_description": "An AI tutor that creates adaptive study plans for students.",
    "target_customers": "Students",
    "industry": "Education",
    "problem_statement": "Students struggle to create consistent study plans.",
    "estimated_budget": 50000,
    "estimated_budget_currency": "INR",
}
VALID_RESPONSE = {
    "problem": "Students lack study plans",
    "proposed_solution": "Adaptive study planner",
    "target_customers": ["Students"],
    "industry": "Education",
    "key_features": ["Adaptive schedule"],
    "value_proposition": "Consistent study habits",
    "assumptions_and_missing_information": [],
}


@pytest.fixture(autouse=True)
def mock_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "mock-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")


def mock_completion(content, status=200):
    return httpx.MockTransport(
        lambda request: httpx.Response(
            status,
            json={"choices": [{"message": {"content": content}}]}
            if status < 400
            else {"error": {"code": "test_error", "message": "mock provider error"}},
            request=request,
        )
    )


@pytest.mark.asyncio
async def test_extraction_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(ServiceError, match="LLM_API_KEY"):
        await IdeaExtractionAgent().extract(StartupIdea(**IDEA_INPUT))


@pytest.mark.asyncio
async def test_extraction_accepts_valid_structured_json_and_excludes_budget_from_prompt():
    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][1]["content"]
        assert body["response_format"] == {"type": "json_object"}
        assert "50000" not in prompt
        assert "estimated_budget" not in prompt
        assert "estimated_budget_currency" not in prompt
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(VALID_RESPONSE)}}]},
            request=request,
        )

    result = await IdeaExtractionAgent(transport=httpx.MockTransport(handler)).extract(
        StartupIdea(**IDEA_INPUT)
    )
    assert result.industry == "Education"
    assert result.target_customers == ["Students"]


@pytest.mark.asyncio
async def test_extraction_accepts_markdown_fenced_json():
    content = "```json\n" + json.dumps(VALID_RESPONSE) + "\n```"
    result = await IdeaExtractionAgent(transport=mock_completion(content)).extract(
        StartupIdea(**IDEA_INPUT)
    )
    assert result.proposed_solution == "Adaptive study planner"


@pytest.mark.asyncio
async def test_extraction_reports_invalid_json():
    with pytest.raises(ServiceError, match="invalid JSON"):
        await IdeaExtractionAgent(transport=mock_completion("not json")).extract(
            StartupIdea(**IDEA_INPUT)
        )


@pytest.mark.asyncio
async def test_extraction_allows_unknown_optional_values_when_disclosed():
    payload = {
        **VALID_RESPONSE,
        "proposed_solution": None,
        "key_features": [],
        "value_proposition": None,
        "assumptions_and_missing_information": [
            "proposed_solution not specified",
            "key_features not specified",
            "value_proposition not specified",
        ],
    }
    result = await IdeaExtractionAgent(
        transport=mock_completion(json.dumps(payload))
    ).extract(StartupIdea(**IDEA_INPUT))
    assert result.proposed_solution is None
    assert result.key_features == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_RESPONSE, "target_customers": "Students"},
        {key: value for key, value in VALID_RESPONSE.items() if key != "industry"},
    ],
)
async def test_extraction_rejects_wrong_type_and_missing_required_key(payload):
    with pytest.raises(ServiceError, match="required idea fields"):
        await IdeaExtractionAgent(
            transport=mock_completion(json.dumps(payload))
        ).extract(StartupIdea(**IDEA_INPUT))


@pytest.mark.asyncio
async def test_extraction_distinguishes_quota_from_rate_limit(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            429,
            json={"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}},
            request=request,
        )
    )
    with pytest.raises(ServiceError, match="billing limit"):
        await IdeaExtractionAgent(transport=transport).extract(StartupIdea(**IDEA_INPUT))


@pytest.mark.parametrize(
    "status,payload,expected",
    [
        (401, {"error": {"code": "invalid_api_key"}}, "rejected the API credential"),
        (400, {"error": {"code": "model_not_found", "message": "model not found"}}, "model is invalid or unavailable"),
        (429, {"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error"}}, "rate limit reached"),
    ],
)
@pytest.mark.asyncio
async def test_extraction_provider_errors_are_distinguished(status, payload, expected):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json=payload, request=request)
    )
    with pytest.raises(ServiceError, match=expected):
        await IdeaExtractionAgent(transport=transport).extract(StartupIdea(**IDEA_INPUT))


def test_structured_idea_requires_disclosure_of_missing_values():
    with pytest.raises(ValidationError, match="must be named"):
        StructuredIdea.model_validate(
            {
                "problem": None,
                **{key: value for key, value in VALID_RESPONSE.items() if key != "problem"},
                "assumptions_and_missing_information": [],
            }
        )
