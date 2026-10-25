"""Regression tests for IdeaExtractionAgent using the Gemini SDK mock pattern."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from app.schemas import StartupIdea, StructuredIdea
from app.services.idea_extraction import IdeaExtractionAgent, ServiceError
from app.services.market_analysis import GEMINI_MODEL


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
def configured_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
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
    clients = []

    def factory(api_key):
        captured["api_key_received"] = api_key
        c = FakeGeminiClient(response=response, error=error, captured=captured)
        clients.append(c)
        return c

    return IdeaExtractionAgent(client_factory=factory), captured, clients


@pytest.mark.asyncio
async def test_extraction_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ServiceError, match="GEMINI_API_KEY"):
        await IdeaExtractionAgent().extract(StartupIdea(**IDEA_INPUT))


@pytest.mark.asyncio
async def test_extraction_accepts_valid_structured_json_and_excludes_budget_from_prompt():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(VALID_RESPONSE), captured=captured)

    result = await agent.extract(StartupIdea(**IDEA_INPUT))

    assert result.industry == "Education"
    assert result.target_customers == ["Students"]
    assert captured["api_key_received"] == "mock-gemini-key"

    req = captured["request"]
    assert req["model"] == GEMINI_MODEL
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["config"]["system_instruction"]
    prompt = req["contents"]
    assert "50000" not in prompt
    assert "estimated_budget" not in prompt
    assert "estimated_budget_currency" not in prompt
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_extraction_accepts_markdown_fenced_json():
    content = "```json\n" + json.dumps(VALID_RESPONSE) + "\n```"
    agent, _, _ = make_agent(content)
    result = await agent.extract(StartupIdea(**IDEA_INPUT))
    assert result.proposed_solution == "Adaptive study planner"


@pytest.mark.asyncio
async def test_extraction_reports_invalid_json():
    agent, _, _ = make_agent("not json")
    with pytest.raises(ServiceError, match="invalid JSON"):
        await agent.extract(StartupIdea(**IDEA_INPUT))


@pytest.mark.asyncio
async def test_extraction_empty_response_is_rejected():
    agent, _, _ = make_agent("")
    with pytest.raises(ServiceError, match="empty response"):
        await agent.extract(StartupIdea(**IDEA_INPUT))


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
    agent, _, _ = make_agent(json.dumps(payload))
    result = await agent.extract(StartupIdea(**IDEA_INPUT))
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
    agent, _, _ = make_agent(json.dumps(payload))
    with pytest.raises(ServiceError, match="required idea fields"):
        await agent.extract(StartupIdea(**IDEA_INPUT))


@pytest.mark.asyncio
async def test_extraction_timeout_returns_504():
    agent, _, _ = make_agent(error=asyncio.TimeoutError())
    with pytest.raises(ServiceError, match="timed out") as exc:
        await agent.extract(StartupIdea(**IDEA_INPUT))
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_extraction_httpx_timeout_returns_504():
    agent, _, _ = make_agent(error=httpx.ReadTimeout("timeout"))
    with pytest.raises(ServiceError, match="timed out") as exc:
        await agent.extract(StartupIdea(**IDEA_INPUT))
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_extraction_connection_failure_returns_502():
    agent, _, _ = make_agent(error=httpx.ConnectError("offline"))
    with pytest.raises(ServiceError, match="Could not connect") as exc:
        await agent.extract(StartupIdea(**IDEA_INPUT))
    assert exc.value.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected,status_code",
    [
        (401, "configured API credential", 503),
        (403, "denied access", 503),
        (429, "rate or quota limit", 503),
        (400, "HTTP 400", 502),
        (500, "provider request failed", 502),
    ],
)
async def test_extraction_provider_errors_are_distinguished(status, expected, status_code, caplog):
    error = RuntimeError("sensitive provider message mock-gemini-key")
    error.code = status
    error.message = "sensitive provider message mock-gemini-key"
    agent, _, _ = make_agent(error=error)
    caplog.set_level(logging.WARNING, logger="app.services.idea_extraction")

    with pytest.raises(ServiceError, match=expected) as raised:
        await agent.extract(StartupIdea(**IDEA_INPUT))

    assert raised.value.status_code == status_code
    assert "mock-gemini-key" not in caplog.text


@pytest.mark.asyncio
async def test_extraction_aclose_called_after_success():
    captured = {}
    agent, captured, clients = make_agent(json.dumps(VALID_RESPONSE), captured=captured)
    await agent.extract(StartupIdea(**IDEA_INPUT))
    clients[0].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_extraction_aclose_called_after_error():
    error = RuntimeError("provider failure")
    error.code = 500
    error.message = "provider failure"
    captured = {}
    agent, captured, clients = make_agent(error=error, captured=captured)
    with pytest.raises(ServiceError):
        await agent.extract(StartupIdea(**IDEA_INPUT))
    clients[0].aio.aclose.assert_awaited_once()


def test_structured_idea_requires_disclosure_of_missing_values():
    with pytest.raises(ValidationError, match="must be named"):
        StructuredIdea.model_validate(
            {
                "problem": None,
                **{key: value for key, value in VALID_RESPONSE.items() if key != "problem"},
                "assumptions_and_missing_information": [],
            }
        )
