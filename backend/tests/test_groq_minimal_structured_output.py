"""Isolated mock for the smallest Groq strict structured-output request."""

import json

import httpx
import pytest


MINIMAL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "diagnostic_status",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_parameters",
    [{}, {"temperature": 0}, {"include_reasoning": False}],
    ids=["baseline", "temperature-zero", "reasoning-excluded"],
)
async def test_minimal_strict_schema_request_is_mocked_and_validates_response(
    extra_parameters,
):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["authorization"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"status":"ok"}'}}]},
            request=request,
        )

    request_body = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "Return a JSON object with status set to ok."}
        ],
        "response_format": MINIMAL_RESPONSE_FORMAT,
    }
    request_body.update(extra_parameters)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer offline-test-placeholder"},
            json=request_body,
        )

    assert response.status_code == 200
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-oss-20b"
    assert captured["body"] == request_body
    assert captured["body"]["response_format"] == MINIMAL_RESPONSE_FORMAT
    assert captured["authorization"] == "Bearer offline-test-placeholder"
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {
        "status": "ok"
    }
