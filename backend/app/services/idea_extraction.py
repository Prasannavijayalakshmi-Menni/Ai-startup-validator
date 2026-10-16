import json
import logging
import os
import re
import httpx
from pydantic import ValidationError
from app.schemas import StartupIdea, StructuredIdea

logger = logging.getLogger(__name__)


def _trust_env() -> bool:
    return os.getenv("HTTPX_TRUST_ENV", "true").strip().lower() not in {"0", "false", "no"}


def _provider_error(response: httpx.Response) -> tuple[str, str, str]:
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        error = {}
    if not isinstance(error, dict):
        error = {}
    code = re.sub(r"[^A-Za-z0-9_.-]", "", str(error.get("code") or ""))[:60].lower()
    error_type = re.sub(r"[^A-Za-z0-9_.-]", "", str(error.get("type") or ""))[:60].lower()
    message = str(error.get("message") or "").strip().lower()
    return code, error_type, message

class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        self.message, self.status_code = message, status_code
        super().__init__(message)

class IdeaExtractionAgent:
    def __init__(self, transport=None):
        self.transport = transport

    async def extract(self, idea: StartupIdea) -> StructuredIdea:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ServiceError("LLM_API_KEY is not configured.", 503)
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        prompt = {
            "role": "user",
            "content": ("Extract the startup idea into a JSON object with exactly these fields: "
                "problem (string or null), proposed_solution (string or null), "
                "target_customers (array of strings, possibly empty), industry (string or null), "
                "key_features (array of strings, possibly empty), value_proposition (string or null), "
                "assumptions_and_missing_information (array of strings). Use only facts explicitly "
                "supported by the submitted fields; do not add outside knowledge, market statistics, "
                "competitors, dates, or claims. If a field is absent or unsupported, use null or an "
                "empty array and name that exact field in assumptions_and_missing_information. List "
                "any inference that still needs user confirmation there as well. Return only the JSON "
                "object: no Markdown fences, preamble, or explanatory text. Do not transform or "
                "restate the submitted budget; the backend returns it separately with its original "
                "currency. Budget is not part of the extraction task and must not be included in "
                "assumptions or other extracted fields. Input: " + idea.model_dump_json(
                    exclude={"estimated_budget", "estimated_budget_currency"}
                ))
        }
        try:
            async with httpx.AsyncClient(timeout=30.0, transport=self.transport, trust_env=_trust_env()) as client:
                response = await client.post(f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": "Return exactly one valid JSON object matching the requested fields. Do not use Markdown fences or add explanatory text."}, prompt]})
            code, error_type, provider_message = _provider_error(response)
            quota_codes = {
                "blocked_api_access", "billing_hard_limit_reached", "billing_not_active",
                "credit_balance_exhausted", "insufficient_quota", "quota_exceeded",
            }
            if response.status_code == 402 or (response.status_code in {400, 403, 429} and code in quota_codes):
                logger.warning("LLM provider account quota or billing limit: HTTP %s code=%s", response.status_code, code)
                raise ServiceError("The Groq account quota or billing limit has been reached. Check account usage and limits; no automatic retry was made.", 503)
            if response.status_code == 401:
                logger.warning("LLM provider authentication failed: HTTP 401")
                raise ServiceError("Groq rejected the API credential (HTTP 401). Check the configured API key.", 503)
            if response.status_code == 403:
                logger.warning("LLM provider permission denied: HTTP 403")
                raise ServiceError("The Groq credential does not have permission to use this resource (HTTP 403).", 503)
            if response.status_code in {400, 404} and (
                code in {"invalid_model", "model_not_found", "model_decommissioned", "unknown_model"}
                or ("model" in provider_message and any(term in provider_message for term in ("not found", "does not exist", "unavailable", "decommissioned")))
            ):
                logger.warning("LLM provider rejected configured model: HTTP %s", response.status_code)
                raise ServiceError("The configured Groq model is invalid or unavailable. Check LLM_MODEL.", 503)
            if response.status_code == 429:
                logger.warning("LLM provider rate limit: HTTP 429 code=%s type=%s", code or "unknown", error_type or "unknown")
                raise ServiceError("Groq rate limit reached. Wait for the account or model limit to reset before retrying.", 503)
            if response.status_code >= 400:
                logger.warning("LLM provider HTTP error: status=%s code=%s", response.status_code, code or "unknown")
                if response.status_code == 404:
                    raise ServiceError("The configured Groq endpoint was not found. Check LLM_BASE_URL.", 502)
                raise ServiceError(f"Groq returned HTTP {response.status_code}; check the provider request configuration.")
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise ServiceError("Groq returned an incomplete response; no analysis was produced.")
            if not isinstance(content, str):
                raise ServiceError("Groq returned an incomplete response; no analysis was produced.")
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ServiceError("Groq returned invalid JSON; no analysis was produced.") from exc
            if not isinstance(payload, dict):
                raise ServiceError("Groq returned JSON that does not match the required idea fields; no analysis was produced.")
            try:
                return StructuredIdea.model_validate(payload)
            except ValidationError as exc:
                raise ServiceError("Groq returned JSON that does not match the required idea fields; no analysis was produced.") from exc
        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("LLM provider timeout: %s", type(exc).__name__)
            raise ServiceError(f"LLM provider request timed out ({type(exc).__name__}).", 504) from exc
        except httpx.ConnectError as exc:
            logger.warning("LLM provider connection failed: %s", type(exc).__name__)
            raise ServiceError("Could not connect to the LLM provider; check network or proxy settings.", 502) from exc
        except httpx.RequestError as exc:
            logger.warning("LLM provider request failed: %s", type(exc).__name__)
            raise ServiceError(f"LLM provider request failed ({type(exc).__name__}).", 502) from exc
        except (KeyError, IndexError, ValueError, ValidationError, TypeError):
            raise ServiceError("Groq returned an incomplete response; no analysis was produced.")
