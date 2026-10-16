import os
import logging
import httpx
from app.schemas import Source, StructuredIdea
from app.services.idea_extraction import ServiceError

logger = logging.getLogger(__name__)

class MarketResearchAgent:
    def __init__(self, transport=None):
        self.transport = transport

    async def research(self, idea: StructuredIdea) -> dict[str, list[Source]]:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ServiceError("TAVILY_API_KEY is not configured.", 503)
        subject = f"{idea.industry} {idea.proposed_solution} for {' '.join(idea.target_customers)}"
        queries = {
            "market_information": f"{subject} market size demand trends",
            "competitors_and_alternatives": f"{subject} competitors existing alternatives",
            "industry_trends": f"{idea.industry} industry technology trends {idea.problem}",
        }
        out = {}
        try:
            trust_env = os.getenv("HTTPX_TRUST_ENV", "true").strip().lower() not in {"0", "false", "no"}
            async with httpx.AsyncClient(timeout=20.0, transport=self.transport, trust_env=trust_env) as client:
                for category, query in queries.items():
                    response = await client.post("https://api.tavily.com/search",
                        json={"api_key": api_key, "query": query, "max_results": 5, "search_depth": "basic"})
                    if response.status_code == 429:
                        raise ServiceError("Tavily rate limit reached; try again later.", 503)
                    if response.status_code >= 400:
                        raise ServiceError(f"Tavily search returned HTTP {response.status_code}.")
                    results = response.json().get("results", [])
                    sources = []
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        title, url, content = result.get("title"), result.get("url"), result.get("content")
                        if not isinstance(title, str) or not title.strip():
                            continue
                        if not isinstance(url, str) or not url.strip():
                            continue
                        if not isinstance(content, str):
                            continue
                        sources.append(Source(title=title, url=url, content=content))
                    out[category] = sources
            if not any(out.values()):
                raise ServiceError("Tavily returned no market research results.", 502)
            return out
        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Tavily request timeout: %s", type(exc).__name__)
            raise ServiceError(f"Tavily request timed out ({type(exc).__name__}).", 504) from exc
        except httpx.ConnectError as exc:
            logger.warning("Tavily connection failed: %s", type(exc).__name__)
            raise ServiceError("Could not connect to Tavily; check network or proxy settings.", 502) from exc
        except httpx.RequestError as exc:
            logger.warning("Tavily request failed: %s", type(exc).__name__)
            raise ServiceError(f"Tavily request failed ({type(exc).__name__}).", 502) from exc
        except (ValueError, TypeError):
            raise ServiceError("Tavily returned an invalid response.", 502)
