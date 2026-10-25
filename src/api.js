const DEFAULT_API_BASE_URL = "http://127.0.0.1:8010";

export function getApiBaseUrl(configuredUrl = import.meta.env.VITE_API_BASE_URL) {
  return (configuredUrl?.trim() || DEFAULT_API_BASE_URL).replace(/\/+$/, "");
}

export class ApiConnectionError extends Error {
  constructor(message) {
    super(message);
    this.name = "ApiConnectionError";
  }
}

export class ApiHttpError extends Error {
  constructor(status, detail) {
    super(`Backend returned HTTP ${status}: ${detail}`);
    this.name = "ApiHttpError";
    this.status = status;
  }
}

export class ApiResponseError extends Error {
  constructor(message) {
    super(message);
    this.name = "ApiResponseError";
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isAnalysisResponse(value) {
  if (!isRecord(value) || !isRecord(value.idea)) return false;
  if (!isRecord(value.submitted_budget)
    || !Number.isFinite(value.submitted_budget.amount)
    || typeof value.submitted_budget.currency !== "string") return false;
  if (!isRecord(value.market_research)
    || !Object.values(value.market_research).every(Array.isArray)) return false;

  const market = value.market_analysis;
  if (!isRecord(market)
    || !isRecord(market.demand_assessment)
    || !Array.isArray(market.customer_segments)
    || !Array.isArray(market.opportunities)
    || !Array.isArray(market.challenges)) return false;

  const competitors = value.competitor_analysis;
  if (!isRecord(competitors)
    || !Array.isArray(competitors.competitors)
    || !Array.isArray(competitors.market_gaps)) return false;

  const swot = value.swot_risk_analysis;
  return isRecord(swot)
    && ["strengths", "weaknesses", "opportunities", "threats", "execution_risks"]
      .every((field) => Array.isArray(swot[field]));
}

function getErrorDetail(payload, status) {
  if (Array.isArray(payload?.detail)) {
    return payload.detail
      .map((item) => `${item.loc?.slice(-1)[0] || "field"}: ${item.msg}`)
      .join("; ");
  }
  return payload?.detail || payload?.message || `Request failed (HTTP ${status}).`;
}

export async function analyzeIdea(payload, fetchImpl = fetch) {
  let response;
  try {
    response = await fetchImpl(`${getApiBaseUrl()}/api/ideas/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new ApiConnectionError(
        `Could not reach the backend at ${getApiBaseUrl()}. Check that it is running and that CORS allows this page's origin (network/CORS failure).`,
      );
    }
    throw error;
  }

  if (!response.ok) {
    let errorPayload = {};
    try {
      errorPayload = await response.json();
    } catch {
      // Keep the HTTP status visible without reflecting an arbitrary response body.
    }
    throw new ApiHttpError(response.status, getErrorDetail(errorPayload, response.status));
  }

  let result;
  try {
    result = await response.json();
  } catch {
    throw new ApiResponseError("The backend returned invalid JSON; analysis results were not displayed.");
  }
  if (!isAnalysisResponse(result)) {
    throw new ApiResponseError(
      "The backend response did not match the expected analysis structure; analysis results were not displayed.",
    );
  }
  return result;
}
