import { describe, expect, it, vi } from "vitest";
import {
  analyzeIdea,
  ApiConnectionError,
  ApiHttpError,
  ApiResponseError,
  getApiBaseUrl,
} from "./api";

const validResponse = {
  idea: {},
  submitted_budget: { amount: 50000, currency: "INR" },
  market_research: { market_information: [], competitors_and_alternatives: [], industry_trends: [] },
  market_analysis: {
    demand_assessment: { claim: "Unknown", classification: "unknown", evidence: [] },
    customer_segments: [], opportunities: [], challenges: [],
  },
  competitor_analysis: { competitors: [], market_gaps: [] },
  swot_risk_analysis: { strengths: [], weaknesses: [], opportunities: [], threats: [], execution_risks: [] },
};

describe("API client", () => {
  it("uses the configured backend URL and strips trailing slashes", () => {
    expect(getApiBaseUrl()).toBe("http://127.0.0.1:8010");
    expect(getApiBaseUrl(" http://127.0.0.1:8010/// ")).toBe("http://127.0.0.1:8010");
    expect(getApiBaseUrl("")).toBe("http://127.0.0.1:8010");
  });

  it("posts the analysis payload to the configured endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => validResponse,
    });

    const result = await analyzeIdea({ industry: "Education" }, fetchMock);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8010/api/ideas/analyze",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ industry: "Education" }),
      }),
    );
    expect(result).toEqual(validResponse);
  });

  it("labels fetch rejections as network or CORS failures", async () => {
    await expect(analyzeIdea({}, vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))))
      .rejects.toBeInstanceOf(ApiConnectionError);
  });

  it("preserves HTTP and backend provider errors distinctly", async () => {
    await expect(analyzeIdea({}, vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: "Gemini denied access to market analysis." }),
    }))).rejects.toMatchObject({
      name: ApiHttpError.name,
      status: 503,
      message: expect.stringContaining("Gemini denied access"),
    });
  });

  it("preserves sanitized Groq rate-limit and retry information without retrying", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        detail: "Groq token rate limit reached during competitor analysis (HTTP 429). Provider reported retry-after=2. No automatic retry was made.",
      }),
    });

    await expect(analyzeIdea({}, fetchMock)).rejects.toMatchObject({
      name: ApiHttpError.name,
      status: 503,
      message: expect.stringContaining("retry-after=2"),
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects non-JSON and structurally incomplete successful responses", async () => {
    const nonJson = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => { throw new SyntaxError("invalid JSON"); },
    });
    await expect(analyzeIdea({}, nonJson)).rejects.toBeInstanceOf(ApiResponseError);

    const incomplete = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ market_analysis: {} }),
    });
    await expect(analyzeIdea({}, incomplete)).rejects.toThrow("expected analysis structure");
  });
});
