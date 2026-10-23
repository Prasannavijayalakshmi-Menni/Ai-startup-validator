import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

const sourceUrl = "https://research.example/mock-source";
const validResponse = {
  idea: {
    problem: "Students struggle to plan study time",
    proposed_solution: "An adaptive study planner",
    target_customers: ["Students"],
    industry: "Education",
    key_features: ["Adaptive schedule"],
    value_proposition: "Consistent study habits",
    assumptions_and_missing_information: [],
  },
  submitted_budget: { amount: 50000, currency: "INR" },
  market_research: {
    market_information: [
      { title: "Synthetic test source", url: sourceUrl, content: "Mock snippet" },
    ],
    competitors_and_alternatives: [],
    industry_trends: [],
  },
  market_analysis: {
    demand_assessment: {
      claim: "Study planning is a test-data need.",
      classification: "evidence_based",
      evidence: [sourceUrl],
    },
    customer_segments: [],
    opportunities: [],
    challenges: [],
  },
  competitor_analysis: {
    competitors: [
      {
        name: "Synthetic Study Tool",
        competitor_type: "direct",
        offering: "Study planning tools",
        target_customers: "Students",
        pricing: "Unknown",
        strengths: ["Mock strength"],
        limitations: ["Mock limitation"],
        evidence: [sourceUrl],
      },
    ],
    market_gaps: [
      {
        description: "A synthetic test gap",
        rationale: "This is fixture data only.",
        gap_type: "hypothesis",
        evidence: [],
      },
    ],
  },
  swot_risk_analysis: {
    strengths: [],
    weaknesses: [],
    opportunities: [],
    threats: [],
    execution_risks: [],
  },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function completeForm(user) {
  await user.type(
    screen.getByLabelText(/Startup idea description/i),
    "An adaptive study planner for students who need flexible study support.",
  );
  await user.type(screen.getByLabelText(/Target customers/i), "Students, parents");
  await user.selectOptions(screen.getByLabelText(/Industry/i), "Education");
  await user.type(screen.getByLabelText(/Problem statement/i), "Students need study plans.");
  await user.type(screen.getByLabelText(/Estimated budget/i), "50000");
}

describe("AI Startup Validator frontend", () => {
  it("sends the analysis request and renders mocked market, competitor, evidence, and SWOT results", async () => {
    const user = userEvent.setup();
    let resolveRequest;
    const request = new Promise((resolve) => {
      resolveRequest = resolve;
    });
    const fetchMock = vi.fn().mockReturnValue(request);
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await completeForm(user);
    await user.click(screen.getByRole("button", { name: /Prepare My Idea/i }));
    expect(screen.getByRole("button", { name: /Analyzing\.\.\./i })).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/ideas\/analyze$/),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const sent = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(sent).toMatchObject({
      target_customers: "Students, parents",
      industry: "Education",
      estimated_budget: 50000,
      estimated_budget_currency: "INR",
    });

    resolveRequest({
      ok: true,
      status: 200,
      json: async () => validResponse,
    });

    expect(await screen.findByText("Market opportunity and customer analysis")).toBeTruthy();
    expect(screen.getByText("Synthetic Study Tool")).toBeTruthy();
    expect(screen.getByText(/Direct competitor/)).toBeTruthy();
    expect(screen.getByText("A synthetic test gap")).toBeTruthy();
    expect(screen.getByText("SWOT and execution risks")).toBeTruthy();
    const evidenceLinks = screen.getAllByRole("link", { name: /Source/ });
    expect(evidenceLinks.length).toBeGreaterThan(0);
    expect(evidenceLinks[0].getAttribute("href")).toBe(sourceUrl);
    expect(evidenceLinks[0].getAttribute("rel")).toContain("noopener");
    expect(document.body.textContent).not.toContain("Groq");
  });

  it("shows the backend provider error without displaying an obsolete Groq message", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: async () => ({ detail: "Gemini rejected the market analysis request (HTTP 400)." }),
      }),
    );
    render(<App />);

    await completeForm(user);
    await user.click(screen.getByRole("button", { name: /Prepare My Idea/i }));

    const status = await screen.findByRole("status");
    expect(status.textContent).toContain("Gemini rejected the market analysis request");
    expect(status.textContent).not.toContain("Groq");
  });

  it("shows form validation errors without making an API request", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Prepare My Idea/i }));

    expect(screen.getByText("Please enter at least 20 characters.")).toBeTruthy();
    expect(screen.getByText("Enter your target customers.")).toBeTruthy();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
