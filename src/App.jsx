
import { useState } from "react";
import "./App.css";
import { analyzeIdea, ApiConnectionError } from "./api";

function isSafeExternalUrl(value) {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

function DetailList({ title, values }) {
  const items = Array.isArray(values)
    ? values.filter((value) => typeof value === "string" && value.trim())
    : [];
  return (
    <div className="competitor-detail">
      <h4>{title}</h4>
      {items.length > 0 ? (
        <ul>{items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}</ul>
      ) : <p className="muted">Not available in the supplied research.</p>}
    </div>
  );
}

function EvidenceLinks({ urls }) {
  const links = Array.isArray(urls) ? urls.filter(isSafeExternalUrl) : [];
  if (!links.length) return <p className="muted">No source links available.</p>;
  return (
    <ul className="competitor-source-list">
      {links.map((url, index) => (
        <li key={`${url}-${index}`}>
          <a href={url} target="_blank" rel="noopener noreferrer">Source {index + 1}</a>
        </li>
      ))}
    </ul>
  );
}

function MarketClaimView({ item }) {
  if (!item || typeof item.claim !== "string") {
    return <p className="muted">Unknown</p>;
  }
  const label = item.classification === "evidence_based"
    ? "Evidence-based"
    : item.classification === "hypothesis"
      ? "Hypothesis"
      : "Unknown";
  return (
    <div className="market-claim">
      <p><strong>{item.claim}</strong> <span className={`claim-type claim-${item.classification || "unknown"}`}>{label}</span></p>
      <EvidenceLinks urls={item.evidence} />
    </div>
  );
}

function MarketOpportunitySection({ analysis }) {
  if (!analysis) return null;
  const segments = Array.isArray(analysis.customer_segments) ? analysis.customer_segments : [];
  const opportunities = Array.isArray(analysis.opportunities) ? analysis.opportunities : [];
  const challenges = Array.isArray(analysis.challenges) ? analysis.challenges : [];
  return (
    <section className="competitor-analysis market-analysis" aria-labelledby="market-analysis-heading">
      <h3 id="market-analysis-heading">Market opportunity and customer analysis</h3>
      <p className="muted">Findings cite supplied research. Hypotheses are ideas to validate, not verified facts.</p>
      <article className="competitor-card">
        <h4>Market demand</h4>
        <MarketClaimView item={analysis.demand_assessment} />
      </article>
      <h4>Customer segments and needs</h4>
      {segments.length ? segments.map((segment, index) => (
        <article className="competitor-card" key={`${segment.segment?.claim || "segment"}-${index}`}>
          <MarketClaimView item={segment.segment} />
          <h4>Needs</h4>
          {Array.isArray(segment.needs) && segment.needs.length
            ? segment.needs.map((need, needIndex) => <MarketClaimView key={`${need.claim}-${needIndex}`} item={need} />)
            : <p className="muted">Unknown</p>}
        </article>
      )) : <p className="muted">No customer segments were supported by the supplied research.</p>}
      <div className="market-claim-columns">
        <div>
          <h4>Opportunities</h4>
          {opportunities.length ? opportunities.map((item, index) => <MarketClaimView key={`${item.claim}-${index}`} item={item} />) : <p className="muted">Unknown</p>}
        </div>
        <div>
          <h4>Challenges</h4>
          {challenges.length ? challenges.map((item, index) => <MarketClaimView key={`${item.claim}-${index}`} item={item} />) : <p className="muted">Unknown</p>}
        </div>
      </div>
    </section>
  );
}

function SWOTRiskSection({ analysis }) {
  if (!analysis) return null;
  const sections = [
    ["Strengths", analysis.strengths],
    ["Weaknesses", analysis.weaknesses],
    ["Opportunities", analysis.opportunities],
    ["Threats", analysis.threats],
  ];
  const risks = Array.isArray(analysis.execution_risks) ? analysis.execution_risks : [];
  return (
    <section className="competitor-analysis swot-analysis" aria-labelledby="swot-heading">
      <h3 id="swot-heading">SWOT and execution risks</h3>
      <p className="muted">Evidence-based items cite supplied research. Hypotheses are possibilities to validate, not established facts.</p>
      <div className="swot-grid">
        {sections.map(([title, items]) => (
          <article className="competitor-card" key={title}>
            <h4>{title}</h4>
            {Array.isArray(items) && items.length
              ? items.map((item, index) => <MarketClaimView key={`${item.claim}-${index}`} item={item} />)
              : <p className="muted">Unknown</p>}
          </article>
        ))}
      </div>
      <h4>Execution risks and potential impact</h4>
      {risks.length ? risks.map((item, index) => (
        <article className="competitor-card" key={`${item.risk?.claim || "risk"}-${index}`}>
          <h4>Risk</h4>
          <MarketClaimView item={item.risk} />
          <h4>Potential impact</h4>
          <MarketClaimView item={item.potential_impact} />
        </article>
      )) : <p className="muted">No execution risks could be assessed from the supplied context.</p>}
    </section>
  );
}

function CompetitorAnalysisSection({ analysis }) {
  if (!analysis) {
    return (
      <section className="competitor-analysis" aria-labelledby="competitor-heading">
        <h3 id="competitor-heading">Competitor analysis</h3>
        <p className="muted">Competitor analysis is unavailable in this response.</p>
      </section>
    );
  }

  const competitors = Array.isArray(analysis.competitors) ? analysis.competitors : [];
  const gaps = Array.isArray(analysis.market_gaps) ? analysis.market_gaps : [];
  return (
    <section className="competitor-analysis" aria-labelledby="competitor-heading">
      <h3 id="competitor-heading">Competitor analysis</h3>
      <p className="muted">Competitor details are based on the supplied research. Source links identify the material used; claims are not independently fact-checked.</p>
      {competitors.length > 0 ? (
        <div className="competitor-list">
          {competitors.map((competitor, index) => (
            <article className="competitor-card" key={`${competitor.name || "competitor"}-${index}`}>
              <h4>{competitor.name || "Competitor name unavailable"}</h4>
              <p><strong>Classification:</strong> {competitor.competitor_type === "direct" ? "Direct competitor" : competitor.competitor_type === "indirect" ? "Indirect competitor" : "Unknown"}</p>
              <p><strong>Offering:</strong> {competitor.offering || "Unknown"}</p>
              <p><strong>Target customers:</strong> {competitor.target_customers || "Unknown"}</p>
              <p><strong>Pricing:</strong> {competitor.pricing || "Unknown"}</p>
              <DetailList title="Strengths" values={competitor.strengths} />
              <DetailList title="Limitations" values={competitor.limitations} />
              <div className="competitor-detail">
                <h4>Evidence</h4>
                <EvidenceLinks urls={competitor.evidence} />
              </div>
            </article>
          ))}
        </div>
      ) : <p className="muted">No competitors were supported by the available competitor research.</p>}

      <h4>Potential market gaps</h4>
      {gaps.length > 0 ? (
        <div className="competitor-list">
          {gaps.map((gap, index) => (
            <article className="competitor-card" key={`${gap.description || "gap"}-${index}`}>
              <h4>{gap.description || "Description unavailable"}</h4>
              <p><strong>Type:</strong> {gap.gap_type === "evidence_based" ? "Evidence-based" : gap.gap_type === "hypothesis" ? "Hypothesis" : "Unknown"}</p>
              <p>{gap.rationale || "No rationale was supplied."}</p>
              <div className="competitor-detail">
                <h4>Evidence</h4>
                <EvidenceLinks urls={gap.evidence} />
              </div>
            </article>
          ))}
        </div>
      ) : <p className="muted">No potential market gaps were returned from available evidence.</p>}
    </section>
  );
}

function App() {
  const [theme, setTheme] = useState(
    localStorage.getItem("asv-theme") || "dark"
  );

  const [formData, setFormData] = useState({
    idea: "",
    customers: "",
    industry: "",
    problem: "",
    budget: "",
    budgetCurrency: "INR",
    additionalInfo: "",
  });

  const [errors, setErrors] = useState({});
  const [submittedData, setSubmittedData] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiMessage, setApiMessage] = useState("");

  function changeTheme() {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("asv-theme", nextTheme);
  }

  function handleChange(event) {
    const { name, value } = event.target;
    setApiMessage("");

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));

    setErrors((previous) => ({
      ...previous,
      [name]: "",
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (isSubmitting) return;

    const newErrors = {};

    if (formData.idea.trim().length < 20) {
      newErrors.idea = "Please enter at least 20 characters.";
    }

    if (!formData.customers.trim()) {
      newErrors.customers = "Enter your target customers.";
    }

    if (!formData.industry) {
      newErrors.industry = "Please select an industry.";
    }

    if (formData.problem.trim().length < 10) {
      newErrors.problem = "Please describe the problem in 10+ characters.";
    }

    if (
      formData.budget.trim() === "" ||
      !Number.isFinite(Number(formData.budget)) ||
      Number(formData.budget) <= 0
    ) {
      newErrors.budget = "Enter a budget greater than zero.";
    }

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) {
      setSubmittedData(null);
      return;
    }

    const structuredData = {
      startup_name: "AI Startup Validator",
      idea_description: formData.idea.trim(),
      target_customers: formData.customers
        .split(",")
        .map((customer) => customer.trim())
        .filter(Boolean),
      industry: formData.industry,
      problem_statement: formData.problem.trim(),
      estimated_budget: {
        amount: Number(formData.budget),
        currency: formData.budgetCurrency,
      },
      additional_information: formData.additionalInfo.trim() || null,
    };

    setSubmittedData(null);
    setApiMessage("");
    setIsSubmitting(true);
    try {
      const result = await analyzeIdea({
          idea_description: structuredData.idea_description,
          target_customers: formData.customers.trim(),
          industry: structuredData.industry,
          problem_statement: structuredData.problem_statement,
          estimated_budget: structuredData.estimated_budget.amount,
          estimated_budget_currency: structuredData.estimated_budget.currency,
          additional_information: structuredData.additional_information,
      });
      setSubmittedData(result);
      setApiMessage("Idea analyzed successfully.");
    } catch (error) {
      const message = error instanceof ApiConnectionError
        ? error.message
        : error.message || "Unexpected backend error.";
      setApiMessage(`Submission failed: ${message}`);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className={`app ${theme}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">✦</div>
          <div>
            <h2>AI Startup</h2>
            <h2>Validator</h2>
            <p>Validate today. Build tomorrow.</p>
          </div>
        </div>

        <nav className="navigation">
          <a href="#dashboard">⌂ <span>Dashboard</span></a>
          <a className="active" href="#submission">✧ <span>Validate Idea</span></a>
          <a href="#ideas">▤ <span>My Ideas</span></a>
          <a href="#reports">▥ <span>Reports</span></a>
          <a href="#settings">⚙ <span>Settings</span></a>
        </nav>

        <div className="sidebar-card">
          <div className="rocket">🚀</div>
          <h3>Big ideas start here.</h3>
          <p>Turn your ideas into opportunities with structured AI-powered analysis.</p>
        </div>
      </aside>

      <main className="main-content" id="submission">
        <header className="topbar">
          <span className="status-label">
            <span className="status-dot" /> IDEA WORKSPACE
          </span>

          <button
            type="button"
            className="theme-toggle"
            onClick={changeTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            <span>{theme === "dark" ? "☾ Dark" : "☀ Light"}</span>
            <span className="toggle-track">
              <span className="toggle-thumb" />
            </span>
          </button>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">YOUR NEXT BIG THING STARTS HERE</span>
            <h1>
              Turn your idea into a
              <span> business opportunity.</span>
            </h1>
            <p>
              Share your startup idea. Organize your vision and prepare it
              for market research, competitor analysis, and validation.
            </p>

            <div className="hero-pills">
              <span>✦ AI-ready input</span>
              <span>◈ Structured data</span>
              <span>✓ Input validation</span>
            </div>
          </div>

          <div className="hero-art" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="brain">✳</div>
            <div className="art-label label-top">MARKET INSIGHTS</div>
            <div className="art-label label-bottom">IDEA ANALYSIS</div>
            <div className="art-base" />
          </div>
        </section>

        <section className="form-card">
          <div className="section-heading">
            <div className="section-icon">✧</div>
            <div>
              <h2>Submit your startup idea</h2>
              <p>Tell us what you're building. Start with the essentials.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="idea">Startup idea description <b>*</b></label>
                <textarea
                  id="idea"
                  name="idea"
                  rows="4"
                  maxLength={500}
                  placeholder="Describe your startup idea, how it works, and what makes it unique..."
                  value={formData.idea}
                  onChange={handleChange}
                  aria-invalid={Boolean(errors.idea)}
                />
                <div className="field-footer">
                  {errors.idea ? (
                    <span className="error">{errors.idea}</span>
                  ) : (
                    <span>Explain your idea in simple words.</span>
                  )}
                  <span>{formData.idea.length}/500</span>
                </div>
              </div>

              <div className="field">
                <label htmlFor="customers">Target customers <b>*</b></label>
                <input
                  id="customers"
                  name="customers"
                  placeholder="e.g., Students, entrepreneurs, small businesses"
                  value={formData.customers}
                  onChange={handleChange}
                  aria-invalid={Boolean(errors.customers)}
                />
                {errors.customers && <span className="error">{errors.customers}</span>}
                <small>Who will benefit from your idea?</small>
              </div>

              <div className="field">
                <label htmlFor="industry">Industry <b>*</b></label>
                <select
                  id="industry"
                  name="industry"
                  value={formData.industry}
                  onChange={handleChange}
                  aria-invalid={Boolean(errors.industry)}
                >
                  <option value="">Select an industry</option>
                  <option value="Artificial Intelligence">Artificial Intelligence</option>
                  <option value="Education">Education</option>
                  <option value="Healthcare">Healthcare</option>
                  <option value="Agriculture">Agriculture</option>
                  <option value="E-commerce">E-commerce</option>
                  <option value="Finance">Finance</option>
                  <option value="Environment">Environment</option>
                  <option value="Other">Other</option>
                </select>
                {errors.industry && <span className="error">{errors.industry}</span>}
                <small>Choose the closest industry.</small>
              </div>

              <div className="field">
                <label htmlFor="problem">Problem statement <b>*</b></label>
                <input
                  id="problem"
                  name="problem"
                  placeholder="What problem does your idea solve?"
                  value={formData.problem}
                  onChange={handleChange}
                  aria-invalid={Boolean(errors.problem)}
                />
                {errors.problem && <span className="error">{errors.problem}</span>}
                <small>Identify the main customer pain point.</small>
              </div>

              <div className="field">
                <label htmlFor="budget">Estimated budget <b>*</b></label>
                <div className="input-with-symbol">
                  <select
                    id="budgetCurrency"
                    name="budgetCurrency"
                    aria-label="Budget currency"
                    value={formData.budgetCurrency}
                    onChange={handleChange}
                  >
                    <option value="INR">₹ INR</option>
                    <option value="USD">$ USD</option>
                  </select>
                  <input
                    id="budget"
                    name="budget"
                    type="number"
                    min="1"
                    step="any"
                    placeholder="e.g., 50000"
                    value={formData.budget}
                    onChange={handleChange}
                    aria-invalid={Boolean(errors.budget)}
                  />
                </div>
                {errors.budget && <span className="error">{errors.budget}</span>}
                <small>Approximate initial investment.</small>
              </div>

              <div className="field">
                <label htmlFor="additionalInfo">Additional information</label>
                <input
                  id="additionalInfo"
                  name="additionalInfo"
                  placeholder="Competitors, location, special requirements..."
                  value={formData.additionalInfo}
                  onChange={handleChange}
                />
                <small>Optional details that provide more context.</small>
              </div>
            </div>

            <div className="form-actions">
              <span><b>*</b> Required fields</span>
              <button className="submit-button" type="submit" disabled={isSubmitting}>
                <span>✧</span> {isSubmitting ? "Analyzing..." : "Prepare My Idea"} <span>→</span>
              </button>
            </div>
            {apiMessage && <p className={apiMessage.startsWith("Submission failed:") ? "error" : "success-message"} role="status">{apiMessage}</p>}
          </form>
        </section>

        <section className="insight-grid">
          <article className="insight-card">
            <div className="insight-icon purple">▥</div>
            <div><h3>Market demand</h3><p>Prepare for market research.</p></div>
          </article>
          <article className="insight-card">
            <div className="insight-icon blue">♧</div>
            <div><h3>Target customers</h3><p>Define your ideal audience.</p></div>
          </article>
          <article className="insight-card">
            <div className="insight-icon pink">◎</div>
            <div><h3>Competitor analysis</h3><p>Identify competing solutions.</p></div>
          </article>
          <article className="insight-card">
            <div className="insight-icon cyan">▤</div>
            <div><h3>Validation report</h3><p>Organize inputs for analysis.</p></div>
          </article>
        </section>

        {submittedData && (
          <section className="output-card" aria-live="polite">
            <div className="section-heading">
              <div className="section-icon">{"{ }"}</div>
              <div>
                <h2>Structured idea output</h2>
                <p>Backend analysis response, formatted as JSON.</p>
              </div>
            </div>
            <pre>{JSON.stringify(submittedData, null, 2)}</pre>
            <MarketOpportunitySection analysis={submittedData.market_analysis} />
            <CompetitorAnalysisSection analysis={submittedData.competitor_analysis} />
            <SWOTRiskSection analysis={submittedData.swot_risk_analysis} />
            <button
              type="button"
              className="copy-button"
              onClick={() => {
                const json = JSON.stringify(submittedData, null, 2);
                if (navigator.clipboard?.writeText) {
                  navigator.clipboard.writeText(json).catch(() => {
                    window.prompt("Copy your JSON:", json);
                  });
                } else {
                  window.prompt("Copy your JSON:", json);
                }
              }}
            >
              Copy JSON
            </button>
            <p className="output-note">
              Research snippets are provider supplied and not independently verified. Review each source URL before relying on a claim or figure.
            </p>
          </section>
        )}

        <footer className="footer">
          <span>AI Startup Validator</span>
          <span>Idea submission workspace · Milestone 1</span>
        </footer>
      </main>
    </div>
  );
}

export default App;

