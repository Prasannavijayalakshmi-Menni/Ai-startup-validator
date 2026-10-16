
import { useState } from "react";
import "./App.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8001").replace(/\/$/, "");

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
      const response = await fetch(`${API_BASE_URL}/api/ideas/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          idea_description: structuredData.idea_description,
          target_customers: formData.customers.trim(),
          industry: structuredData.industry,
          problem_statement: structuredData.problem_statement,
          estimated_budget: structuredData.estimated_budget.amount,
          estimated_budget_currency: structuredData.estimated_budget.currency,
          additional_information: structuredData.additional_information,
        }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = Array.isArray(result.detail)
          ? result.detail.map((item) => `${item.loc?.slice(-1)[0] || "field"}: ${item.msg}`).join("; ")
          : result.detail || result.message || `Request failed (HTTP ${response.status}).`;
        throw new Error(detail);
      }
      setSubmittedData(result);
      setApiMessage("Idea analyzed successfully.");
    } catch (error) {
      const message = error instanceof TypeError
        ? "Could not connect to the backend. Confirm it is running and VITE_API_BASE_URL is correct."
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
                <span>✧</span> {isSubmitting ? "Analyzing?" : "Prepare My Idea"} <span>→</span>
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

