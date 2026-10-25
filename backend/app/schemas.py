from typing import Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationInfo,
    field_validator,
    model_validator,
)

class StartupIdea(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    idea_description: str = Field(min_length=20, max_length=500)
    target_customers: str = Field(min_length=1, max_length=500)
    industry: str = Field(min_length=1, max_length=100)
    problem_statement: str = Field(min_length=10, max_length=1000)
    estimated_budget: float = Field(gt=0)
    estimated_budget_currency: str = Field(default="INR", pattern=r"^[A-Za-z]{3}$")
    additional_information: str | None = Field(default=None, max_length=2000)

class StructuredIdea(BaseModel):
    problem: str | None
    proposed_solution: str | None
    target_customers: list[str]
    industry: str | None
    key_features: list[str]
    value_proposition: str | None
    assumptions_and_missing_information: list[str]

    @model_validator(mode="after")
    def missing_fields_are_disclosed(self):
        missing = [
            name for name, value in (
                ("problem", self.problem),
                ("proposed_solution", self.proposed_solution),
                ("target_customers", self.target_customers),
                ("industry", self.industry),
                ("key_features", self.key_features),
                ("value_proposition", self.value_proposition),
            ) if value is None or value == []
        ]
        notes = " ".join(self.assumptions_and_missing_information).lower().replace("_", " ")
        undisclosed = [name for name in missing if name.replace("_", " ") not in notes]
        if undisclosed:
            raise ValueError("Missing extracted fields must be named in assumptions_and_missing_information: " + ", ".join(undisclosed))
        return self

class Source(BaseModel):
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    content: str

class SubmittedBudget(BaseModel):
    amount: float = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Za-z]{3}$")


class Competitor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    competitor_type: Literal["direct", "indirect", "unknown"] = "unknown"
    offering: str = "Unknown"
    target_customers: str = "Unknown"
    pricing: str = "Unknown"
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(min_length=1)

    @field_validator("offering", "target_customers", "pricing", mode="before")
    @classmethod
    def normalize_unavailable_text(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return "Unknown"
        return value


class MarketGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    gap_type: Literal["evidence_based", "hypothesis"] = "hypothesis"
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_based_gap_has_sources(self):
        if self.gap_type == "evidence_based" and not self.evidence:
            raise ValueError("Evidence-based market gaps must cite at least one source URL")
        return self


class CompetitorAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competitors: list[Competitor] = Field(default_factory=list)
    market_gaps: list[MarketGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def citations_match_research(self, info: ValidationInfo):
        context = info.context or {}
        if "allowed_urls" not in context:
            return self
        allowed_urls = set(context["allowed_urls"])
        citations = {
            url
            for item in [*self.competitors, *self.market_gaps]
            for url in item.evidence
        }
        if not citations.issubset(allowed_urls):
            raise ValueError("Competitor analysis contains a URL absent from supplied research")
        return self


class MarketClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    classification: Literal["evidence_based", "hypothesis", "unknown"]
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_matches_classification(self):
        if self.classification == "evidence_based" and not self.evidence:
            raise ValueError("Evidence-based market claims must cite at least one source URL")
        if self.classification == "unknown" and self.claim.strip().lower() != "unknown":
            raise ValueError("Unknown market claims must use the claim 'Unknown'")
        return self


class CustomerSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment: MarketClaim
    needs: list[MarketClaim] = Field(default_factory=list)


class MarketOpportunityAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    demand_assessment: MarketClaim
    industry_trends: list[MarketClaim] = Field(default_factory=list)
    customer_segments: list[CustomerSegment] = Field(default_factory=list)
    opportunities: list[MarketClaim] = Field(default_factory=list)
    challenges: list[MarketClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def citations_match_research(self, info: ValidationInfo):
        context = info.context or {}
        if "allowed_urls" not in context:
            return self
        allowed_urls = set(context["allowed_urls"])
        claims = [
         self.demand_assessment,
        *self.industry_trends,
        *self.opportunities,
        *self.challenges,
        ]
        for segment in self.customer_segments:
            claims.extend([segment.segment, *segment.needs])
        citations = {url for claim in claims for url in claim.evidence}
        if not citations.issubset(allowed_urls):
            raise ValueError("Market analysis contains a URL absent from supplied research")
        return self


class ExecutionRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: MarketClaim
    potential_impact: MarketClaim


class SWOTRiskAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strengths: list[MarketClaim] = Field(default_factory=list)
    weaknesses: list[MarketClaim] = Field(default_factory=list)
    opportunities: list[MarketClaim] = Field(default_factory=list)
    threats: list[MarketClaim] = Field(default_factory=list)
    execution_risks: list[ExecutionRisk] = Field(default_factory=list)

    @model_validator(mode="after")
    def citations_match_research(self, info: ValidationInfo):
        context = info.context or {}
        if "allowed_urls" not in context:
            return self
        allowed_urls = set(context["allowed_urls"])
        claims = [
            *self.strengths,
            *self.weaknesses,
            *self.opportunities,
            *self.threats,
        ]
        for risk in self.execution_risks:
            claims.extend([risk.risk, risk.potential_impact])
        citations = {url for claim in claims for url in claim.evidence}
        if not citations.issubset(allowed_urls):
            raise ValueError("SWOT or risk analysis contains a URL absent from supplied research")
        return self

class AnalysisResponse(BaseModel):
    idea: StructuredIdea
    submitted_budget: SubmittedBudget
    market_research: dict[str, list[Source]]
    market_analysis: MarketOpportunityAnalysis = Field(
        default_factory=lambda: MarketOpportunityAnalysis(
            demand_assessment={"claim": "Unknown", "classification": "unknown"}
        )
    )
    competitor_analysis: CompetitorAnalysis = Field(default_factory=CompetitorAnalysis)
    swot_risk_analysis: SWOTRiskAnalysis = Field(default_factory=SWOTRiskAnalysis)
    research_note: str = (
        "Search snippets are provider-supplied excerpts, not independently "
        "verified facts. Review each linked source before relying on claims or figures."
    )

    @model_validator(mode="after")
    def citations_exist_in_returned_research(self):
        available = {
            source.url
            for category_sources in self.market_research.values()
            for source in category_sources
        }
        citations = {
            url
            for item in [
                *self.competitor_analysis.competitors,
                *self.competitor_analysis.market_gaps,
            ]
            for url in item.evidence
        }
        market_claims = [
       self.market_analysis.demand_assessment,
        *self.market_analysis.industry_trends,
        *self.market_analysis.opportunities,
         *self.market_analysis.challenges,

        ]
        for segment in self.market_analysis.customer_segments:
            market_claims.extend([segment.segment, *segment.needs])
        market_citations = {url for claim in market_claims for url in claim.evidence}
        swot_claims = [
            *self.swot_risk_analysis.strengths,
            *self.swot_risk_analysis.weaknesses,
            *self.swot_risk_analysis.opportunities,
            *self.swot_risk_analysis.threats,
        ]
        for risk in self.swot_risk_analysis.execution_risks:
            swot_claims.extend([risk.risk, risk.potential_impact])
        swot_citations = {url for claim in swot_claims for url in claim.evidence}
        if not citations.issubset(available):
            raise ValueError("Competitor citations must match URLs in returned market research")
        if not market_citations.issubset(available):
            raise ValueError("Market analysis citations must match URLs in returned market research")
        if not swot_citations.issubset(available):
            raise ValueError("SWOT and risk citations must match URLs in returned market research")
        return self
