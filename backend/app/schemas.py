from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

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

class AnalysisResponse(BaseModel):
    idea: StructuredIdea
    submitted_budget: SubmittedBudget
    market_research: dict[str, list[Source]]
    research_note: str = "Search snippets are provider-supplied excerpts, not independently verified facts. Review each linked source before relying on claims or figures."
