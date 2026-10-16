from app.schemas import AnalysisResponse, StartupIdea, SubmittedBudget
from app.services.idea_extraction import IdeaExtractionAgent
from app.services.market_research import MarketResearchAgent

class ValidationReportService:
    def __init__(self, extractor=None, researcher=None):
        self.extractor = extractor or IdeaExtractionAgent()
        self.researcher = researcher or MarketResearchAgent()

    async def analyze(self, idea: StartupIdea) -> AnalysisResponse:
        structured = await self.extractor.extract(idea)
        research = await self.researcher.research(structured)
        return AnalysisResponse(
            idea=structured,
            submitted_budget=SubmittedBudget(
                amount=idea.estimated_budget,
                currency=idea.estimated_budget_currency,
            ),
            market_research=research,
        )
