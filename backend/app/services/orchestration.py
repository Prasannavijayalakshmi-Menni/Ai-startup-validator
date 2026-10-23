from app.schemas import AnalysisResponse, StartupIdea, SubmittedBudget
from app.services.idea_extraction import IdeaExtractionAgent
from app.services.market_research import MarketResearchAgent
from app.services.competitor_analysis import CompetitorAnalysisAgent
from app.services.market_analysis import MarketOpportunityAnalysisAgent
from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent


class ValidationReportService:
    def __init__(
        self,
        extractor=None,
        researcher=None,
        competitor_analyzer=None,
        market_analyzer=None,
        swot_risk_analyzer=None,
    ):
        self.extractor = extractor or IdeaExtractionAgent()
        self.researcher = researcher or MarketResearchAgent()
        self.competitor_analyzer = (
            competitor_analyzer or CompetitorAnalysisAgent()
        )
        self.market_analyzer = market_analyzer or MarketOpportunityAnalysisAgent()
        self.swot_risk_analyzer = swot_risk_analyzer or SWOTRiskAnalysisAgent()

    async def analyze(self, idea: StartupIdea) -> AnalysisResponse:
        structured = await self.extractor.extract(idea)
        research = await self.researcher.research(structured)

        market_analysis = await self.market_analyzer.analyze(structured, research)

        competitor_analysis = await self.competitor_analyzer.analyze(
            structured, research, market_analysis
        )
        swot_risk_analysis = await self.swot_risk_analyzer.analyze(
            structured, market_analysis, competitor_analysis, research
        )

        return AnalysisResponse(
            idea=structured,
            submitted_budget=SubmittedBudget(
                amount=idea.estimated_budget,
                currency=idea.estimated_budget_currency,
            ),
            market_research=research,
            market_analysis=market_analysis,
            competitor_analysis=competitor_analysis,
            swot_risk_analysis=swot_risk_analysis,
        )
