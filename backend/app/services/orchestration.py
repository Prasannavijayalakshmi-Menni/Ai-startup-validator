import logging

from app.schemas import (
    AnalysisResponse,
    CompetitorAnalysis,
    MarketOpportunityAnalysis,
    SWOTRiskAnalysis,
    StartupIdea,
    SubmittedBudget,
)
from app.services.competitor_analysis import CompetitorAnalysisAgent
from app.services.idea_extraction import IdeaExtractionAgent, ServiceError
from app.services.market_analysis import MarketOpportunityAnalysisAgent
from app.services.market_research import MarketResearchAgent
from app.services.swot_risk_analysis import SWOTRiskAnalysisAgent

logger = logging.getLogger(__name__)


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

        market_analysis = None
        market_error = None
        try:
            market_analysis = await self.market_analyzer.analyze(structured, research)
        except ServiceError as exc:
            market_error = exc
            logger.warning("Market analysis agent failed: %s", exc.message)

        competitor_analysis = None
        competitor_error = None
        try:
            competitor_analysis = await self.competitor_analyzer.analyze(
                structured, research, market_analysis
            )
        except ServiceError as exc:
            competitor_error = exc
            logger.warning("Competitor analysis agent failed: %s", exc.message)

        swot_risk_analysis = None
        swot_error = None
        try:
            swot_risk_analysis = await self.swot_risk_analyzer.analyze(
                structured,
                market_analysis or MarketOpportunityAnalysis(
                    demand_assessment={"claim": "Unknown", "classification": "unknown", "evidence": []}
                ),
                competitor_analysis or CompetitorAnalysis(),
                research,
            )
        except ServiceError as exc:
            swot_error = exc
            logger.warning("SWOT/risk analysis agent failed: %s", exc.message)

        # If all three analysis stages failed, propagate the earliest error so
        # the client is informed why no analysis was produced.
        if market_analysis is None and competitor_analysis is None and swot_risk_analysis is None:
            raise market_error or competitor_error or swot_error or ServiceError(
                "All analysis stages failed to produce output.", 503
            )

        return AnalysisResponse(
            idea=structured,
            submitted_budget=SubmittedBudget(
                amount=idea.estimated_budget,
                currency=idea.estimated_budget_currency,
            ),
            market_research=research,
            market_analysis=market_analysis or MarketOpportunityAnalysis(
                demand_assessment={"claim": "Unknown", "classification": "unknown", "evidence": []}
            ),
            competitor_analysis=competitor_analysis or CompetitorAnalysis(),
            swot_risk_analysis=swot_risk_analysis or SWOTRiskAnalysis(),
        )
