
%%writefile /content/AI-Startup-Idea-Validator/orchestrator.py
import os
from tavily import TavilyClient
from idea_extractor import extract_idea
from tavily_search import search_startup_idea


def orchestrate_startup_validation(startup_idea: str) -> dict:
    """Coordinate idea extraction and market research."""

    if not startup_idea or not startup_idea.strip():
        return {
            "success": False,
            "error": "Please enter a startup idea."
        }

    try:
        idea = startup_idea.strip()

        # Task 2: Extract structured startup information
        structured_idea = extract_idea(idea)

        if not isinstance(structured_idea, dict):
            raise ValueError(
                "Idea extraction must return a dictionary."
            )

        # Task 3: Create client and retrieve research
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is missing.")

        client = TavilyClient(api_key=api_key)
        search_results = search_startup_idea(idea, client)

        if not isinstance(search_results, dict):
            raise ValueError(
                "Web search must return a dictionary."
            )

        return {
            "success": True,
            "structured_idea": structured_idea,
            "market_research": search_results
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


if __name__ == "__main__":
    idea = input("Enter your startup idea: ")
    print(orchestrate_startup_validation(idea))
