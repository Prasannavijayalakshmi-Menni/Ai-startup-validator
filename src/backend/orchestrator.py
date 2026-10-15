
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
        # Task 2: Extract structured startup information
        structured_idea = extract_idea(startup_idea.strip())

        if not isinstance(structured_idea, dict):
            raise ValueError(
                "Idea extraction must return a dictionary."
            )

        # Task 3: Retrieve market and competitor research
        search_results = search_startup_idea(
            startup_idea.strip()
        )

        if not isinstance(search_results, dict):
            raise ValueError(
                "Web search must return a dictionary."
            )

        # Combine both outputs
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

    result = orchestrate_startup_validation(idea)

    print("\n--- Startup Validation Result ---")
    print(result)
