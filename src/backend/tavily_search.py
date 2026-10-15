import os
import json
from tavily import TavilyClient


def search_startup_idea(startup_idea, client):
    """Search for market demand and competitors for a startup idea."""

    if not startup_idea or not startup_idea.strip():
        raise ValueError("Please enter a startup idea.")

    queries = {
        "Market Demand": (
            f"{startup_idea} market demand trends target customers India"
        ),
        "Competitor Analysis": (
            f"{startup_idea} competitors alternatives similar products India"
        ),
    }

    all_results = {}

    for category, query in queries.items():
        print(f"\nSearching: {category}")
        print("Search query:", query)

        try:
            response = client.search(query=query, max_results=5)
            results = response.get("results", [])
            all_results[category] = results

            for number, result in enumerate(results, start=1):
                print(f"\n--- {category}: Result {number} ---")
                print("Title:", result.get("title", "No title"))
                print("URL:", result.get("url", "No URL"))
                print("Summary:", result.get("content", "No summary"))

        except Exception as error:
            print(f"Search failed for {category}: {error}")
            all_results[category] = []

    return all_results


def main():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        print("TAVILY_API_KEY is missing.")
        print("Please configure it as an environment variable.")
        return

    startup_idea = input("Enter your startup idea: ").strip()

    if not startup_idea:
        print("Startup idea cannot be empty.")
        return

    client = TavilyClient(api_key=api_key)
    results = search_startup_idea(startup_idea, client)

    with open(
        "tavily_search_results.json", "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nResults saved to tavily_search_results.json")


if __name__ == "__main__":
    main()
