import json
from getpass import getpass
from tavily import TavilyClient


def main():
    api_key = getpass("Enter your Tavily API key: ")
    client = TavilyClient(api_key=api_key)

    queries = {
        "Market Demand": (
            "India college students affordable healthy food demand "
            "food habits nutrition meal planning market trends"
        ),
        "Competitor Analysis": (
            "India apps and startups healthy meal planning "
            "affordable meals students competitors"
        ),
    }

    all_results = {}

    for category, query in queries.items():
        print(f"\nSearching: {category}")

        response = client.search(query=query, max_results=5)
        results = response.get("results", [])
        all_results[category] = results

        for number, result in enumerate(results, start=1):
            print(f"\n--- {category}: Result {number} ---")
            print("Title:", result.get("title", "No title"))
            print("URL:", result.get("url", "No URL"))
            print("Summary:", result.get("content", "No summary"))

    with open(
        "tavily_search_results.json", "w", encoding="utf-8"
    ) as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

    print("\nResults saved to tavily_search_results.json")


if __name__ == "__main__":
    main()
