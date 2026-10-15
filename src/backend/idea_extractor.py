import os
import json
from dotenv import load_dotenv
from google import genai
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
def extract_idea(startup_idea):
    if not api_key:
        raise ValueError("API key missing. Check your .env file.")

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze this startup idea and extract:
    1. problem_statement
    2. proposed_solution
    3. target_customers
    4. industry
    5. key_features
    Return only valid JSON with these five keys.
    Startup idea: {startup_idea}
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        if not response.text:
            raise ValueError("The AI returned an empty response.")
        idea_data = json.loads(response.text)
        return idea_data
    except json.JSONDecodeError:
        raise ValueError("The AI returned invalid JSON.")
if __name__ == "__main__":
    startup_idea = input("Enter your startup idea: ")
    try:
        idea_data = extract_idea(startup_idea)
        print(json.dumps(idea_data, indent=4))
    except Exception as e:
        print("Error:", e)
