
import os
import json
import time
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
    key_features must be a list of strings.
    Startup idea: {startup_idea}
    """

    models_to_try = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash"
    ]

    last_error = None

    for model in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json"
                    }
                )

                if not response.text:
                    raise ValueError(
                        "The AI returned an empty response."
                    )

                idea_data = json.loads(response.text)

                required_keys = {
                    "problem_statement",
                    "proposed_solution",
                    "target_customers",
                    "industry",
                    "key_features"
                }

                if not required_keys.issubset(idea_data.keys()):
                    raise ValueError(
                        "The AI response is missing required fields."
                    )

                print(f"Extraction successful using {model}")
                return idea_data

            except Exception as error:
                last_error = error
                message = str(error)

                if "503" in message or "UNAVAILABLE" in message:
                    if attempt == 0:
                        time.sleep(3)
                    else:
                        break
                else:
                    raise

    raise RuntimeError(
        f"All available model attempts failed: {last_error}"
    )
