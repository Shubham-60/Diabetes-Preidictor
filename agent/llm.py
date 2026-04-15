import os

from google import genai



def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def generate_ai_response(prompt: str):
    client = _get_client()
    if client is None:
        return "AI service unavailable. Set GEMINI_API_KEY to enable the generated report."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception:
        return "AI service temporarily unavailable. Please try again."