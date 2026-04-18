import os
import json

from google import genai

FALLBACK_RESPONSE = {
    "explanation": "AI guidance is currently unavailable, so only the model prediction can be shown right now.",
    "recommendations": [
        "Review the predicted risk score with a healthcare professional.",
        "Continue monitoring blood pressure, BMI, and lifestyle habits.",
        "Re-run the assessment after correcting any missing or inaccurate inputs.",
    ],
    "preventive_measures": [
        "Maintain regular physical activity.",
        "Follow a balanced diet with controlled sugar intake.",
        "Schedule routine preventive health checkups.",
    ],
    "suggested_specialists": ["General Physician"],
    "source_citations": [],
    "disclaimer": "This assessment is a screening aid only and not a medical diagnosis. Please consult a qualified healthcare professional.",
}

def _fallback_response_json() -> str:
    return json.dumps(FALLBACK_RESPONSE)


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def generate_ai_response(prompt: str) -> str:
    client = _get_client()
    if client is None:
        return _fallback_response_json()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text or _fallback_response_json()
    except Exception:
        return _fallback_response_json()
