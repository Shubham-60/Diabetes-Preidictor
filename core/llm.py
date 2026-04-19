from __future__ import annotations

import json

from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_gemini_api_key

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


def _get_llm():
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )
    except Exception:
        return None


def generate_ai_response(prompt: str) -> str:
    llm = _get_llm()
    if llm is None:
        return _fallback_response_json()

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", None)
        if isinstance(content, str) and content:
            return content
        return _fallback_response_json()
    except Exception:
        return _fallback_response_json()
