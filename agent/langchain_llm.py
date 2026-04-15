import os

from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GEMINI_API_KEY")
llm = (
    ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )
    if api_key
    else None
)


def generate_response(prompt: str):
    if llm is None:
        return "AI service unavailable. Set GEMINI_API_KEY to enable the generated report."

    try:
        res = llm.invoke(prompt)
        return res.content if getattr(res, "content", None) else "No response generated"
    except Exception:
        return "AI service temporarily unavailable. Please try again."
import os

from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GEMINI_API_KEY")
llm = (
    ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )
    if api_key
    else None
)


def generate_response(prompt: str):
    if llm is None:
        return "AI service unavailable. Set GEMINI_API_KEY to enable the generated report."

    try:
        res = llm.invoke(prompt)
        return res.content if getattr(res, "content", None) else "No response generated"
    except Exception:
        return "AI service temporarily unavailable. Please try again."