"""
Prompt Engineering Module
Creates structured prompts for responsible medical AI responses.
"""

from typing import Any, Dict, List,Optional


def build_prompt(
    data: Dict[str, Any],
    prob: float,
    factors: List[str],
    context: str,
    departments: Optional[List[str]] = None
) -> str:
    """
    Build a structured prompt for medical AI response
    
    Args:
        data: Patient medical data dictionary
        prob: Prediction probability (0-1)
        factors: Key risk factors identified by ML model
        context: Retrieved medical context from RAG
        departments: Recommended specialist departments
    
    Returns:
        Formatted prompt string for Gemini
    """
    
    factors_text = "\n".join([f"- {factor}" for factor in factors]) if factors else "- General health indicators"
    departments_text = "\n".join([f"- {dept}" for dept in departments]) if departments else "- General Physician"
    
    prompt = f"""
You are a responsible and ethical medical AI assistant. Your role is to provide health information 
based on medical guidelines and evidence. IMPORTANT: You are NOT a licensed medical professional and 
cannot provide medical diagnosis or treatment.

PATIENT ASSESSMENT DATA:
{format_patient_data(data)}

MACHINE LEARNING PREDICTION:
- Risk Probability: {prob:.2%}

KEY RISK FACTORS IDENTIFIED:
{factors_text}

RETRIEVED MEDICAL CONTEXT (from certified guidelines):
{context}

RECOMMENDED SPECIALISTS TO CONSULT:
{departments_text}

Return ONLY valid JSON with this schema:
{{
  "explanation": string,
  "recommendations": [string, string, string],
  "preventive_measures": [string, string, string],
  "suggested_specialists": [string],
  "source_citations": [string],
  "disclaimer": string
}}

Rules:
- Use only the provided medical context and patient details.
- Keep all recommendations evidence-grounded and non-diagnostic.
- Include 2-4 concise recommendation items and 2-4 preventive measure items.
- Include source_citations using the provided guideline context when possible.
- The disclaimer must clearly say this is not a diagnosis and the user should consult a qualified healthcare professional.
- Do not wrap the JSON in markdown fences.
"""
    
    return prompt


def format_patient_data(data: Dict[str, Any]) -> str:
    """
    Format patient data for display in prompt
    
    Args:
        data: Patient data dictionary
    
    Returns:
        Formatted patient data string
    """
    formatted = []
    
    # Map data keys to readable names
    key_mapping = {
        "HighBP": "High Blood Pressure",
        "HighChol": "High Cholesterol",
        "BMI": "Body Mass Index (BMI)",
        "Smoker": "Smoker Status",
        "Stroke": "History of Stroke",
        "HeartDiseaseorAttack": "Heart Disease or Attack",
        "PhysActivity": "Physical Activity",
        "Fruits": "Fruit Consumption",
        "Veggies": "Vegetable Consumption",
        "HvyAlcoholConsump": "Heavy Alcohol Consumption",
        "AnyHealthcare": "Healthcare Access",
        "GenHlth": "General Health Status",
        "MentHlth": "Mental Health Status",
        "PhysHlth": "Physical Health Status",
        "DiffWalk": "Difficulty Walking",
        "Sex": "Sex",
        "Age": "Age",
        "Education": "Education Level",
        "Income": "Income Level",
        "EducationLabel": "Education Level (Label)",
        "IncomeLabel": "Income Level (Label)",
    }
    
    for key, value in data.items():
        readable_key = key_mapping.get(key, key)
        formatted.append(f"- {readable_key}: {value}")
    
    return "\n".join(formatted) if formatted else "- No data provided"
