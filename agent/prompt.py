"""
Prompt Engineering Module
Creates structured prompts for responsible medical AI responses
"""

from typing import Any, Dict, List


def build_prompt(
    data: Dict[str, Any],
    prob: float,
    factors: List[str],
    context: str,
    departments: List[str] = None
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
    
    risk_level = "Low"
    if prob > 0.7:
        risk_level = "High"
    elif prob > 0.4:
        risk_level = "Medium"
    
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
- Risk Level: {risk_level}

KEY RISK FACTORS IDENTIFIED:
{factors_text}

RETRIEVED MEDICAL CONTEXT (from certified guidelines):
{context}

RECOMMENDED SPECIALISTS TO CONSULT:
{departments_text}

Based on the above information, generate a structured health report with the following sections:

Return STRICTLY in this format:

Risk Level:
Explanation:
Recommendations:
Preventive Measures:
Suggested Doctor/Specialist to Consult:
Disclaimer:

1. **Risk Assessment**: Brief statement of the risk level based on ML prediction
2. **Key Factors Explanation**: Explain the main health indicators and their significance (2-3 sentences)
3. **Health Recommendations**: Provide 3-4 evidence-based recommendations (bullet points)
4. **Preventive Measures**: Suggest lifestyle modifications and monitoring strategies
5. **Specialist Consultation**: Recommend which specialists should be consulted based on risk factors
6. **Important Disclaimer**: 

⚠️ CRITICAL DISCLAIMERS:
- This assessment is NOT a medical diagnosis
- This system does NOT replace professional medical advice
- Always consult qualified healthcare professionals for diagnosis and treatment
- In case of medical emergency, seek immediate professional medical care
- This system uses ML predictions to highlight potential risk areas, not to diagnose conditions

ENSURE:
- All recommendations are grounded in the provided medical context
- No medical claims or guarantees
- No hallucinations or advice not supported by the context
- Clear and factual language
- Appropriate tone: informative, cautious, and responsible
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
        "Income": "Income Level"
    }
    
    for key, value in data.items():
        readable_key = key_mapping.get(key, key)
        formatted.append(f"- {readable_key}: {value}")
    
    return "\n".join(formatted) if formatted else "- No data provided"
