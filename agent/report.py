"""
Health Report Generation Module
Creates structured health reports with AI and medical context
"""

from typing import Dict, Any, List, Tuple
import json
from datetime import datetime


def generate_health_report(
    user_data: Dict[str, Any],
    prob: float,
    risk_level: str,
    risk_factors: List[str],
    departments: List[str],
    ai_response: str = None,
    medical_context: str = None
) -> str:
    """
    Generate a structured health report
    
    Args:
        user_data: Patient health data
        prob: Diabetes risk probability
        risk_level: Risk level (Low/Medium/High)
        risk_factors: List of identified risk factors
        departments: List of recommended specialists
        ai_response: AI-generated insights (optional)
        medical_context: Retrieved medical guidelines (optional)
    
    Returns:
        Formatted report string
    """
    
    report = []
    report.append("=" * 70)
    report.append("AI-POWERED DIABETES RISK ASSESSMENT REPORT")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n" + "-" * 70)
    
    # Executive Summary
    report.append("\n1. EXECUTIVE SUMMARY")
    report.append("-" * 70)
    report.append(f"\nRisk Probability: {prob:.1%}")
    report.append(f"Risk Level: {risk_level}")
    report.append(f"\nAssessment: Based on machine learning analysis of health indicators,")
    report.append("the patient has been assigned a {} risk level for diabetes.".format(risk_level.lower()))
    
    # Patient Data
    report.append("\n" + "-" * 70)
    report.append("\n2. PATIENT DATA SUMMARY")
    report.append("-" * 70)
    for key, value in user_data.items():
        report.append(f"  {key}: {value}")
    
    # Risk Factors
    report.append("\n" + "-" * 70)
    report.append("\n3. IDENTIFIED RISK FACTORS")
    report.append("-" * 70)
    for i, factor in enumerate(risk_factors, 1):
        report.append(f"  {i}. {factor}")
    
    # Specialist Recommendations
    report.append("\n" + "-" * 70)
    report.append("\n4. RECOMMENDED SPECIALISTS")
    report.append("-" * 70)
    for i, dept in enumerate(departments, 1):
        report.append(f"  {i}. {dept}")
    
    # AI Insights
    if ai_response:
        report.append("\n" + "-" * 70)
        report.append("\n5. AI-GENERATED HEALTH INSIGHTS")
        report.append("-" * 70)
        report.append("\n" + ai_response)
    
    # Medical Context
    if medical_context:
        report.append("\n" + "-" * 70)
        report.append("\n6. RETRIEVED MEDICAL CONTEXT")
        report.append("-" * 70)
        report.append("\n" + medical_context[:500] + "..." if len(medical_context) > 500 else "\n" + medical_context)
    
    # Disclaimer
    report.append("\n" + "-" * 70)
    report.append("\n7. IMPORTANT DISCLAIMER")
    report.append("-" * 70)
    report.append("""
⚠️  CRITICAL LEGAL AND MEDICAL DISCLAIMERS:

1. This assessment is NOT a medical diagnosis. It is a risk screening tool.

2. This system does NOT replace professional medical advice, diagnosis, 
   or treatment by qualified healthcare professionals.

3. The predictions are based on machine learning models trained on 
   historical data and should be interpreted as probability estimates only.

4. All health recommendations should be discussed with qualified 
   healthcare professionals before taking any action.

5. In case of medical emergency, seek immediate professional medical care.

6. The AI responses are generated based on medical guidelines and context 
   retrieval (RAG), designed to reduce hallucinations and ensure grounded 
   information.

7. Individual health outcomes may vary based on additional medical factors 
   not captured in this assessment.

⚠️  PLEASE CONSULT WITH YOUR HEALTHCARE PROVIDER FOR:
   - Proper medical diagnosis
   - Personalized treatment plans
   - Specialist referrals
   - Medication management
   - Long-term health management strategies
""")
    
    # System Information
    report.append("\n" + "-" * 70)
    report.append("\nSYSTEM TECHNOLOGY:")
    report.append("  - ML Model: Random Forest Classifier")
    report.append("  - LLM: Google Gemini")
    report.append("  - RAG Pipeline: PDF Chunking + FAISS Retrieval + CrossEncoder Reranking")
    report.append("  - Knowledge Base: Medical Guidelines (WHO, ADA)")
    report.append("  - Framework: Streamlit")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def export_report_json(
    user_data: Dict[str, Any],
    prob: float,
    risk_level: str,
    risk_factors: List[str],
    departments: List[str],
    ai_response: str = None
) -> str:
    """
    Export report as JSON
    
    Args:
        user_data: Patient health data
        prob: Diabetes risk probability
        risk_level: Risk level
        risk_factors: List of risk factors
        departments: List of specialists
        ai_response: AI response text
    
    Returns:
        JSON string
    """
    
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "assessment": {
            "risk_probability": prob,
            "risk_level": risk_level,
            "patient_data": user_data,
            "risk_factors": risk_factors,
            "recommended_specialists": departments,
        },
        "ai_insights": ai_response,
        "disclaimer": "This is a screening tool only, not a medical diagnosis. Consult healthcare professionals.",
        "system_info": {
            "ml_model": "Random Forest",
            "llm": "Gemini",
            "rag_method": "FAISS + SentenceTransformers + CrossEncoder"
        }
    }
    
    return json.dumps(report_dict, indent=2)


def get_risk_factor_explanation(factor: str) -> str:
    """
    Get explanation for a specific risk factor
    
    Args:
        factor: Risk factor name
    
    Returns:
        Explanation text
    """
    
    explanations = {
        "High Blood Pressure": "Hypertension increases strain on blood vessels and is associated with diabetes risk.",
        "High Cholesterol": "High cholesterol levels can increase cardiovascular complications in diabetes.",
        "High BMI": "Obesity (BMI > 30) significantly increases insulin resistance and diabetes risk.",
        "Smoking Status": "Smoking impairs glucose metabolism and increases complication risk.",
        "History of Stroke": "Previous stroke indicates cardiovascular disease, a common diabetes complication.",
        "Heart Disease": "Existing heart disease increases diabetes-related cardiovascular risk.",
        "Lack of Physical Activity": "Sedentary lifestyle reduces insulin sensitivity and glucose control.",
        "Poor General Health": "Self-reported poor health may indicate undetected metabolic issues.",
        "Age over 45": "Diabetes risk increases significantly with age, especially after 45 years.",
    }
    
    return explanations.get(factor, f"Risk factor: {factor}")


def format_specialist_info(department: str) -> str:
    """
    Get information about a specialist
    
    Args:
        department: Specialist name
    
    Returns:
        Information text
    """
    
    specialist_info = {
        "Endocrinologist": "Specialist in hormones and metabolism, including diabetes management and hormone-related conditions.",
        "Diabetologist": "Specialist focused specifically on diabetes prevention, management, and complications.",
        "Cardiologist": "Heart specialist to assess cardiovascular risk and manage heart disease in diabetes.",
        "Nutritionist / Dietitian": "Healthcare professional specializing in medical nutrition therapy and diet planning.",
        "Lifestyle Medicine Specialist": "Doctor focusing on lifestyle interventions including exercise, diet, and stress management.",
        "General Physician": "Primary care doctor for overall health management and coordination of care.",
    }
    
    return specialist_info.get(department, f"Specialist: {department}")
