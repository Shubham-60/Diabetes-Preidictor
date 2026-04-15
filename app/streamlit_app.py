import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm import generate_ai_response
from agent.prompt import build_prompt
from agent.rag_faiss import load_pdfs, chunk_text, create_index, search
from agent.reranker import rerank
from agent.utils import extract_factors

THRESHOLD = 0.58
FEATURE_ORDER = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
APP_DIR = PROJECT_ROOT
MODEL_PATH = APP_DIR / "models" / "lr_model.pkl"
SCALER_PATH = APP_DIR / "models" / "scaler.pkl"
ENV_PATH = APP_DIR / ".env"

INCOME_OPTIONS = {
    "1 — Less than $10,000": 1,
    "2 — $10,000 to < $15,000": 2,
    "3 — $15,000 to < $20,000": 3,
    "4 — $20,000 to < $25,000": 4,
    "5 — $25,000 to < $35,000": 5,
    "6 — $35,000 to < $50,000": 6,
    "7 — $50,000 to < $75,000": 7,
    "8 — $75,000 or more": 8,
}
EDUCATION_OPTIONS = {
    "1 — Never attended school or only kindergarten": 1,
    "2 — Grades 1 through 8 (Elementary)": 2,
    "3 — Grades 9 through 11 (Some high school)": 3,
    "4 — Grade 12 or GED (High school graduate)": 4,
    "5 — College 1 year to 3 years (Some college or technical school)": 5,
    "6 — College 4 years or more (College graduate)": 6,
}
EDUCATION_LABELS_BY_CODE = {value: label for label, value in EDUCATION_OPTIONS.items()}
FIELD_LABELS = {
    "Smoker": "Smoker",
    "PhysActivity": "Physically Active",
    "Fruits": "Regular Fruit Intake",
    "Veggies": "Regular Vegetable Intake",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Recent Cholesterol Check",
    "DiffWalk": "Difficulty Walking",
    "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Skipped Doctor Due to Cost",
    "BMI": "Body Mass Index (BMI)",
    "GenHlth": "General Health",
    "MentHlth": "Poor Mental Health Days",
    "PhysHlth": "Poor Physical Health Days",
    "Sex": "Sex",
    "Age": "Age",
    "Education": "Education Level",
    "Income": "Income Level",
}


def init_session_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "form"
    if "form_values" not in st.session_state:
        st.session_state.form_values = {}
    if "features" not in st.session_state:
        st.session_state.features = {}
    if "raw_inputs" not in st.session_state:
        st.session_state.raw_inputs = {}
    if "risk_probability" not in st.session_state:
        st.session_state.risk_probability = None
    if "risk_class" not in st.session_state:
        st.session_state.risk_class = None
    if "needs_prediction" not in st.session_state:
        st.session_state.needs_prediction = False
    if "ai_response" not in st.session_state:
        st.session_state.ai_response = ""
    if "key_factors" not in st.session_state:
        st.session_state.key_factors = []
    if "specialists" not in st.session_state:
        st.session_state.specialists = []
    if "source_context" not in st.session_state:
        st.session_state.source_context = ""


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
            :root {
                --bg-1: #f0f4ff; --bg-2: #e8f8f4; --bg-3: #fdf4ff;
                --glass: rgba(255,255,255,0.65); --soft-border: rgba(255,255,255,0.55);
                --text-main:#0f172a; --text-muted:#475569;
                --accent-a:#00c9a7; --accent-b:#7c3aed;
            }
            .stApp { font-family:'DM Sans',sans-serif; color:var(--text-main); background:linear-gradient(140deg,var(--bg-1),var(--bg-2),var(--bg-3)); }
            .block-container { max-width:1020px; padding-top:2.2rem; }
            .hero-shell,.stApp [data-testid="stVerticalBlockBorderWrapper"] { background:var(--glass); border:1px solid var(--soft-border); border-radius:18px; backdrop-filter:blur(12px); }
            .hero-shell { padding:1.1rem; margin-bottom:1rem; }
            .hero-chip { display:inline-block; border-radius:999px; padding:.22rem .62rem; font-size:.75rem; font-weight:700; background:linear-gradient(120deg, rgba(0,201,167,.16), rgba(124,58,237,.16)); }
            .hero-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:2rem; font-weight:800; margin:.25rem 0; }
            .hero-subtitle { color:var(--text-muted); }
            .section-shell { display:flex; gap:.75rem; padding:.78rem .9rem; border-radius:14px; border:1px solid rgba(255,255,255,.6); background:rgba(255,255,255,.55); margin-bottom:.65rem; }
            .section-accent { width:6px; min-height:44px; border-radius:999px; background:linear-gradient(180deg,var(--accent-a),var(--accent-b)); }
            .section-accent.teal { background: linear-gradient(180deg, #00c9a7, #2dd4bf); }
            .section-accent.purple { background: linear-gradient(180deg, #7c3aed, #a855f7); }
            .section-accent.pink { background: linear-gradient(180deg, #ec4899, #f472b6); }
            .section-accent.orange { background: linear-gradient(180deg, #f59e0b, #fb923c); }
            .section-title { font-size:1.06rem; font-weight:700; }
            .section-caption { color:var(--text-muted); font-size:.9rem; }
            .helper-text { color:var(--text-muted); font-size:.8rem; margin-top:-.35rem; margin-bottom:.7rem; }
            .disclaimer { margin-top:1.1rem; text-align:center; font-size:.82rem; color:var(--text-muted); background:rgba(255,255,255,.6); border:1px solid rgba(255,255,255,.62); border-radius:12px; padding:.55rem .7rem; }
            .summary-compact { margin-top:.35rem; border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,.62); }
            .summary-row { display:grid; grid-template-columns:minmax(180px,1fr) minmax(120px,.8fr); gap:.8rem; padding:.5rem .8rem; }
            .summary-row:nth-child(odd) { background:rgba(255,255,255,.7); }
            .summary-row:nth-child(even) { background:rgba(241,245,249,.62); }
            .summary-key { color:#334155; font-weight:600; }
            .summary-value { color:#0f172a; font-weight:600; text-align:right; }
            @media (max-width:768px){ .summary-row { grid-template-columns:1fr; } .summary-value { text-align:left; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_helper(text: str) -> None:
    st.markdown(f"<div class='helper-text'>{text}</div>", unsafe_allow_html=True)


def scroll_to_top() -> None:
    components.html("""<script>window.scrollTo({top:0,left:0,behavior:'auto'});</script>""", height=0)


def section_header(title: str, caption: str, accent: str = "teal") -> None:
    st.markdown(
        f"""
        <div class='section-shell'>
            <div class='section-accent {accent}'></div>
            <div>
                <div class='section-title'>{title}</div>
                <div class='section-caption'>{caption}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def age_to_category(age_years: int) -> int:
    if age_years <= 24: return 1
    if age_years <= 29: return 2
    if age_years <= 34: return 3
    if age_years <= 39: return 4
    if age_years <= 44: return 5
    if age_years <= 49: return 6
    if age_years <= 54: return 7
    if age_years <= 59: return 8
    if age_years <= 64: return 9
    if age_years <= 69: return 10
    if age_years <= 74: return 11
    if age_years <= 79: return 12
    return 13


def ensure_gemini_api_key() -> bool:
    if os.getenv("GEMINI_API_KEY"):
        return True
    if not ENV_PATH.exists():
        return False
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "GEMINI_API_KEY":
            os.environ["GEMINI_API_KEY"] = value.strip().strip('"').strip("'")
            return bool(os.environ["GEMINI_API_KEY"])
    return False


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Missing model artifacts in models/")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@st.cache_data(show_spinner=False)
def load_docs_cached():
    return load_pdfs()


@st.cache_resource(show_spinner=False)
def build_index_cached(chunks):
    return create_index(chunks)


def predict_probability(features: dict) -> float:
    model, scaler = load_model_artifacts()
    row = {feature: features[feature] for feature in FEATURE_ORDER}
    frame = pd.DataFrame([row], columns=FEATURE_ORDER)
    transformed = scaler.transform(frame)
    return max(0.0, min(float(model.predict_proba(transformed)[0][1]), 1.0))


def get_doctor_recommendation(factors):
    departments = []
    if "High Glucose" in factors:
        departments.extend(["Endocrinologist", "Diabetologist"])
    if "High BMI" in factors:
        departments.append("Nutritionist / Dietitian")
    if "Age Risk" in factors:
        departments.append("General Physician")
    return list(dict.fromkeys(departments)) or ["General Physician"]


def run_ai_pipeline(raw_inputs: dict, model_features: dict, probability: float):
    factors = extract_factors(raw_inputs)
    query = " ".join(factors)
    texts = load_docs_cached()
    chunks = chunk_text(texts)
    index, chunks = build_index_cached(chunks)
    retrieved = search(query, index, chunks)
    reranked_texts = rerank(query, [r["content"] for r in retrieved])
    context = "\n".join(reranked_texts)
    specialists = get_doctor_recommendation(factors)

    prompt_data = dict(raw_inputs)
    prompt_data.update(model_features)
    prompt = build_prompt(prompt_data, probability, factors, context, specialists)
    ai_response = generate_ai_response(prompt)

    source_snippets = []
    for passage in reranked_texts[:2]:
        source = "Guideline"
        for item in retrieved:
            if item["content"] == passage:
                source = item.get("source", "Guideline")
                break
        source_snippets.append(f"[{source}] {passage[:180]}...")

    return factors, specialists, ai_response, "\n".join(source_snippets)


def render_form() -> None:
    st.markdown(
        """
        <div class='hero-shell'>
            <div class='hero-chip'>Healthcare AI • Preventive Insights</div>
            <div class='hero-title'>Diabetes Risk Prediction</div>
            <div class='hero-subtitle'>Enter your health details to assess your diabetes risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    defaults = st.session_state.form_values

    with st.container(border=True):
        section_header("Section 1 — Lifestyle", "Daily habits and routine indicators", accent="teal")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            smoker = st.toggle("Smoker", value=defaults.get("smoker", False))
            render_helper("Select Yes if you currently smoke regularly.")
            phys_activity = st.toggle("Physically Active", value=defaults.get("phys_activity", True))
            render_helper("Select Yes if you are physically active.")
            fruits = st.toggle("Regular Fruit Intake", value=defaults.get("fruits", True))
            render_helper("Select Yes if fruits are part of your regular diet.")
        with col2:
            veggies = st.toggle("Regular Vegetable Intake", value=defaults.get("veggies", True))
            render_helper("Select Yes if vegetables are part of your regular diet.")
            alcohol = st.toggle("Heavy Alcohol Consumption", value=defaults.get("alcohol", False))
            render_helper("Select Yes if heavy alcohol consumption applies.")

    with st.container(border=True):
        section_header("Section 2 — Medical History", "Known clinical and access-related history", accent="purple")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            high_bp = st.toggle("High Blood Pressure", value=defaults.get("high_bp", False))
            render_helper("History of high blood pressure.")
            high_chol = st.toggle("High Cholesterol", value=defaults.get("high_chol", False))
            render_helper("History of high cholesterol.")
            chol_check = st.toggle("Recent Cholesterol Check", value=defaults.get("chol_check", True))
            render_helper("Had cholesterol checked in recent care visits.")
        with col2:
            diff_walk = st.toggle("Difficulty Walking", value=defaults.get("diff_walk", False))
            render_helper("Difficulty in walking or climbing stairs.")
            any_healthcare = st.toggle("Healthcare Coverage", value=defaults.get("any_healthcare", True))
            render_helper("Access to healthcare coverage.")
            no_doc_cost = st.toggle("Skipped Doctor Due to Cost", value=defaults.get("no_doc_cost", False))
            render_helper("Could not visit a doctor due to cost.")

    with st.container(border=True):
        section_header("Section 3 — Health Condition", "Current health indicators", accent="pink")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            glucose = st.number_input("Glucose", min_value=70, max_value=250, value=int(defaults.get("glucose", 120)))
            render_helper("Fasting glucose level.")
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=float(defaults.get("bmi", 26.8)), step=0.1)
            render_helper("Body Mass Index range: 10 to 60.")
            gen_hlth = st.slider("General Health", min_value=1, max_value=5, value=int(defaults.get("gen_hlth", 3)))
            render_helper("General health rating: 1 (excellent) to 5 (poor).")
        with col2:
            ment_hlth = st.slider("Poor Mental Health Days", min_value=0, max_value=30, value=int(defaults.get("ment_hlth", 4)))
            render_helper("Number of poor mental health days in last 30 days.")
            phys_hlth = st.slider("Poor Physical Health Days", min_value=0, max_value=30, value=int(defaults.get("phys_hlth", 3)))
            render_helper("Number of poor physical health days in last 30 days.")

    with st.container(border=True):
        section_header("Section 4 — Demographics", "Basic demographic information", accent="orange")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            sex = st.selectbox("Sex", options=["Male", "Female"], index=0 if defaults.get("sex", "Male") == "Male" else 1)
            render_helper("Biological sex used by model features.")
            age = st.slider("Age", min_value=18, max_value=100, value=int(defaults.get("age", 42)))
            render_helper("Select your age in years.")
            education_options = list(EDUCATION_OPTIONS.keys())
            default_education_code = int(defaults.get("education", 4))
            default_education_label = defaults.get("education_label", EDUCATION_LABELS_BY_CODE.get(default_education_code, education_options[3]))
            if default_education_label not in education_options:
                default_education_label = EDUCATION_LABELS_BY_CODE.get(default_education_code, education_options[3])
            education_label = st.selectbox("Education Level", options=education_options, index=education_options.index(default_education_label))
            education = EDUCATION_OPTIONS[education_label]
            render_helper("Select your education level category used by the model.")
        with col2:
            income_options = list(INCOME_OPTIONS.keys())
            default_income_label = defaults.get("income_label", income_options[4])
            if default_income_label not in income_options:
                default_income_label = income_options[4]
            income_label = st.selectbox("Income Level", options=income_options, index=income_options.index(default_income_label))
            render_helper("Household income category used in the model.")

    if st.button("Predict My Risk", type="primary", use_container_width=True):
        scroll_to_top()
        st.session_state.form_values = {
            "smoker": smoker, "phys_activity": phys_activity, "fruits": fruits, "veggies": veggies,
            "alcohol": alcohol, "high_bp": high_bp, "high_chol": high_chol, "chol_check": chol_check,
            "diff_walk": diff_walk, "any_healthcare": any_healthcare, "no_doc_cost": no_doc_cost,
            "glucose": glucose, "bmi": bmi, "gen_hlth": gen_hlth, "ment_hlth": ment_hlth,
            "phys_hlth": phys_hlth, "sex": sex, "age": age, "education": education,
            "education_label": education_label, "income_label": income_label,
        }

        st.session_state.features = {
            "Smoker": int(smoker), "PhysActivity": int(phys_activity), "Fruits": int(fruits),
            "Veggies": int(veggies), "HvyAlcoholConsump": int(alcohol), "HighBP": int(high_bp),
            "HighChol": int(high_chol), "CholCheck": int(chol_check), "DiffWalk": int(diff_walk),
            "AnyHealthcare": int(any_healthcare), "NoDocbcCost": int(no_doc_cost), "BMI": float(bmi),
            "GenHlth": int(gen_hlth), "MentHlth": int(ment_hlth), "PhysHlth": int(phys_hlth),
            "Sex": 1 if sex == "Male" else 0, "Age": age_to_category(int(age)),
            "Education": int(education), "Income": INCOME_OPTIONS[income_label],
        }

        st.session_state.raw_inputs = {
            "Age": int(age), "Glucose": float(glucose), "BMI": float(bmi), "HighBP": int(high_bp),
            "HighChol": int(high_chol), "Smoker": int(smoker), "PhysActivity": int(phys_activity),
            "Fruits": int(fruits), "Veggies": int(veggies), "CholCheck": int(chol_check),
            "Stroke": 0, "HeartDiseaseorAttack": 0, "HvyAlcoholConsump": int(alcohol),
            "AnyHealthcare": int(any_healthcare), "NoDocbcCost": int(no_doc_cost), "GenHlth": int(gen_hlth),
            "MentHlth": int(ment_hlth), "PhysHlth": int(phys_hlth), "DiffWalk": int(diff_walk),
            "Sex": 1 if sex == "Male" else 0, "Education": int(education), "Income": INCOME_OPTIONS[income_label],
        }

        st.session_state.needs_prediction = True
        st.session_state.page = "loading"
        st.rerun()

    st.markdown("<div class='disclaimer'>Disclaimer: This is an ML model output only, not a medical diagnosis, and it provides no medical recommendations.</div>", unsafe_allow_html=True)


def render_loading() -> None:
    scroll_to_top()
    st.markdown("""<div class='hero-shell'><div class='hero-chip'>Analyzing</div><div class='hero-title'>Preparing Your Assessment</div><div class='hero-subtitle'>Please wait while we process your inputs</div></div>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""<div class='loader-shell'><div class='loader-orbit'><div class='loader-dot'></div></div><div class='loader-text'>Running clinical inference...</div></div>""", unsafe_allow_html=True)

    if st.session_state.needs_prediction:
        time.sleep(0.4)
        try:
            probability = predict_probability(st.session_state.features)
            factors, specialists, ai_response, source_context = run_ai_pipeline(st.session_state.raw_inputs, st.session_state.features, probability)
        except Exception as error:
            st.session_state.needs_prediction = False
            st.error("Model inference failed. Ensure models/lr_model.pkl and models/scaler.pkl exist.")
            st.caption(str(error))
            if st.button("Back to Form", use_container_width=True):
                st.session_state.page = "form"
                st.rerun()
            return

        st.session_state.risk_probability = probability
        st.session_state.risk_class = int(probability >= THRESHOLD)
        st.session_state.key_factors = factors
        st.session_state.specialists = specialists
        st.session_state.ai_response = ai_response
        st.session_state.source_context = source_context
        st.session_state.needs_prediction = False
        st.session_state.page = "result"
        st.rerun()


def render_result() -> None:
    probability = st.session_state.risk_probability
    if probability is None:
        st.session_state.page = "form"
        st.rerun()

    scroll_to_top()

    if st.button("← Back", key="top_back_result"):
        st.session_state.page = "form"
        st.rerun()

    risk_class = int(probability >= THRESHOLD)
    risk_label = "Low Risk" if risk_class == 0 else "High Risk"
    score_text = f"{int(round(probability * 100))}%"

    st.markdown(f"""
    <div class='hero-shell'>
      <div class='hero-chip'>Assessment Summary</div>
      <div class='hero-title'>Your Diabetes Risk Assessment</div>
      <div class='hero-subtitle'>Model output with clinically oriented interpretation</div>
    </div>
    """, unsafe_allow_html=True)
    st.success(f"Risk Score: {score_text} • {risk_label}")

    with st.container(border=True):
        st.subheader("AI Medical Guidance")

        # --- Key Factors ---
        st.markdown("### ⚠️ Key Risk Factors")
        for f in st.session_state.key_factors:
            st.write(f"• {f}")

        # --- Specialists ---
        st.markdown("### 👨‍⚕️ Recommended Specialists")
        for specialist in st.session_state.specialists:
            st.write(f"• {specialist}")

        # --- Parse AI Response ---
        def parse_response(text):
            sections = {"Risk Level": "", "Explanation": "", "Recommendations": "", "Preventive Measures": ""}
            current = None
            for line in text.split("\n"):
                line = line.strip()
                if "Risk Level:" in line:
                    current = "Risk Level"
                    sections[current] = line.replace("Risk Level:", "").strip()
                elif "Explanation:" in line:
                    current = "Explanation"
                    sections[current] = line.replace("Explanation:", "").strip()
                elif "Recommendations:" in line:
                    current = "Recommendations"
                    sections[current] = line.replace("Recommendations:", "").strip()
                elif "Preventive Measures:" in line:
                    current = "Preventive Measures"
                    sections[current] = line.replace("Preventive Measures:", "").strip()
                elif current:
                    sections[current] += " " + line
            return sections

        parsed = parse_response(st.session_state.ai_response)

        # --- Risk Level ---
        st.markdown("### 🩺 Risk Assessment")
        if "low" in parsed["Risk Level"].lower():
            st.success(f"Risk Level: {parsed['Risk Level']}")
        elif "medium" in parsed["Risk Level"].lower():
            st.warning(f"Risk Level: {parsed['Risk Level']}")
        else:
            st.error(f"Risk Level: {parsed['Risk Level']}")

        # --- Explanation ---
        st.markdown("### 📋 Explanation")
        st.write(parsed["Explanation"])

        # --- Recommendations ---
        st.markdown("### 💡 Recommendations")
        st.write(parsed["Recommendations"])

        # --- Preventive Measures ---
        st.markdown("### 🛡 Preventive Measures")
        st.write(parsed["Preventive Measures"])

        # --- Source Context ---
        st.markdown("### 📚 Source")
        for line in st.session_state.source_context.split("\n")[:2]:
            st.caption(line)

    with st.container(border=True):
        section_header("Input Summary", "Feature values used for this prediction", accent="teal")
        grouped_features = {
            "Lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
            "Medical History": ["HighBP", "HighChol", "CholCheck", "AnyHealthcare", "NoDocbcCost", "DiffWalk"],
            "Health Condition": ["BMI", "GenHlth", "MentHlth", "PhysHlth"],
            "Demographics": ["Sex", "Age", "Education", "Income"],
        }
        for group_name, keys in grouped_features.items():
            rows = ""
            for key in keys:
                value = st.session_state.features.get(key, "-")
                rows += f"<div class='summary-row'><div class='summary-key'>{FIELD_LABELS.get(key, key)}</div><div class='summary-value'>{value}</div></div>"
            with st.expander(group_name, expanded=(group_name == "Lifestyle")):
                st.markdown(f"<div class='summary-compact'>{rows}</div>", unsafe_allow_html=True)

    if st.button("Check Again", use_container_width=True):
        st.session_state.page = "form"
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
    init_session_state()
    inject_styles()

    if not ensure_gemini_api_key():
        st.warning("GEMINI_API_KEY not found in environment or .env. AI recommendation may be unavailable.")

    if st.session_state.page == "form":
        render_form()
    elif st.session_state.page == "loading":
        render_loading()
    else:
        render_result()


if __name__ == "__main__":
    main()
