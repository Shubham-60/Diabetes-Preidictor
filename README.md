<p align="center">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,50:2563eb,100:7c3aed&height=220&section=header&text=Diabetes%20Risk%20Prediction&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Machine%20Learning%20%7C%20Healthcare%20Screening&descAlignY=58" alt="Diabetes Risk Prediction banner" width="100%">
</p>

<p align="center">
    <em>Predicting diabetes risk from lifestyle and clinical indicators for earlier intervention.</em>
</p>

<p align="center">
    <a href="https://diabetespreidictor.streamlit.app/"><img alt="Streamlit App" src="https://img.shields.io/badge/Live_App-Streamlit-red?logo=streamlit&logoColor=white"></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
    <img alt="Scikit-learn" src="https://img.shields.io/badge/ML-Scikit--learn-f7931e?logo=scikitlearn&logoColor=white">
    <img alt="ANN" src="https://img.shields.io/badge/ANN-MLPClassifier-2563eb">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## ✨ Overview

This project builds and compares multiple machine learning models to predict diabetes risk, then deploys the best-performing model in an interactive Streamlit app.

- **Goal:** Early diabetes risk prediction for preventive action
- **Focus:** Minimize false negatives in healthcare screening
- **Scope:** Model comparison, tuning, threshold optimization, and deployment

## 🧰 Tech Stack

| Area | Tools |
| :--- | :--- |
| Models | Logistic Regression, Random Forest, XGBoost, ANN (MLPClassifier) |
| ML / Data | Scikit-learn, XGBoost, Imbalanced-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Orchestration | LangGraph |
| Deployment | Streamlit |
| Dataset | BRFSS 2015 (Kaggle) |

## 📊 Dataset

| Attribute | Value |
| :--- | :--- |
| Source | [BRFSS 2015 Diabetes Health Indicators (Kaggle)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| Records | ~253,680 |
| Features | 21 |
| Target | `Diabetes_binary` (0 = No, 1 = Yes) |

## 🏗️ Prediction Flow

```mermaid
flowchart TD
        UI["User Input Form"] --> ML["Pretrained Logistic Regression"]
        ML --> SCORE["Risk Probability + Threshold"]
        SCORE --> FACTORS["Risk Factor Extraction"]
        FACTORS --> DOCTOR["Specialist Recommendation"]
        DOCTOR --> RETRIEVE["FAISS Guideline Retrieval"]
        RETRIEVE --> RERANK["CrossEncoder Reranking"]
        RERANK --> LLM["Gemini Structured JSON Output"]
        LLM --> OUT["Streamlit Results View"]
```

> If Mermaid does not render in your Markdown viewer, open this README on GitHub.

## 🚀 Milestones

- Benchmarked models: Logistic Regression, Random Forest, XGBoost, ANN
- Selected final deployment model: **Logistic Regression**

## 🔁 Reproducibility

The deployed app uses the shipped pretrained artifacts in [`models/`](./models):

- `lr_model.pkl`
- `scaler.pkl`
- `model_metadata.json`

Run the app:

```bash
./.venv/bin/streamlit run app/streamlit_app.py
```

The Streamlit app executes a LangGraph workflow at runtime and relies on the pretrained logistic-regression artifacts already stored in the repo.

## 🧠 Agentic Workflow

```mermaid
stateDiagram-v2
    [*] --> Input
    Input --> Predict
    Predict --> ExtractFactors
    ExtractFactors --> RecommendSpecialists
    RecommendSpecialists --> RetrieveGuidelines
    RetrieveGuidelines --> RerankContext
    RerankContext --> GenerateReport
    GenerateReport --> Display
    Display --> [*]
```

## 🗂️ Runtime State

```mermaid
erDiagram
    USER_INPUT ||--|| MODEL_FEATURES : becomes
    MODEL_FEATURES ||--|| GRAPH_STATE : enters
    GRAPH_STATE ||--o{ RETRIEVED_PASSAGE : contains
    GRAPH_STATE ||--o{ RERANKED_PASSAGE : ranks
    GRAPH_STATE ||--o{ SPECIALIST : suggests
    GRAPH_STATE ||--|| STRUCTURED_REPORT : produces
```

## 🔗 Quick Links

- Live app: [Diabetes Risk Predictor](https://diabetespreidictor.streamlit.app/)

## 👥 Team

- [Shubham Aggarwal](https://github.com/Shubham-60)
- [Atharva Sharma](https://github.com/alpha-sml)
- [Bhavya Punj](https://github.com/Rravya14)
