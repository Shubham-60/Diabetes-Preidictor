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
    <img alt="TensorFlow" src="https://img.shields.io/badge/DL-TensorFlow-ff6f00?logo=tensorflow&logoColor=white">
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
| Models | Logistic Regression, Random Forest, XGBoost, ANN |
| ML / Data | Scikit-learn, TensorFlow, Imbalanced-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
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
        A[User Input] --> B[Preprocessing]
        B --> C[Feature Scaling]
        C --> D[Logistic Regression]
        D --> E[Probability Score]
        E --> F{Threshold 0.58}
        F -->|High| G[Risk: Likely Diabetic]
        F -->|Low| H[Risk: Low]
        G --> I[Streamlit Output]
        H --> I
```

> If Mermaid does not render in your Markdown viewer, open this README on GitHub.

## 🚀 Milestones

- Benchmarked models: Logistic Regression, Random Forest, XGBoost, ANN
- Selected final deployment model: **Logistic Regression**

## 🔗 Quick Links

- Live app: [Diabetes Risk Predictor](https://diabetespreidictor.streamlit.app/)
- Implementation details: [Final/README.md](Final/README.md)

## 👥 Team

- [Shubham Aggarwal](https://github.com/Shubham-60)
- [Atharva Sharma](https://github.com/alpha-sml)
- [Bhavya Punj](https://github.com/Rravya14)
