import os
import json
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

from agent.langgraph_flow import build_graph
from agent.workflow import run_patient_workflow

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
                --bg-1: #f0f4ff;
                --bg-2: #e8f8f4;
                --bg-3: #fdf4ff;
                --glass: rgba(255, 255, 255, 0.65);
                --glass-strong: rgba(255, 255, 255, 0.8);
                --text-main: #0f172a;
                --text-muted: #475569;
                --soft-border: rgba(255, 255, 255, 0.55);
                --accent-a: #00c9a7;
                --accent-b: #7c3aed;
                --accent-pink: #ec4899;
                --accent-orange: #f59e0b;
                --risk-low: #14b8a6;
                --risk-med: #f59e0b;
                --risk-high: #f43f5e;
            }

            html,
            body,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                color-scheme: light !important;
            }

            html {
                scroll-behavior: smooth;
            }

            .stApp *,
            .stApp *::before,
            .stApp *::after {
                box-sizing: border-box;
            }

            .stApp {
                font-family: 'DM Sans', sans-serif;
                color: var(--text-main);
                background: linear-gradient(140deg, var(--bg-1), var(--bg-2), var(--bg-3));
                background-size: 220% 220%;
                animation: meshFlow 24s ease-in-out infinite;
                min-height: 100vh;
                position: relative;
            }

            .stApp::before,
            .stApp::after {
                content: "";
                position: fixed;
                border-radius: 50%;
                filter: blur(75px);
                opacity: 0.52;
                pointer-events: none;
                z-index: 0;
                animation: blobFloat 18s ease-in-out infinite;
            }

            .stApp::before {
                width: 310px;
                height: 310px;
                background: radial-gradient(circle at 35% 35%, rgba(0, 201, 167, 0.32), rgba(124, 58, 237, 0.1));
                top: 8%;
                left: -70px;
            }

            .stApp::after {
                width: 350px;
                height: 350px;
                background: radial-gradient(circle at 65% 40%, rgba(236, 72, 153, 0.24), rgba(14, 165, 233, 0.12));
                bottom: -110px;
                right: -90px;
                animation-delay: -9s;
            }

            .block-container {
                max-width: 1020px;
                padding-top: 2.8rem;
                padding-bottom: 2.2rem;
                position: relative;
                z-index: 1;
            }

            [data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0.35);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.45);
            }

            .hero-shell {
                background: var(--glass);
                border: 1px solid var(--soft-border);
                border-radius: 22px;
                padding: 1.2rem 1.2rem;
                margin-bottom: 1.1rem;
                backdrop-filter: blur(16px);
                box-shadow: 0 18px 35px rgba(15, 23, 42, 0.07);
                animation: fadeSlideUp 620ms ease both;
            }

            .hero-chip {
                display: inline-block;
                background: linear-gradient(120deg, rgba(0, 201, 167, 0.16), rgba(124, 58, 237, 0.16));
                color: #334155;
                border: 1px solid rgba(255, 255, 255, 0.65);
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                padding: 0.22rem 0.62rem;
                margin-bottom: 0.45rem;
            }

            .hero-title {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 2.1rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.25rem;
                letter-spacing: -0.01em;
            }

            .hero-subtitle {
                font-size: 0.96rem;
                color: var(--text-muted);
                margin-bottom: 0.2rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 18px;
                border: 1px solid var(--soft-border);
                box-shadow: 0 16px 30px rgba(15, 23, 42, 0.06);
                background: var(--glass);
                backdrop-filter: blur(16px);
                padding: 1rem 1.1rem;
                animation: fadeSlideUp 620ms ease both;
                overflow: hidden;
            }

            [data-testid="stVerticalBlockBorderWrapper"]::before {
                content: "";
                position: absolute;
                top: 0;
                left: -110%;
                width: 120%;
                height: 100%;
                background: linear-gradient(105deg, transparent, rgba(255, 255, 255, 0.38), transparent);
                animation: sheen 8s ease-in-out infinite;
                pointer-events: none;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2) { animation-delay: 80ms; }
            [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3) { animation-delay: 140ms; }
            [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(4) { animation-delay: 200ms; }
            [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(5) { animation-delay: 260ms; }

            .section-shell {
                display: flex;
                gap: 0.75rem;
                align-items: flex-start;
                padding: 0.78rem 0.9rem;
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.6);
                background: rgba(255, 255, 255, 0.55);
                margin-bottom: 0.65rem;
                animation: fadeSlideUp 520ms ease both;
            }

            .section-accent {
                width: 6px;
                min-height: 44px;
                border-radius: 999px;
                background: linear-gradient(180deg, var(--accent-a), var(--accent-b));
                box-shadow: 0 10px 16px rgba(0, 201, 167, 0.22);
            }

            .section-accent.teal { background: linear-gradient(180deg, #00c9a7, #2dd4bf); }
            .section-accent.purple { background: linear-gradient(180deg, #7c3aed, #a855f7); }
            .section-accent.pink { background: linear-gradient(180deg, #ec4899, #f472b6); }
            .section-accent.orange { background: linear-gradient(180deg, #f59e0b, #fb923c); }

            .section-title {
                font-size: 1.1rem;
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 0.4rem;
            }

            .section-caption {
                color: var(--text-muted);
                margin-bottom: 0.8rem;
                font-size: 0.9rem;
            }

            hr.section-divider {
                border: none;
                border-top: 1px solid rgba(148, 163, 184, 0.22);
                margin: 0.2rem 0 1rem 0;
            }

            .helper-text {
                color: var(--text-muted);
                font-size: 0.8rem;
                margin-top: -0.35rem;
                margin-bottom: 0.7rem;
            }

            .score-grid {
                display: grid;
                grid-template-columns: minmax(210px, 250px) minmax(0, 1fr);
                gap: 1.1rem;
                align-items: center;
                padding: 0.35rem 0.2rem 0.2rem;
            }

            .donut-wrap {
                width: 190px;
                height: 190px;
                position: relative;
                display: grid;
                place-items: center;
                justify-self: center;
                flex-shrink: 0;
            }

            .donut-svg {
                width: 190px;
                height: 190px;
                transform: rotate(-90deg);
            }

            .donut-center {
                position: absolute;
                width: 126px;
                height: 126px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.72);
                box-shadow: inset 0 0 16px rgba(148, 163, 184, 0.22);
                display: grid;
                place-items: center;
            }

            .score-center {
                position: relative;
                text-align: center;
                z-index: 2;
            }

            .score-value {
                font-size: 2.1rem;
                font-weight: 760;
                color: var(--text-main);
                margin-bottom: 0.1rem;
            }

            .score-label {
                font-size: 0.85rem;
                color: var(--text-muted);
            }

            .score-metrics {
                min-width: 0;
            }

            .metric-row {
                display: grid;
                grid-template-columns: repeat(2, minmax(140px, 1fr));
                gap: 0.75rem;
                margin-bottom: 0.6rem;
            }

            div[data-testid="stToggle"] div[role="switch"] {
                width: 48px;
                height: 26px;
                padding: 2px;
                background: linear-gradient(120deg, rgba(148, 163, 184, 0.98), rgba(100, 116, 139, 0.98)) !important;
                background-color: rgba(100, 116, 139, 0.98) !important;
                border: 1px solid rgba(30, 41, 59, 0.62) !important;
                border-radius: 999px;
                box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.16), inset 0 0 0 1px rgba(51, 65, 85, 0.36), 0 4px 10px rgba(15, 23, 42, 0.16) !important;
                transition: all 260ms cubic-bezier(0.22, 1, 0.36, 1);
            }

            div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] {
                background: linear-gradient(120deg, rgba(148, 163, 184, 0.98), rgba(100, 116, 139, 0.98)) !important;
                background-color: rgba(100, 116, 139, 0.98) !important;
                border: 1px solid rgba(30, 41, 59, 0.62) !important;
                opacity: 1 !important;
            }

            div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
                background: linear-gradient(120deg, var(--accent-a), var(--accent-b)) !important;
                background-color: var(--accent-b) !important;
                box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 10px 22px rgba(0, 201, 167, 0.18), 0 10px 22px rgba(124, 58, 237, 0.24) !important;
            }

            div[data-testid="stToggle"] div[role="switch"] > div {
                width: 22px;
                height: 22px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 1) !important;
                border: 2px solid rgba(30, 41, 59, 0.7) !important;
                box-shadow: 0 5px 12px rgba(15, 23, 42, 0.24) !important;
                transition: transform 260ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 220ms ease, background 220ms ease;
            }

            div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] > div {
                background: rgba(255, 255, 255, 1) !important;
                border: 2px solid rgba(30, 41, 59, 0.7) !important;
                opacity: 1 !important;
            }

            div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] > div {
                transform: translateX(22px);
                box-shadow: 0 10px 20px rgba(71, 85, 105, 0.28);
            }

            .metric-card {
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.62);
                background: rgba(255, 255, 255, 0.7);
                padding: 0.65rem 0.78rem;
                box-shadow: 0 12px 22px rgba(15, 23, 42, 0.06);
                animation: fadeSlideUp 540ms ease both;
            }

            .metric-card.low {
                background: linear-gradient(120deg, rgba(20, 184, 166, 0.18), rgba(255, 255, 255, 0.72));
            }

            .metric-card.medium {
                background: linear-gradient(120deg, rgba(245, 158, 11, 0.2), rgba(255, 255, 255, 0.72));
            }

            .metric-card.high {
                background: linear-gradient(120deg, rgba(244, 63, 94, 0.18), rgba(255, 255, 255, 0.72));
            }

            .metric-card.verdict-card.low {
                background: linear-gradient(125deg, rgba(20, 184, 166, 0.34), rgba(45, 212, 191, 0.16) 42%, rgba(255, 255, 255, 0.82));
                border: 1px solid rgba(20, 184, 166, 0.28);
                box-shadow: 0 14px 26px rgba(20, 184, 166, 0.16), 0 8px 16px rgba(15, 23, 42, 0.05);
            }

            .metric-card.verdict-card.high {
                background: linear-gradient(125deg, rgba(244, 63, 94, 0.34), rgba(251, 113, 133, 0.16) 42%, rgba(255, 255, 255, 0.82));
                border: 1px solid rgba(244, 63, 94, 0.28);
                box-shadow: 0 14px 26px rgba(244, 63, 94, 0.16), 0 8px 16px rgba(15, 23, 42, 0.05);
            }

            .metric-title {
                font-size: 0.78rem;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.25rem;
            }

            .metric-value {
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text-main);
            }

            .badge {
                display: inline-block;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 650;
                margin-bottom: 0.4rem;
            }

            .badge-green {
                color: #115e59;
                background: rgba(20, 184, 166, 0.16);
            }

            .badge-red {
                color: #9f1239;
                background: rgba(244, 63, 94, 0.16);
            }

            .assessment {
                font-size: 1.1rem;
                font-weight: 640;
                margin-bottom: 0.2rem;
            }

            .result-note {
                color: var(--text-muted);
                font-size: 0.9rem;
            }

            .disclaimer {
                margin-top: 1.25rem;
                text-align: center;
                font-size: 0.82rem;
                color: var(--text-muted);
                background: rgba(255, 255, 255, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.62);
                border-radius: 12px;
                padding: 0.55rem 0.7rem;
            }

            .loading-chip {
                margin-top: 0.5rem;
                color: #5b21b6;
                font-weight: 600;
                font-size: 0.92rem;
                animation: pulseFade 1.1s ease-in-out infinite;
            }

            .loader-shell {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 0.75rem;
                padding: 2rem 1.5rem;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.62);
                background: rgba(255, 255, 255, 0.62);
                backdrop-filter: blur(12px);
                box-shadow: 0 14px 28px rgba(15, 23, 42, 0.07);
            }

            .loader-orbit {
                width: 110px;
                height: 110px;
                border-radius: 50%;
                background: conic-gradient(
                    from 0deg,
                    rgba(0, 201, 167, 0.15),
                    rgba(124, 58, 237, 0.55),
                    rgba(0, 201, 167, 0.15)
                );
                display: grid;
                place-items: center;
                animation: orbitSpin 1.1s linear infinite;
                position: relative;
            }

            .loader-orbit::before {
                content: "";
                width: 74px;
                height: 74px;
                border-radius: 50%;
                background: #ffffff;
                box-shadow: inset 0 0 18px rgba(148, 163, 184, 0.2);
            }

            .loader-dot {
                position: absolute;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #00c9a7;
                box-shadow: 0 6px 14px rgba(0, 201, 167, 0.35);
                transform: translate(0, -55px);
            }

            .loader-text {
                font-size: 0.95rem;
                color: var(--text-muted);
            }

            .summary-compact {
                margin-top: 0.35rem;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.62);
            }

            .summary-row {
                display: grid;
                grid-template-columns: minmax(180px, 1fr) minmax(120px, 0.8fr);
                gap: 0.8rem;
                padding: 0.55rem 0.8rem;
                align-items: center;
                font-size: 0.9rem;
            }

            .summary-row.compact {
                padding: 0.44rem 0.7rem;
                font-size: 0.86rem;
            }

            .summary-row:nth-child(odd) {
                background: rgba(255, 255, 255, 0.7);
            }

            .summary-row:nth-child(even) {
                background: rgba(241, 245, 249, 0.62);
            }

            .summary-key {
                color: #334155;
                font-weight: 600;
            }

            .summary-value {
                color: #0f172a;
                font-weight: 600;
                text-align: right;
            }

            [data-testid="stExpander"] {
                border: 1px solid rgba(255, 255, 255, 0.62) !important;
                border-radius: 14px !important;
                background: rgba(255, 255, 255, 0.55) !important;
                backdrop-filter: blur(10px);
                margin-bottom: 0.42rem;
            }

            [data-testid="stExpander"] details summary {
                color: #0f172a !important;
                font-weight: 700;
                padding-top: 0.15rem;
                padding-bottom: 0.15rem;
            }

            div[data-baseweb="select"] > div,
            .stNumberInput input,
            .stTextInput input {
                border-radius: 999px !important;
                border: 1px solid rgba(255, 255, 255, 0.7) !important;
                background: rgba(255, 255, 255, 0.82) !important;
                transition: border-color 150ms ease, box-shadow 150ms ease;
            }

            .stNumberInput input,
            .stNumberInput input::placeholder,
            .stTextInput input,
            .stTextInput input::placeholder {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                opacity: 1 !important;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="select"] > div * {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                opacity: 1 !important;
            }

            div[data-baseweb="select"] input,
            div[data-baseweb="select"] input::placeholder,
            div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
            div[data-baseweb="select"] [data-testid="stMarkdownContainer"] * {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                opacity: 1 !important;
            }

            div[data-baseweb="select"] svg {
                color: #0f172a !important;
                fill: #0f172a !important;
                stroke: #0f172a !important;
                opacity: 1 !important;
            }

            div[role="listbox"] {
                background: rgba(255, 255, 255, 0.96) !important;
                border: 1px solid rgba(203, 213, 225, 0.9) !important;
            }

            div[role="option"] {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                opacity: 1 !important;
            }

            div[role="option"][aria-selected="true"] {
                background: rgba(124, 58, 237, 0.14) !important;
            }

            div[data-baseweb="select"] > div:focus-within,
            .stNumberInput input:focus,
            .stTextInput input:focus {
                border-color: rgba(124, 58, 237, 0.45) !important;
                box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.13) !important;
            }

            div[data-baseweb="slider"] div[role="slider"] {
                border: 2px solid #ffffff !important;
                background: linear-gradient(120deg, #00c9a7, #7c3aed) !important;
                box-shadow: 0 8px 16px rgba(0, 201, 167, 0.16), 0 8px 16px rgba(124, 58, 237, 0.2);
                transition: transform 220ms ease, box-shadow 220ms ease, background 220ms ease;
            }

            div[data-baseweb="slider"] div[role="slider"]:hover {
                transform: scale(1.04);
                box-shadow: 0 12px 24px rgba(0, 201, 167, 0.2), 0 12px 24px rgba(124, 58, 237, 0.28);
            }

            div[data-baseweb="slider"] > div > div:first-child {
                background: linear-gradient(120deg, rgba(0, 201, 167, 0.36), rgba(124, 58, 237, 0.34)) !important;
                border-radius: 999px;
                box-shadow: 0 1px 4px rgba(15, 23, 42, 0.1);
                transition: width 240ms ease, background 220ms ease, box-shadow 220ms ease;
            }

            div.stButton > button[kind="primary"] {
                background: linear-gradient(118deg, var(--accent-a), var(--accent-b));
                background-size: 200% 200%;
                color: white;
                border: none;
                border-radius: 999px;
                padding: 0.72rem 1rem;
                font-weight: 700;
                transition: transform 120ms ease-in-out, box-shadow 170ms ease-in-out;
                box-shadow: 0 12px 24px rgba(124, 58, 237, 0.24);
                animation: gradientShift 5.5s ease infinite;
            }

            div.stButton > button[kind="primary"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 16px 30px rgba(124, 58, 237, 0.33);
            }

            div.stButton > button[kind="secondary"] {
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.7);
                color: #334155;
                background: rgba(255, 255, 255, 0.85);
                font-weight: 600;
            }

            label, .stNumberInput label, .stSelectbox label, .stSlider label, .stToggle label {
                color: #0f172a !important;
                font-weight: 600 !important;
            }

            div[data-testid="stToggle"] [data-testid="stWidgetLabel"],
            div[data-testid="stToggle"] [data-testid="stWidgetLabel"] *,
            div[data-testid="stToggle"] [data-testid="stMarkdownContainer"],
            div[data-testid="stToggle"] [data-testid="stMarkdownContainer"] *,
            div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"],
            div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] *,
            div[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"],
            div[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] *,
            div[role="switch"] + div,
            div[role="switch"] + div *,
            div[role="switch"] ~ div,
            div[role="switch"] ~ div * {
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
                opacity: 1 !important;
            }

            .stMarkdown p, .stMarkdown li, .stMarkdown div {
                color: #475569;
            }

            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes meshFlow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes blobFloat {
                0%, 100% { transform: translateY(0) translateX(0) scale(1); }
                50% { transform: translateY(-18px) translateX(10px) scale(1.04); }
            }

            @keyframes pulseFade {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.55; }
            }

            @keyframes fadeSlideUp {
                from {
                    opacity: 0;
                    transform: translateY(16px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes sheen {
                0%, 70%, 100% { left: -110%; }
                84% { left: 120%; }
            }

            @keyframes orbitSpin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation: none !important;
                    transition: none !important;
                }
            }

            @media (max-width: 900px) {
                .metric-row {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-top: 1.4rem;
                    padding-left: 0.85rem;
                    padding-right: 0.85rem;
                }

                .hero-title {
                    font-size: 1.56rem;
                }

                .score-grid {
                    grid-template-columns: 1fr;
                    justify-items: center;
                    gap: 1rem;
                }

                .score-metrics {
                    width: 100%;
                }

                .donut-wrap,
                .donut-svg {
                    width: 168px;
                    height: 168px;
                }

                .donut-center {
                    width: 110px;
                    height: 110px;
                }

                .summary-row {
                    grid-template-columns: 1fr;
                    gap: 0.28rem;
                }

                .summary-value {
                    text-align: left;
                }

                .stApp::after {
                    display: none;
                }
            }
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
    from agent.rag_faiss import load_pdfs

    return load_pdfs()


@st.cache_resource(show_spinner=False)
def build_index_cached(chunks):
    from agent.rag_faiss import create_index

    return create_index(chunks)


@st.cache_resource(show_spinner=False)
def build_workflow_graph():
    return build_graph()


def predict_probability(features: dict) -> float:
    model, scaler = load_model_artifacts()
    row = {feature: features[feature] for feature in FEATURE_ORDER}
    frame = pd.DataFrame([row], columns=FEATURE_ORDER)
    transformed = scaler.transform(frame)
    return max(0.0, min(float(model.predict_proba(transformed)[0][1]), 1.0))


def run_ai_pipeline(raw_inputs: dict, model_features: dict, probability: float):
    from agent.llm import generate_ai_response
    from agent.prompt import build_prompt
    from agent.rag_faiss import chunk_text, search
    from agent.reranker import rerank
    from agent.utils import extract_factors
    from agent.doctor import recommend_department

    texts = load_docs_cached()
    chunks = chunk_text(texts)
    index, chunks = build_index_cached(chunks)
    result = run_patient_workflow(
        raw_inputs=raw_inputs,
        model_features=model_features,
        probability=probability,
        index=index,
        chunks=chunks,
        graph_builder=build_workflow_graph,
        factor_fn=extract_factors,
        doctor_fn=recommend_department,
        search_fn=search,
        rerank_fn=rerank,
        prompt_fn=build_prompt,
        response_fn=generate_ai_response,
    )

    return (
        result["factors"],
        result["specialists"],
        result["ai_response"],
        result["source_context"],
    )


def parse_ai_response(text: str) -> dict:
    fallback = {
        "risk_level": "Unavailable",
        "explanation": text.strip() or "AI explanation unavailable.",
        "recommendations": [],
        "preventive_measures": [],
        "suggested_specialists": [],
        "source_citations": [],
        "disclaimer": "This assessment is a screening aid only and not a medical diagnosis. Please consult a qualified healthcare professional.",
    }

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {
                "risk_level": str(parsed.get("risk_level", fallback["risk_level"])),
                "explanation": str(parsed.get("explanation", fallback["explanation"])),
                "recommendations": list(parsed.get("recommendations", [])),
                "preventive_measures": list(parsed.get("preventive_measures", [])),
                "suggested_specialists": list(parsed.get("suggested_specialists", [])),
                "source_citations": list(parsed.get("source_citations", [])),
                "disclaimer": str(parsed.get("disclaimer", fallback["disclaimer"])),
            }
    except Exception:
        pass

    return fallback


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
        except ImportError as error:
            st.session_state.needs_prediction = False
            st.error("Workflow dependencies are missing. Install the packages from requirements.txt and rerun the app.")
            st.caption(str(error))
            if st.button("Back to Form", use_container_width=True):
                st.session_state.page = "form"
                st.rerun()
            return
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
    score_text = f"{int(round(probability * 100))}%"
    risk_label = "Low Risk" if risk_class == 0 else "High Risk"
    risk_tone = "low" if risk_class == 0 else "high"
    if risk_class == 0:
        score_color = "#0f766e"
        donut_grad_start = "#14b8a6"
        donut_grad_end = "#2dd4bf"
        donut_track = "rgba(20, 184, 166, 0.18)"
    else:
        score_color = "#be123c"
        donut_grad_start = "#f43f5e"
        donut_grad_end = "#fb7185"
        donut_track = "rgba(244, 63, 94, 0.18)"

    features = st.session_state.features
    near_band = 0.05

    def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    def feature_state(score: float) -> tuple[str, str]:
        if score < THRESHOLD - near_band:
            return "low", "Low"
        if score > THRESHOLD + near_band:
            return "high", "High"
        return "medium", "Near"

    high_bp_value = int(features.get("HighBP", 0))
    bmi_value = float(features.get("BMI", 0.0))
    gen_hlth_value = int(features.get("GenHlth", 1))

    factor_cards = [
        {
            "title": "High Blood Pressure",
            "value": "Yes" if high_bp_value == 1 else "No",
            "score": clamp(float(high_bp_value)),
        },
        {
            "title": "BMI",
            "value": f"{bmi_value:.1f}",
            "score": clamp(bmi_value / 60),
        },
        {
            "title": "General Health",
            "value": f"{gen_hlth_value}/5",
            "score": clamp(gen_hlth_value / 5),
        },
    ]

    st.markdown(f"""
    <div class='hero-shell'>
      <div class='hero-chip'>Assessment Summary</div>
      <div class='hero-title'>Your Diabetes Risk Assessment</div>
      <div class='hero-subtitle'>Model output with clinically oriented interpretation</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        circumference = 2 * 3.1416 * 80
        progress = max(0.0, min(1.0, probability))
        dash = round(circumference * progress, 2)
        gap = round(circumference - dash, 2)
        st.markdown(
            f"""
            <div class='score-grid'>
                <div class='donut-wrap'>
                    <svg class='donut-svg' viewBox='0 0 190 190' aria-label='Risk donut chart'>
                        <defs>
                            <linearGradient id='donutGrad' x1='0%' y1='0%' x2='100%' y2='100%'>
                                <stop offset='0%' stop-color='{donut_grad_start}'></stop>
                                <stop offset='100%' stop-color='{donut_grad_end}'></stop>
                            </linearGradient>
                        </defs>
                        <circle cx='95' cy='95' r='80' fill='none' stroke='{donut_track}' stroke-width='16'></circle>
                        <circle cx='95' cy='95' r='80' fill='none' stroke='url(#donutGrad)' stroke-width='16' stroke-linecap='round' stroke-dasharray='{dash} {gap}'></circle>
                    </svg>
                    <div class='donut-center'>
                        <div class='score-center'>
                            <div class='score-value' style='color:{score_color};'>{score_text}</div>
                            <div class='score-label'>Risk Score</div>
                        </div>
                    </div>
                </div>
                <div class='score-metrics'>
                    <div class='metric-row'>
                        <div class='metric-card {risk_tone}'>
                            <div class='metric-title'>Verdict</div>
                            <div class='metric-value'>{risk_label}</div>
                        </div>
                        <div class='metric-card {risk_tone}'>
                            <div class='metric-title'>Risk Tone</div>
                            <div class='metric-value'>{risk_tone.title()}</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        badge_class = "badge-green" if risk_class == 0 else "badge-red"
        result_text = "This classification is based on your submitted health profile features."
        st.markdown(f"<span class='badge {badge_class}'>{risk_label}</span>", unsafe_allow_html=True)
        st.markdown("<div class='assessment'>Risk Classification</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-note'>{result_text}</div>", unsafe_allow_html=True)

        cards_html = ""
        for card in factor_cards:
            tone_class, tone_label = feature_state(card["score"])
            cards_html += (
                f"<div class='metric-card {tone_class}'>"
                f"<div class='metric-title'>{card['title']}</div>"
                f"<div class='metric-value'>{card['value']} • {tone_label}</div>"
                f"</div>"
            )

        st.markdown(
            f"""
            <div class='result-note' style='margin-top:0.75rem;'>Key Correlated Factors</div>
            <div class='metric-row' style='margin-top:0.35rem;'>{cards_html}</div>
            """,
            unsafe_allow_html=True,
        )

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

        parsed = parse_ai_response(st.session_state.ai_response)

        # --- Risk Level ---
        st.markdown("### 🩺 Risk Assessment")
        risk_level = parsed["risk_level"]
        if "low" in risk_level.lower():
            st.success(f"Risk Level: {risk_level}")
        elif "medium" in risk_level.lower():
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.error(f"Risk Level: {risk_level}")

        # --- Explanation ---
        st.markdown("### 📋 Explanation")
        st.write(parsed["explanation"])

        # --- Recommendations ---
        st.markdown("### 💡 Recommendations")
        for item in parsed["recommendations"]:
            st.write(f"• {item}")

        # --- Preventive Measures ---
        st.markdown("### 🛡 Preventive Measures")
        for item in parsed["preventive_measures"]:
            st.write(f"• {item}")

        st.markdown("### 🧭 Suggested Specialists")
        for item in parsed["suggested_specialists"]:
            st.write(f"• {item}")

        st.markdown("### ⚖ Disclaimer")
        st.caption(parsed["disclaimer"])

        # --- Source Context ---
        st.markdown("### 📚 Source")
        combined_sources = parsed["source_citations"] or st.session_state.source_context.split("\n")[:2]
        for line in combined_sources[:3]:
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
