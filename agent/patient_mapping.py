from __future__ import annotations

from typing import Any


_BINARY_FIELD_LABELS = {
    "HighBP": {0: "No high BP", 1: "High BP"},
    "HighChol": {0: "No high cholesterol", 1: "High cholesterol"},
    "CholCheck": {0: "No cholesterol check in 5 years", 1: "Yes cholesterol check in 5 years"},
    "Smoker": {
        0: "No (smoked fewer than 100 cigarettes in lifetime)",
        1: "Yes (smoked at least 100 cigarettes in lifetime)",
    },
    "Stroke": {0: "No", 1: "Yes"},
    "HeartDiseaseorAttack": {
        0: "No coronary heart disease (CHD) or myocardial infarction (MI)",
        1: "Yes coronary heart disease (CHD) or myocardial infarction (MI)",
    },
    "PhysActivity": {0: "No physical activity in past 30 days (excluding job)", 1: "Yes physical activity in past 30 days (excluding job)"},
    "Fruits": {0: "No fruit 1 or more times per day", 1: "Yes fruit 1 or more times per day"},
    "Veggies": {0: "No vegetables 1 or more times per day", 1: "Yes vegetables 1 or more times per day"},
    "HvyAlcoholConsump": {
        0: "No heavy alcohol consumption",
        1: "Yes heavy alcohol consumption (men >=14 drinks/week, women >=7 drinks/week)",
    },
    "AnyHealthcare": {0: "No health care coverage", 1: "Has health care coverage"},
    "NoDocbcCost": {
        0: "No, did not skip doctor due to cost in past 12 months",
        1: "Yes, skipped doctor due to cost in past 12 months",
    },
    "DiffWalk": {0: "No serious difficulty walking/climbing stairs", 1: "Yes serious difficulty walking/climbing stairs"},
}

_GEN_HLTH_MAP = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}


def _to_binary(value: Any) -> int | None:
    if value in (0, 1, "0", "1"):
        return int(value)
    return None


def map_inputs_for_llm(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(raw_inputs)

    for field, value_map in _BINARY_FIELD_LABELS.items():
        binary = _to_binary(raw_inputs.get(field))
        if binary is not None:
            mapped[field] = value_map[binary]

    sex_binary = _to_binary(raw_inputs.get("Sex"))
    if sex_binary is not None:
        mapped["Sex"] = "Male" if sex_binary == 1 else "Female"

    gen_hlth = raw_inputs.get("GenHlth")
    if isinstance(gen_hlth, (int, float)):
        gen_hlth_int = int(gen_hlth)
        if gen_hlth_int in _GEN_HLTH_MAP:
            mapped["GenHlth"] = f"{gen_hlth_int} ({_GEN_HLTH_MAP[gen_hlth_int]})"

    ment_hlth = raw_inputs.get("MentHlth")
    if isinstance(ment_hlth, (int, float)):
        ment_hlth_int = int(ment_hlth)
        mapped["MentHlth"] = f"{ment_hlth_int} day(s) of poor mental health in past 30 days"

    phys_hlth = raw_inputs.get("PhysHlth")
    if isinstance(phys_hlth, (int, float)):
        phys_hlth_int = int(phys_hlth)
        mapped["PhysHlth"] = f"{phys_hlth_int} day(s) physical illness/injury in past 30 days"

    if raw_inputs.get("EducationLabel"):
        mapped["Education"] = raw_inputs["EducationLabel"]
    if raw_inputs.get("IncomeLabel"):
        mapped["Income"] = raw_inputs["IncomeLabel"]

    mapped.pop("EducationLabel", None)
    mapped.pop("IncomeLabel", None)

    return mapped


def build_rag_profile(mapped_inputs: dict[str, Any]) -> str:
    ordered_keys = [
        "Age",
        "Sex",
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
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
        "Education",
        "Income",
    ]

    parts: list[str] = []
    for key in ordered_keys:
        if key in mapped_inputs:
            parts.append(f"{key}: {mapped_inputs[key]}")

    return " | ".join(parts)