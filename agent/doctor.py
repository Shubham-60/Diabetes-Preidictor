from __future__ import annotations

from collections.abc import Mapping


def recommend_department(data: Mapping[str, object], prob: float) -> list[str]:
    departments: list[str] = []

    def add(name: str) -> None:
        if name not in departments:
            departments.append(name)

    if prob > 0.6:
        add("Endocrinologist")

    if int(data.get("HighBP", 0)) == 1:
        add("Cardiologist")

    if float(data.get("BMI", 0)) > 30:
        add("Nutritionist / Dietitian")

    if int(data.get("Age", 0)) > 6:
        add("General Physician")

    if int(data.get("Smoker", 0)) == 1 or int(data.get("PhysActivity", 1)) == 0:
        add("Lifestyle Medicine Specialist")

    if not departments:
        add("General Physician")

    return departments
