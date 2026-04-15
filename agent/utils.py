def extract_factors(data):
    factors = []

    # Glucose
    if data.get("Glucose", 0) >= 140:
        factors.append("High Glucose Level")
    elif data.get("Glucose", 0) >= 110:
        factors.append("Borderline Glucose Level")

    # BMI
    if data.get("BMI", 0) >= 30:
        factors.append("Obesity (High BMI)")
    elif data.get("BMI", 0) >= 25:
        factors.append("Overweight")

    # Blood Pressure
    if data.get("HighBP", 0) == 1:
        factors.append("High Blood Pressure")

    # Cholesterol
    if data.get("HighChol", 0) == 1:
        factors.append("High Cholesterol")

    # Lifestyle
    if data.get("Smoker", 0) == 1:
        factors.append("Smoking")

    if data.get("PhysActivity", 0) == 0:
        factors.append("Low Physical Activity")

    if data.get("Fruits", 0) == 0:
        factors.append("Low Fruit Intake")

    if data.get("Veggies", 0) == 0:
        factors.append("Low Vegetable Intake")

    if data.get("HvyAlcoholConsump", 0) == 1:
        factors.append("High Alcohol Consumption")

    # Age: handle both encoded categories (1-13) and real age in years.
    age_value = data.get("Age", 0)
    try:
        age_num = float(age_value)
    except (TypeError, ValueError):
        age_num = 0

    # Encoded category (1-13): 8+ roughly maps to 55+ years.
    if 0 < age_num <= 13 and age_num >= 8:
        factors.append("Age-related Risk")

    # Raw age in years.
    if age_num > 13 and age_num >= 55:
        factors.append("Age-related Risk")

    # Fallback
    if not factors:
        factors.append("No significant risk factors detected")

    return factors
