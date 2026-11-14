import os
import pickle
import warnings

import pandas as pd
from flask import Flask, render_template, request

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knnmodel.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Column order must match the order used while training the model.
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# Label-encoded categories (alphabetical order, as produced by LabelEncoder).
SEX = {"F": 0, "M": 1}
CHEST_PAIN = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
RESTING_ECG = {"LVH": 0, "Normal": 1, "ST": 2}
EXERCISE_ANGINA = {"N": 0, "Y": 1}
ST_SLOPE = {"Down": 0, "Flat": 1, "Up": 2}

# Allowed ranges for the numeric inputs: (min, max, label)
RANGES = {
    "age": (1, 120, "Age"),
    "resting_bp": (50, 250, "Resting blood pressure"),
    "cholesterol": (50, 700, "Cholesterol"),
    "max_hr": (50, 250, "Maximum heart rate"),
    "oldpeak": (-3, 7, "ST depression"),
}


def parse_form(form):
    """Validate the submitted form and return (row_dict, errors)."""
    errors, values = [], {}

    for key, (low, high, label) in RANGES.items():
        raw = form.get(key, "").strip()
        try:
            number = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not low <= number <= high:
            errors.append(f"{label} must be between {low} and {high}.")
            continue
        values[key] = number

    try:
        values["sex"] = SEX[form.get("sex", "")]
        values["chest_pain"] = CHEST_PAIN[form.get("chest_pain", "")]
        values["fasting_bs"] = 1 if form.get("fasting_bs") == "1" else 0
        values["resting_ecg"] = RESTING_ECG[form.get("resting_ecg", "")]
        values["exercise_angina"] = EXERCISE_ANGINA[form.get("exercise_angina", "")]
        values["st_slope"] = ST_SLOPE[form.get("st_slope", "")]
    except KeyError:
        errors.append("Please choose a value for every option.")

    if errors:
        return None, errors

    row = {
        "Age": values["age"],
        "Sex": values["sex"],
        "ChestPainType": values["chest_pain"],
        "RestingBP": values["resting_bp"],
        "Cholesterol": values["cholesterol"],
        "FastingBS": values["fasting_bs"],
        "RestingECG": values["resting_ecg"],
        "MaxHR": values["max_hr"],
        "ExerciseAngina": values["exercise_angina"],
        "Oldpeak": values["oldpeak"],
        "ST_Slope": values["st_slope"],
    }
    return row, []


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors = None, []
    form_data = request.form if request.method == "POST" else {}

    if request.method == "POST":
        row, errors = parse_form(request.form)
        if row:
            frame = pd.DataFrame([row], columns=FEATURES)
            prediction = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][1]) * 100
            result = {
                "high_risk": prediction == 1,
                "probability": round(probability),
                "neighbors": model.n_neighbors,
                "agree": round(probability * model.n_neighbors / 100),
            }

    return render_template("index.html", result=result, errors=errors, f=form_data)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
