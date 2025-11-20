from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("model.pkl")

# Single mapping: bank ID → (bank_name, interest_rate)
bank_mapping = {
    0: ("SBI", 7.9),
    1: ("ICICI", 8.0),
    2: ("HDFC", 8.2),
    3: ("AXIS", 8.3)
}

def to_python(obj):
    """Convert numpy types → python types recursively."""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(i) for i in obj]
    return obj

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Encode employment type if present
        if "employment_type" in df.columns:
            df["employment_type"] = df["employment_type"].astype("category").cat.codes

        # Match model feature order
        feature_names = model.get_booster().feature_names
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_names]

        # Prediction
        pred_id = int(model.predict(df)[0])  # get bank ID
        recommended_bank, interest_rate = bank_mapping.get(pred_id, ("Unknown", 0.0))

        response = {
            "recommended_bank": recommended_bank,
            "interest_rate": float(interest_rate)
        }

        return jsonify(to_python(response))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
