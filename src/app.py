from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model + encoder
model = joblib.load("model.pkl")
le = joblib.load("le.pkl")   # LabelEncoder for bank names

# Employment mapping
employment_mapping = {"salaried": 1, "self-employed": 0}

def to_python(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
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

        # Encode employment type
        df["employment_type"] = df["employment_type"].map(employment_mapping).fillna(0)

        # Ensure column order
        feature_names = model.get_booster().feature_names
        for f in feature_names:
            if f not in df:
                df[f] = 0
        df = df[feature_names]

        probs = model.predict_proba(df)[0]
        top_indices = np.argsort(probs)[::-1][:3]

        top_banks = [
            {"bank": le.inverse_transform([i])[0], "probability": float(probs[i])}
            for i in top_indices
        ]

        return jsonify({"top_recommendations": top_banks})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
