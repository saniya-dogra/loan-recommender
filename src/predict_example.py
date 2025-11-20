import requests
import json
import numpy as np

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 30,
    "income": 60000,
    "credit_score": 720,
    "employment_type": "salaried",
    "existing_loan": 0
}

response = requests.post(url, json=data)

try:
    # Convert JSON response to Python dict
    res = response.json()

    # Convert any numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_response = {k: convert_numpy(v) for k, v in res.items()}
    print("Response:", clean_response)

except Exception as e:
    print("Error:", str(e))
