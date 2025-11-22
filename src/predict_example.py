import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 30,
    "income": 6000,
    "credit_score": 720,
    "employment_type": "unsalaried",
    "loan_amount": 200000,
    "loan_tenure": 5,
    "interest_rate": 8
}

response = requests.post(url, json=data)
res = response.json()

print("\nTop Bank Recommendations:")
print("---------------------------")

for i, rec in enumerate(res.get("top_recommendations", []), 1):
    bank = rec["bank"]
    prob = rec["probability"] * 100
    print(f"{i}. {bank} → {prob:.1f}%")
