import pandas as pd
import os

def load_and_prepare_data(path=None):
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "..", "data", "customers.csv")

    df = pd.read_csv(path)

    # Encode employment_type as numeric (this is fine)
    df["employment_type"] = df["employment_type"].astype("category")
    df["employment_type"] = df["employment_type"].cat.codes

    # Do NOT encode approved_bank here — leave y as strings
    X = df[[ 
        "age", "income", "credit_score",
        "employment_type",
        "loan_amount", "loan_tenure", "interest_rate"
    ]]
    y = df["approved_bank"]

    return X, y
