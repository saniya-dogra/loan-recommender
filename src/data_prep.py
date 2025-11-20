import pandas as pd
import joblib
import os

def load_and_prepare_data(path=None):

    # Auto-detect correct CSV path
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "..", "data", "customers.csv")

    print("Loading CSV from:", path)

    df = pd.read_csv(path)

    # Encode employment type
    df["employment_type"] = df["employment_type"].astype("category").cat.codes

    # Encode approved_bank
    df["approved_bank"] = df["approved_bank"].astype("category")
    bank_mapping = dict(enumerate(df["approved_bank"].cat.categories))

    # Save mapping
    joblib.dump(bank_mapping, os.path.join(os.path.dirname(path), "..", "bank_mapping.pkl"))

    X = df[[
        "age", "income", "credit_score",
        "employment_type",
        "loan_amount", "loan_tenure", "interest_rate"
    ]]

    y = df["approved_bank"].cat.codes

    return X, y


if __name__ == "__main__":
    X, y = load_and_prepare_data()
    print(X.head())
