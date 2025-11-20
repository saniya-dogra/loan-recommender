import joblib
from xgboost import XGBClassifier
from data_prep import load_and_prepare_data
from sklearn.preprocessing import LabelEncoder

def train():
    X, y = load_and_prepare_data()

    # Encode bank names
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save bank number → bank name
    bank_mapping = dict(enumerate(le.classes_))
    joblib.dump(bank_mapping, "bank_mapping.pkl")
    print("bank_mapping.pkl saved")

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X, y_encoded)

    # Save model
    joblib.dump(model, "model.pkl")
    print("model.pkl saved")

if __name__ == "__main__":
    train()
