import json
import joblib
import pandas as pd
from .config import MODEL_DIR, METADATA_PATH, FEATURE_COLUMNS

class FraudPredictor:
    def __init__(self):
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Trained model metadata not found. Run "
                "'python -m backend.app.model_training' first."
            )
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        model_path = MODEL_DIR / metadata["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        self.model_name = metadata["best_model"]

    def predict(self, transaction: dict) -> dict:
        missing = set(FEATURE_COLUMNS) - set(transaction)
        extra = set(transaction) - set(FEATURE_COLUMNS)
        if missing or extra:
            raise ValueError(f"Invalid features. Missing={sorted(missing)}, extra={sorted(extra)}")
        row = pd.DataFrame([[transaction[c] for c in FEATURE_COLUMNS]],
                           columns=FEATURE_COLUMNS)
        if row.isna().any().any():
            raise ValueError("Transaction contains missing values.")
        prediction = int(self.model.predict(row)[0])
        probability = float(self.model.predict_proba(row)[0][1])
        risk = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.30 else "LOW"
        return {
            "prediction": prediction,
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": probability,
            "risk_level": risk,
            "model": self.model_name,
        }
