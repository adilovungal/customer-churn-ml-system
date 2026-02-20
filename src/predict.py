import joblib
import pandas as pd
import os

# Get project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# Load artifacts safely
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

def predict_churn(input_dict):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_dict])

        # One-hot encode
        df = pd.get_dummies(df)

        # Align with training features
        model_features = feature_columns
        df = df.reindex(columns=model_features, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prob = model.predict_proba(df_scaled)[0][1]
        pred = int(prob >= 0.5)

        return {
            "churn_probability": float(prob),
            "prediction": "Yes" if pred == 1 else "No"
        }

    except Exception as e:
        return {"error": str(e)}