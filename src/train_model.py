import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_preprocessing import load_data, preprocess_data, split_and_scale

DATA_PATH = r"C:\Users\HP\Downloads\customer_churn_ml\data\telco_churn.csv"

def train():

    # Load & preprocess
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    feature_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Models
    lr = LogisticRegression(max_iter=5000, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    # Train
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Evaluate LR
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, lr.predict(X_test)))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

    # Evaluate RF
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print("\n=== Random Forest ===")
    print(classification_report(y_test, rf.predict(X_test)))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

    # Cross-validation (LR)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=5000, class_weight='balanced'),
        X, y,
        cv=skf,
        scoring='roc_auc'
    )
    print("\nLR CV ROC-AUC mean:", cv_scores.mean())

    # Ensure models folder exists
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs("models", exist_ok=True)

    # Save model and scaler
    joblib.dump(rf, os.path.join(MODEL_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print("DEBUG: saving feature columns, count =", len(feature_columns))
    print("\n✅ Model saved.")

if __name__ == "__main__":
    train()