import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(r"C:\Users\HP\Downloads\customer_churn_ml\data\telco_churn.csv")
    return df

def preprocess_data(df):
    df = df.copy()

    # Drop customerID if exists
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric if needed
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler