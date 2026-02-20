# End-to-End Customer Churn Prediction
Production-style machine learning system to predict telecom customer churn using an end-to-end deployable pipeline.
## Problem Statement
Telecom companies face significant revenue loss due to customer churn.  
This project builds a production-style machine learning system to predict customer churn risk and enable proactive retention strategies.

---

## Project Highlights

- End-to-end ML pipeline
- Class imbalance handling
- Cross-validation and ROC-AUC evaluation
- Feature engineering and preprocessing
- Model persistence with joblib
- REST API using Flask
- Interactive UI using Streamlit
- Modular production-style project structure

---

## Tech Stack

**Languages & Libraries**

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Flask
- Streamlit
- joblib

---

## Project Architecture

```
User → Streamlit UI → Flask API → ML Model → Prediction
```

---

## Model Performance

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.82 |

*(values may vary slightly depending on run)*

---

## How to Run Locally

### 1. Clone repo

```
git clone <your-repo-link>
cd customer-churn-ml
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train model
Dataset not included due to size; place telco_churn.csv in data/ before training.
```
python -m src.train_model
```

### 4. Run Flask API

```
python -m app.flask_app
```

### 5. Run Streamlit UI

```
streamlit run app/streamlit_app.py
```

---

## Sample Prediction

Input:

```
tenure=12, MonthlyCharges=70, TotalCharges=1000
```

Output:

```
Churn Probability: 0.20  
Prediction: No
```

---

## Future Improvements

- Threshold tuning based on business cost
- XGBoost / LightGBM comparison
- Docker containerization
- Full-feature input UI
- Model monitoring

---

## Author

**Adil O**

- LinkedIn: https://www.linkedin.com/in/adil-ovungal/
- GitHub: https://github.com/adilovungal

