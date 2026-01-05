import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd 
from fastapi.middleware.cors import CORSMiddleware

bundle = joblib.load("loan_model_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Loan Prediction API is running."}

@app.post("/predict")
def predict(data: dict):
    X = preprocess_input(data, scaler)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][prediction]

    return {
        "loan_status": "Approved" if prediction == 1 else "Rejected",
        "confidence": round(float(probability) * 100, 2)
    }

def preprocess_input(data: dict, scaler):
    """
    Preprocess frontend input to match training pipeline exactly
    """

    # --- Encode categorical ---
    education_map = {
        "Graduate": 1,
        "Not Graduate": 0
    }

    self_employed_map = {
        "Yes": 1,
        "No": 0
    }

    education = education_map.get(data["education"], 0)
    self_employed = self_employed_map.get(data["self_employed"], 0)

    # --- Base numerical features ---
    income_annum = float(data["annual_income"])
    loan_amount = float(data["loan_amount"])
    loan_term = float(data["loan_term"])
    cibil_score = float(data["cibil_score"])

    residential_assets = float(data["residential_assets_value"])
    commercial_assets = float(data["commerical_assets_value"])
    luxury_assets = float(data["luxury_assets_value"])
    bank_assets = float(data["bank_assets_value"])

    # --- Feature engineering (SAME AS TRAINING) ---
    total_assets_value = (
        residential_assets +
        commercial_assets +
        luxury_assets +
        bank_assets
    )

    loan_to_income_ratio = loan_amount / income_annum if income_annum > 0 else 0

    # --- Create dataframe (order matters) ---
    df = pd.DataFrame([{
        "no_of_dependents": int(data["num_dependents"]),
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets,
        "commercial_assets_value": commercial_assets,
        "luxury_assets_value": luxury_assets,
        "bank_asset_value": bank_assets,
        "total_assets_value": total_assets_value,
        "loan_to_income_ratio": loan_to_income_ratio
    }])

    # --- Columns that were scaled ---
    num_cols_scaled = [
        "income_annum",
        "loan_amount",
        "loan_term",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "total_assets_value",
        "loan_to_income_ratio"
    ]

    # --- Apply SAME scaler ---
    df[num_cols_scaled] = scaler.transform(df[num_cols_scaled])

    return df


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)