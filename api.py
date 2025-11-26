from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Load all artefacts on server start
model, scaler, enc_state, enc_area = (
    joblib.load("churn_model.pkl"),
    joblib.load("churn_scaler.pkl"),
    joblib.load("churn_encoder_state.pkl"),
    joblib.load("churn_encoder_area.pkl"),
)

# Feature subset used for prediction
XGB_IMPORTANCE_COLS = [
    "Total charge",
    "Area code_415",
    "Area code_408",
    "Customer service calls",
    "Area code_510",
    "Total intl calls",
    "International plan",
    "Number vmail messages",
    "State_SC",
    "State_TX",
    "State_MT",
    "Total intl charge",
    "State_IL",
    "CScalls Rate",
]


@app.get("/")
def root():
    return {"message": "Churn Prediction API running ✔"}


@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])

    # Ensure feature order
    df = df[XGB_IMPORTANCE_COLS]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {"prediction": int(prediction), "churn_probability": float(probability)}