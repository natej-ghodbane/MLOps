from fastapi import FastAPI
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# ============================
# Load artefacts on server start
# ============================
model, scaler, enc_state, enc_area = (
    joblib.load("churn_model.pkl"),
    joblib.load("churn_scaler.pkl"),
    joblib.load("churn_encoder_state.pkl"),
    joblib.load("churn_encoder_area.pkl"),
)

TRAIN_PATH = "churn-bigml-80.csv"

class HyperParams(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float

# ============================
# Features used for prediction
# ============================
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


# ======================================================
#                 PREDICT ENDPOINT
# ======================================================
@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])   
    df = df[XGB_IMPORTANCE_COLS]  # enforce order
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }


# ======================================================
#                 RETRAIN ENDPOINT
# ======================================================
@app.post("/retrain")
def retrain(hyperparams: HyperParams):
    global model, scaler, enc_state, enc_area

    # Load training file
    df = pd.read_csv(TRAIN_PATH)

    # ======================================================
    # 0. Recompute engineered features (must match prepare_data)
    # ======================================================

    # Total charge (sum of all charges)
    df["Total charge"] = (
        df["Total day charge"]
        + df["Total eve charge"]
        + df["Total night charge"]
        + df["Total intl charge"]
    )

    # Customer service calls rate
    df["CScalls Rate"] = df["Customer service calls"] / (df["Total day calls"] + 1)

    # "International plan" → yes/no → 1/0
    df["International plan"] = (
        df["International plan"]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
    )

    # ======================================================
    # 1. Recreate & fit encoders (State, Area Code)
    # ======================================================
    enc_state = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc_area = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # State Encoding
    state_encoded = enc_state.fit_transform(df[["State"]])
    state_cols = enc_state.get_feature_names_out(["State"])
    df[state_cols] = state_encoded

    # Area Code Encoding
    area_encoded = enc_area.fit_transform(df[["Area code"]])
    area_cols = enc_area.get_feature_names_out(["Area code"])
    df[area_cols] = area_encoded

    # ======================================================
    # 2. Select X & y
    # ======================================================
    X = df[XGB_IMPORTANCE_COLS]
    y = df["Churn"].astype(int)

    # ======================================================
    # 3. Scale features
    # ======================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================================================
    # 4. Train new XGB model with user hyperparameters
    # ======================================================
    new_model = XGBClassifier(
        **hyperparams.dict(),
        eval_metric="logloss"
    )
    new_model.fit(X_scaled, y)

    # ======================================================
    # 5. Save artefacts
    # ======================================================
    joblib.dump(new_model, "churn_model.pkl")
    joblib.dump(scaler, "churn_scaler.pkl")
    joblib.dump(enc_state, "churn_encoder_state.pkl")
    joblib.dump(enc_area, "churn_encoder_area.pkl")

    # Hot reload
    model = new_model

    return {
        "status": "success",
        "message": "Model retrained successfully ✔",
        "used_hyperparameters": hyperparams,
    }
