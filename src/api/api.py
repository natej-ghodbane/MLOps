import os
from fastapi import FastAPI
import joblib
import pandas as pd
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# =====================================================
# Resolve project absolute paths dynamically
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "churn-bigml-80.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "churn_scaler.pkl")
ENC_STATE_PATH = os.path.join(MODELS_DIR, "churn_encoder_state.pkl")
ENC_AREA_PATH = os.path.join(MODELS_DIR, "churn_encoder_area.pkl")

# =====================================================
# Load artefacts on server startup
# =====================================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
enc_state = joblib.load(ENC_STATE_PATH)
enc_area = joblib.load(ENC_AREA_PATH)


# =====================================================
# Input schema for retraining
# =====================================================
class HyperParams(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float


# =====================================================
# Columns required for prediction
# =====================================================
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


# =====================================================
# PREDICT ENDPOINT
# =====================================================
@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    df = df[XGB_IMPORTANCE_COLS]  # enforce order

    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(pred),
        "churn_probability": float(prob),
    }


# =====================================================
# RETRAIN ENDPOINT
# =====================================================
@app.post("/retrain")
def retrain(hyperparams: HyperParams):
    global model, scaler, enc_state, enc_area

    with mlflow.start_run(run_name="API_Retrain"):

        mlflow.log_params(hyperparams.dict())

        df = pd.read_csv(TRAIN_PATH)

        df["Total charge"] = (
            df["Total day charge"]
            + df["Total eve charge"]
            + df["Total night charge"]
            + df["Total intl charge"]
        )
        df["CScalls Rate"] = df["Customer service calls"] / (df["Total day calls"] + 1)

        df["International plan"] = (
            df["International plan"]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
            .fillna(0)
        )

        enc_state = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_area = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        df_state = enc_state.fit_transform(df[["State"]])
        df_area = enc_area.fit_transform(df[["Area code"]])

        df[enc_state.get_feature_names_out(["State"])] = df_state
        df[enc_area.get_feature_names_out(["Area code"])] = df_area

        X = df[XGB_IMPORTANCE_COLS]
        y = df["Churn"].astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        new_model = XGBClassifier(
            **hyperparams.dict(),
            eval_metric="logloss"
        )
        new_model.fit(X_scaled, y)

        mlflow.sklearn.log_model(new_model, "model")

        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(enc_state, ENC_STATE_PATH)
        joblib.dump(enc_area, ENC_AREA_PATH)

        model = new_model

    return {
        "status": "success",
        "message": "Model retrained and logged in MLflow ✔",
        "used_hyperparameters": hyperparams,
    }
