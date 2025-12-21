import os
import json
import tempfile
from datetime import datetime

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.ml.model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
)

from src.monitoring.es_logger import (
    log_to_elasticsearch,
    log_system_metrics,   # ⭐ ADD
)

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Churn Prediction API")

# =====================================================
# PATHS
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
# LOAD ARTEFACTS
# =====================================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
enc_state = joblib.load(ENC_STATE_PATH)
enc_area = joblib.load(ENC_AREA_PATH)

# =====================================================
# SCHEMAS
# =====================================================
class HyperParams(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float


class TrainAllResponse(BaseModel):
    accuracy: float
    roc_auc: float
    message: str

# =====================================================
# FEATURE LIST
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

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"message": "Churn Prediction API running ✔"}

# =====================================================
# TRAIN ALL
# =====================================================
@app.post("/train-all", response_model=TrainAllResponse)
def train_all():
    global model, scaler, enc_state, enc_area

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Churn_API_Full_Pipeline")

    log_system_metrics(stage="before_training")  # ⭐ ADD

    with mlflow.start_run(run_name="API_Train_All") as run:

        X_raw = pd.read_csv(TRAIN_PATH)
        y_raw = pd.read_csv(TRAIN_PATH)

        X_train, X_test, y_train, y_test, enc_state, enc_area = prepare_data(
            X_raw, y_raw
        )

        model, scaler, params = train_model(X_train, y_train)
        mlflow.log_params(params)

        metrics = evaluate_model(model, scaler, X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, name="model")

        log_to_elasticsearch(
            event_type="train",
            payload={
                "run_id": run.info.run_id,
                "metrics": metrics,
                "params": params,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            enc_state_path = os.path.join(tmpdir, "encoder_state.pkl")
            enc_area_path = os.path.join(tmpdir, "encoder_area.pkl")
            features_path = os.path.join(tmpdir, "features.json")

            joblib.dump(scaler, scaler_path)
            joblib.dump(enc_state, enc_state_path)
            joblib.dump(enc_area, enc_area_path)

            with open(features_path, "w") as f:
                json.dump(XGB_IMPORTANCE_COLS, f, indent=2)

            mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
            mlflow.log_artifact(enc_state_path, artifact_path="preprocessing")
            mlflow.log_artifact(enc_area_path, artifact_path="preprocessing")
            mlflow.log_artifact(features_path, artifact_path="metadata")

        save_model(
            model,
            scaler,
            enc_state,
            enc_area,
            prefix=os.path.join(MODELS_DIR, "churn"),
        )

    log_system_metrics(stage="after_training")  # ⭐ ADD

    return {
        "message": "Full training pipeline executed successfully",
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
    }

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict")
def predict(features: dict):
    log_system_metrics(stage="prediction")  # ⭐ ADD

    df = pd.DataFrame([features])
    df = df[XGB_IMPORTANCE_COLS]

    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    log_to_elasticsearch(
        event_type="prediction",
        payload={
            "prediction": int(pred),
            "churn_probability": float(prob),
        },
    )

    return {
        "prediction": int(pred),
        "churn_probability": float(prob),
    }

# =====================================================
# RETRAIN
# =====================================================
@app.post("/retrain")
def retrain(hyperparams: HyperParams):
    global model, scaler, enc_state, enc_area

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Churn_Retrain_API")

    log_system_metrics(stage="before_retrain")  # ⭐ ADD

    with mlflow.start_run(run_name="API_Retrain") as run:

        mlflow.log_params(hyperparams.dict())

        df = pd.read_csv(TRAIN_PATH)

        df["Total charge"] = (
            df["Total day charge"]
            + df["Total eve charge"]
            + df["Total night charge"]
            + df["Total intl charge"]
        )
        df["CScalls Rate"] = df["Customer service calls"] / (
            df["Total day calls"] + 1
        )

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

        model = XGBClassifier(**hyperparams.dict(), eval_metric="logloss")
        model.fit(X_scaled, y)

        mlflow.sklearn.log_model(model, name="model")

        log_to_elasticsearch(
            event_type="retrain",
            payload={
                "run_id": run.info.run_id,
                "hyperparameters": hyperparams.dict(),
            },
        )

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(enc_state, ENC_STATE_PATH)
        joblib.dump(enc_area, ENC_AREA_PATH)

    log_system_metrics(stage="after_retrain")  # ⭐ ADD

    return {
        "status": "success",
        "message": "Model retrained and logged in MLflow ✔",
        "used_hyperparameters": hyperparams,
    }
