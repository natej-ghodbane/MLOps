import argparse
import os
import joblib
import mlflow
import mlflow.sklearn

from src.ml.model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(MODELS_DIR, "raw_data.pkl")
PREPARED_DATA_PATH = os.path.join(MODELS_DIR, "prepared_data.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")

# ============================================================
# MLflow (USE SAME AS API)
# ============================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Churn_Prediction_CLI")

# ============================================================
# Validation Helper
# ============================================================
def validate_io(X_train, X_test, y_train, y_test):
    if len(X_train) != len(y_train):
        raise ValueError("Train X/y mismatch")
    if len(X_test) != len(y_test):
        raise ValueError("Test X/y mismatch")
    print("IO validation passed!")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["load", "prepare", "train", "evaluate", "all"]
    )
    args = parser.parse_args()

    # ------------------------------
    # LOAD
    # ------------------------------
    if args.action in ("load", "all"):
        print("Loading data...")
        X_raw, y_raw = load_data(
            train_path=os.path.join(DATA_DIR, "churn-bigml-80.csv"),
            test_path=os.path.join(DATA_DIR, "churn-bigml-20.csv"),
        )
        joblib.dump((X_raw, y_raw), RAW_DATA_PATH)
        print("Raw data saved")

        if args.action == "load":
            return


    # ------------------------------
    # PREPARE
    # ------------------------------
    if args.action in ("prepare", "all"):
        print("Preparing data...")
        X_raw, y_raw = joblib.load(RAW_DATA_PATH)

        X_train, X_test, y_train, y_test, enc_state, enc_area = prepare_data(X_raw, y_raw)
        validate_io(X_train, X_test, y_train, y_test)

        joblib.dump((X_train, X_test, y_train, y_test), PREPARED_DATA_PATH)
        joblib.dump((enc_state, enc_area), ENCODERS_PATH)

        print("Prepared data saved")

        if args.action == "prepare":
            return

    # ------------------------------
    # TRAIN
    # ------------------------------
    if args.action in ("train", "all"):
        print("Training model...")

        X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
        enc_state, enc_area = joblib.load(ENCODERS_PATH)

        with mlflow.start_run(run_name="CLI_Training"):
            model, scaler, params = train_model(X_train, y_train)

            # 🔥 LOG PARAMS
            mlflow.log_params(params)

            # 🔥 EVALUATE & LOG METRICS
            metrics = evaluate_model(model, scaler, X_test, y_test)
            mlflow.log_metrics(metrics)

            # 🔥 LOG MODEL
            mlflow.sklearn.log_model(model, name="model")

            save_model(
                model,
                scaler,
                enc_state,
                enc_area,
                prefix=os.path.join(MODELS_DIR, "churn"),
            )

        print("Training completed")

        if args.action == "train":
            return


    # ------------------------------
    # EVALUATE ONLY
    # ------------------------------
    if args.action in ("evaluate", "all"):
        print("Evaluating model...")

        model, scaler, enc_state, enc_area = load_model(
            prefix=os.path.join(MODELS_DIR, "churn")
        )
        X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)

        with mlflow.start_run(run_name="CLI_Evaluation"):
            metrics = evaluate_model(model, scaler, X_test, y_test)
            mlflow.log_metrics(metrics)

        print("Evaluation completed")


if __name__ == "__main__":
    main()
