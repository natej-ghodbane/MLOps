import argparse
import os
import joblib

from src.ml.model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


# ============================================================
# Directories (adapted to new structure)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")

RAW_DATA_PATH = os.path.join(MODELS_DIR, "raw_data.pkl")
PREPARED_DATA_PATH = os.path.join(MODELS_DIR, "prepared_data.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")


# ============================================================
# Validation Helper
# ============================================================
def validate_io(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None:
        raise ValueError("X_train or X_test is None")

    if y_train is None or y_test is None:
        raise ValueError("y_train or y_test is None")

    if len(X_train) != len(y_train):
        raise ValueError("Training data mismatch")

    if len(X_test) != len(y_test):
        raise ValueError("Test data mismatch")

    print("IO validation passed!")


# ============================================================
# Main CLI Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Churn ML pipeline CLI")
    parser.add_argument(
        "action",
        choices=["load", "prepare", "train", "evaluate", "all"],
        help="Pipeline step to execute.",
    )
    args = parser.parse_args()

    # ------------------------------
    # LOAD STEP
    # ------------------------------
    if args.action in ("load", "all"):
        print("Loading raw CSV files...")

        X_raw, y_raw = load_data(
            train_path=os.path.join(DATA_DIR, "churn-bigml-80.csv"),
            test_path=os.path.join(DATA_DIR, "churn-bigml-20.csv"),
        )

        print("\nRaw data loaded:")
        print("  Train shape:", X_raw.shape)
        print("  Test shape:", y_raw.shape)

        joblib.dump((X_raw, y_raw), RAW_DATA_PATH)
        print(f"Raw samples saved to {RAW_DATA_PATH}")

        if args.action == "load":
            return

    # ------------------------------
    # PREPARE STEP
    # ------------------------------
    if args.action in ("prepare", "all"):
        print("Preparing data…")

        X_raw, y_raw = joblib.load(RAW_DATA_PATH)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            enc_state,
            enc_area,
        ) = prepare_data(X_raw, y_raw)

        validate_io(X_train, X_test, y_train, y_test)

        joblib.dump((X_train, X_test, y_train, y_test), PREPARED_DATA_PATH)
        joblib.dump((enc_state, enc_area), ENCODERS_PATH)

        print(f"Prepared data saved → {PREPARED_DATA_PATH}")
        print(f"Encoders saved → {ENCODERS_PATH}")

        if args.action == "prepare":
            return

    # ------------------------------
    # TRAIN STEP
    # ------------------------------
    if args.action in ("train", "all"):
        print("Training model…")

        X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
        enc_state, enc_area = joblib.load(ENCODERS_PATH)

        model, scaler = train_model(X_train, y_train)
        save_model(model, scaler, enc_state, enc_area, prefix=os.path.join(MODELS_DIR, "churn"))

        print("Train step done!")

        if args.action == "train":
            return

    # ------------------------------
    # EVALUATE STEP
    # ------------------------------
    if args.action in ("evaluate", "all"):
        print("Evaluating model…")

        model, scaler, enc_state, enc_area = load_model(prefix=os.path.join(MODELS_DIR, "churn"))
        X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)

        evaluate_model(model, scaler, X_test, y_test)
        print("Evaluate step done!")


if __name__ == "__main__":
    main()
