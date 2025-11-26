import argparse
import joblib
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


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


def main():
    parser = argparse.ArgumentParser(description="Churn ML pipeline CLI")
    parser.add_argument(
        "action",
        choices=["load", "prepare", "train", "evaluate", "all"],
        help="Pipeline step to execute.",
    )
    args = parser.parse_args()

    # --- LOAD ---
    if args.action in ("load", "all"):
        print("Loading raw CSV files...")
        X_raw, y_raw = load_data()

        print("\nRaw data loaded:")
        print("  Train shape:", X_raw.shape)
        print("  Test shape:", y_raw.shape)

        joblib.dump((X_raw, y_raw), "raw_data.pkl")
        print("Raw data saved to raw_data.pkl")

        if args.action == "load":
            return

    # --- PREPARE ---
    if args.action in ("prepare", "all"):
        print("Preparing data…")

        X_raw, y_raw = joblib.load("raw_data.pkl")
        (
            X_train,
            X_test,
            y_train,
            y_test,
            enc_state,
            enc_area,
        ) = prepare_data(X_raw, y_raw)

        validate_io(X_train, X_test, y_train, y_test)

        joblib.dump((X_train, X_test, y_train, y_test), "prepared_data.pkl")
        joblib.dump((enc_state, enc_area), "encoders.pkl")

        print("Prepare step done!")

        if args.action == "prepare":
            return

    # --- TRAIN ---
    if args.action in ("train", "all"):
        print("Training model…")

        X_train, X_test, y_train, y_test = joblib.load("prepared_data.pkl")
        enc_state, enc_area = joblib.load("encoders.pkl")

        model, scaler = train_model(X_train, y_train)
        save_model(model, scaler, enc_state, enc_area)

        print("Train step done!")

        if args.action == "train":
            return

    # --- EVALUATE ---
    if args.action in ("evaluate", "all"):
        print("Evaluating model…")

        model, scaler, enc_state, enc_area = load_model()
        X_train, X_test, y_train, y_test = joblib.load("prepared_data.pkl")

        evaluate_model(model, scaler, X_test, y_test)
        print("Evaluate step done!")


if __name__ == "__main__":
    main()