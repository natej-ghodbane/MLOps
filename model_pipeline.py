import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import joblib

TRAIN_PATH = "churn-bigml-80.csv"
TEST_PATH = "churn-bigml-20.csv"

# Feature subset extracted from notebook XGB_importances_column
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


def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    """
    Load train/test CSV files and return raw DataFrames.
    """
    X = pd.read_csv(train_path)
    y = pd.read_csv(test_path)

    if "Churn" not in X.columns or "Churn" not in y.columns:
        raise ValueError("Missing 'Churn' column in CSV files.")

    print("load_data(): files loaded successfully.")
    return X, y


def prepare_data(X, y):
    """
    Full data preparation:
    - Encoding
    - One-hot
    - Feature engineering
    - Drop correlations
    - SMOTEENN
    """

    # -------- Binary encoding --------
    for col in ["International plan", "Voice mail plan"]:
        X[col] = X[col].map({"No": 0, "Yes": 1})
        y[col] = y[col].map({"No": 0, "Yes": 1})

    # -------- One-hot encoders --------
    encoder_state = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder_area = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    state_train = encoder_state.fit_transform(X[["State"]])
    state_test = encoder_state.transform(y[["State"]])

    area_train = encoder_area.fit_transform(X[["Area code"]])
    area_test = encoder_area.transform(y[["Area code"]])

    # Convert encoded features to DataFrames
    state_train_df = pd.DataFrame(
        state_train,
        columns=encoder_state.get_feature_names_out(["State"]),
        index=X.index,
    )
    state_test_df = pd.DataFrame(
        state_test,
        columns=encoder_state.get_feature_names_out(["State"]),
        index=y.index,
    )
    area_train_df = pd.DataFrame(
        area_train,
        columns=encoder_area.get_feature_names_out(["Area code"]),
        index=X.index,
    )
    area_test_df = pd.DataFrame(
        area_test,
        columns=encoder_area.get_feature_names_out(["Area code"]),
        index=y.index,
    )

    # Replace categorical with encoded
    X = X.drop(["State", "Area code"], axis=1)
    y = y.drop(["State", "Area code"], axis=1)

    X = pd.concat([X, state_train_df, area_train_df], axis=1)
    y = pd.concat([y, state_test_df, area_test_df], axis=1)

    # -------- Feature engineering --------
    for df in (X, y):
        df["Total calls"] = (
            df["Total day calls"]
            + df["Total eve calls"]
            + df["Total night calls"]
            + df["Total intl calls"]
        )
        df["Total charge"] = (
            df["Total day charge"]
            + df["Total eve charge"]
            + df["Total night charge"]
            + df["Total intl charge"]
        )
        df["CScalls Rate"] = (
            df["Customer service calls"] / df["Account length"]
        )

    # -------- Drop correlated columns --------
    correlated_columns = [
        "Total day minutes",
        "Total eve minutes",
        "Total night minutes",
        "Total intl minutes",
        "Voice mail plan",
    ]
    X = X.drop(correlated_columns, axis=1)
    y = y.drop(correlated_columns, axis=1)

    # -------- Split X/y --------
    X_train = X.drop("Churn", axis=1)
    y_train = X["Churn"]

    X_test = y.drop("Churn", axis=1)
    y_test = y["Churn"]

    # -------- SMOTEENN --------
    smote_enn = SMOTEENN(sampling_strategy=30 / 70, random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    print("prepare_data(): preprocessing complete.")
    return (
        X_resampled,
        X_test,
        y_resampled,
        y_test,
        encoder_state,
        encoder_area,
    )


def train_model(X_train, y_train):
    """
    Train XGBoost using tuned hyperparameters.
    """
    for col in XGB_IMPORTANCE_COLS:
        if col not in X_train.columns:
            raise ValueError(f"Missing required feature: {col}")

    X_train_subset = X_train[XGB_IMPORTANCE_COLS].astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.014319674883945213,
        max_depth=10,
        min_child_weight=3,
        subsample=0.9993815446215235,
        colsample_bytree=0.9185894728369491,
        gamma=1.9937854420882333,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train_scaled, y_train)

    print("train_model(): training completed.")
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate model performance.
    """
    X_test_subset = X_test[XGB_IMPORTANCE_COLS].astype(float)
    X_test_scaled = scaler.transform(X_test_subset)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)
    print("ROC-AUC:", roc_auc)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
    }


def save_model(model, scaler, encoder_state, encoder_area, prefix="churn"):
    joblib.dump(model, f"{prefix}_model.pkl")
    joblib.dump(scaler, f"{prefix}_scaler.pkl")
    joblib.dump(encoder_state, f"{prefix}_encoder_state.pkl")
    joblib.dump(encoder_area, f"{prefix}_encoder_area.pkl")
    print("Model and preprocessors saved.")


def load_model(prefix="churn"):
    model = joblib.load(f"{prefix}_model.pkl")
    scaler = joblib.load(f"{prefix}_scaler.pkl")
    encoder_state = joblib.load(f"{prefix}_encoder_state.pkl")
    encoder_area = joblib.load(f"{prefix}_encoder_area.pkl")
    print("Model and preprocessors loaded.")
    return model, scaler, encoder_state, encoder_area
