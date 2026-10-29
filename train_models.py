from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]
        ),
    }


def safe_roc_auc(y_true, y_pred, y_proba):
    if y_proba is None:
        return float("nan")
    return roc_auc_score(y_true, y_proba)


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": safe_roc_auc(y_test, y_pred, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_names = list(X.columns)
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)

    summary_rows = []
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        summary_rows.append({"ML Model Name": model_name, **metrics})

        file_name = model_name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / file_name)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            "ML Model Name",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
        ]
    ]
    summary_df.to_csv("model_metrics.csv", index=False)
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
