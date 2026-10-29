from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIR = Path("model")
FEATURES_PATH = MODEL_DIR / "feature_names.json"
DEFAULT_DATA_PATH = Path("test_data.csv")


@st.cache_resource
def load_model(model_name: str):
    filename = model_name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
    return joblib.load(MODEL_DIR / filename)


@st.cache_data
def default_dataset():
    return pd.read_csv(DEFAULT_DATA_PATH)


@st.cache_data
def get_feature_names():
    if not FEATURES_PATH.exists():
        return []
    return pd.read_json(FEATURES_PATH)


def compute_metrics_for_model(model, X, y_true):
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X)
        except Exception:
            y_proba = None

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics


st.set_page_config(page_title="ML Classification Model Dashboard", layout="wide")
st.title("Breast Cancer Classification Dashboard")
st.caption("Built for the Machine Learning Assignment 2 requirement.")

feature_names = []
try:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = __import__("json").load(f)
except FileNotFoundError:
    feature_names = []

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
]

uploaded_file = st.file_uploader("Upload a CSV file with the required features and a target column", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = default_dataset()

if "target" not in df.columns:
    st.error("The uploaded file must contain a 'target' column.")
    st.stop()

if not feature_names:
    feature_names = [col for col in df.columns if col != "target"]

missing = [col for col in feature_names if col not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = df[feature_names]
y_true = df["target"]
selected_model = st.selectbox("Select a model to evaluate", model_names)
model = load_model(selected_model)
y_pred = model.predict(X)

metrics = compute_metrics_for_model(model, X, y_true)

st.subheader(f"Model: {selected_model}")
col1, col2, col3, col4, col5, col6 = st.columns(6)
for col, key in zip(
    [col1, col2, col3, col4, col5, col6],
    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
):
    with col:
        value = metrics[key]
        st.metric(key, f"{value:.4f}" if pd.notna(value) else "N/A")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
st.pyplot(fig)

comparison_rows = []
for name in model_names:
    loaded_model = load_model(name)
    row = {"ML Model Name": name, **compute_metrics_for_model(loaded_model, X, y_true)}
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df[
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
st.subheader("Comparison of all models on the uploaded dataset")
st.dataframe(comparison_df.round(4), use_container_width=True)

st.subheader("Classification report")
report = pd.DataFrame(
    {
        "precision": [precision_score(y_true, y_pred, zero_division=0)],
        "recall": [recall_score(y_true, y_pred, zero_division=0)],
        "f1-score": [f1_score(y_true, y_pred, zero_division=0)],
        "support": [len(y_true)],
    },
    index=[selected_model],
)
st.dataframe(report)
