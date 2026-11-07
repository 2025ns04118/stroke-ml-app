"""Streamlit front end: evaluate five trained classifiers on stroke test data."""
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.model_utils import (  # noqa: E402
    FEATURE_COLS,
    ID_COL,
    MODEL_FILES,
    NUMERIC_COLS,
    TARGET,
    build_models,
    compute_metrics,
    oversample_minority,
    split_xy,
)

SAMPLE_PATH = os.path.join(ROOT, "test_data.csv")
TRAIN_PATH = os.path.join(ROOT, "model", "train_data.csv")
CLASS_NAMES = ["No stroke (0)", "Stroke (1)"]

st.set_page_config(page_title="Stroke Risk - Model Evaluation", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load the saved pipelines. If they cannot be unpickled (for example the
    cloud runs a different scikit-learn version), retrain from the bundled
    training split so the app still works."""
    try:
        return {n: joblib.load(os.path.join(ROOT, "model", f)) for n, f in MODEL_FILES.items()}, False
    except Exception:
        X_train, y_train = oversample_minority(*split_xy(pd.read_csv(TRAIN_PATH)))
        models = build_models()
        for pipe in models.values():
            pipe.fit(X_train, y_train)
        return models, True


def clean_upload(df):
    """Tolerate the raw Kaggle quirks: 'N/A' strings in bmi, an id column."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=[ID_COL], errors="ignore")


models, retrained = load_models()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
use_sample = st.sidebar.checkbox("Use the bundled test_data.csv", value=uploaded is None)
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
st.sidebar.caption(
    "Expected columns: " + ", ".join(FEATURE_COLS) + f" (+ optional `{TARGET}` for scoring)."
)

# ------------------------------------------------------------------- main
st.title("🩺 Stroke Prediction: Classifier Evaluation")
st.write(
    "Pick a model, upload held-out patient records, and see how well the model "
    "separates stroke from non-stroke cases."
)
if retrained:
    st.info("Saved models were built with another scikit-learn version, so they were retrained at start-up.")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(SAMPLE_PATH)
    source = "test_data.csv (bundled)"
else:
    st.warning("Upload a CSV or tick the box to use the bundled test data.")
    st.stop()

data = clean_upload(data)
missing = [c for c in FEATURE_COLS if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

st.subheader("Data")
st.caption(f"Source: {source}  |  {data.shape[0]} rows, {data.shape[1]} columns")
st.dataframe(data.head(10))

X, y = split_xy(data)
model = models[model_name]

if y is None:
    st.warning(f"No `{TARGET}` column found, so metrics cannot be computed. Showing predictions only.")
    preds = pd.Series(model.predict(X)).map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
    st.write(preds.value_counts().rename("count"))
    st.stop()

# ---------------------------------------------------------------- metrics
st.subheader(f"Results: {model_name}")
metrics = compute_metrics(model, X, y)

row1 = st.columns(3)
row2 = st.columns(3)
for col, (label, value) in zip(row1 + row2, metrics.items()):
    col.metric(label, "n/a" if pd.isna(value) else f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, model.predict(X), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, model.predict(X), labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3))

with st.expander("Compare all five models on this data"):
    table = pd.DataFrame({n: compute_metrics(m, X, y) for n, m in models.items()}).T
    table.index.name = "ML Model Name"
    st.dataframe(table.round(4))
