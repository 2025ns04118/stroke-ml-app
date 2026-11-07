"""Stroke Risk Workbench: a Streamlit front end for five trained classifiers.

Upload held-out patient records, choose a model, and inspect how it performs
(metrics, confusion matrix, ROC curve, leaderboard) or try a single patient.
"""
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.model_utils import (  # noqa: E402
    CATEGORICAL_COLS,
    FEATURE_COLS,
    ID_COL,
    MODEL_FILES,
    NUMERIC_COLS,
    TARGET,
    build_models,
    compute_metrics,
    get_scores,
    oversample_minority,
    split_xy,
)

SAMPLE_PATH = os.path.join(ROOT, "test_data.csv")
TRAIN_PATH = os.path.join(ROOT, "model", "train_data.csv")
CLASS_NAMES = ["No stroke", "Stroke"]

# colour palette (kept in step with .streamlit/config.toml)
TEAL, DEEP, CORAL, SAND, INK = "#0F766E", "#134E4A", "#E4572E", "#F2E8D5", "#12332F"
MODEL_COLORS = ["#0F766E", "#E4572E", "#C9A227", "#5B6C9D", "#8C5E58"]

MODEL_BLURBS = {
    "Logistic Regression": "Linear model that turns a weighted sum of features into a probability.",
    "Decision Tree": "Entropy-based tree of yes/no questions, depth limited to 5.",
    "kNN": "Labels a patient by the 5 most similar patients in the training data.",
    "Naive Bayes": "Bayes rule with a Gaussian, feature-independence assumption.",
    "Support Vector Machine": "RBF-kernel boundary that maximises the margin between classes.",
}
METRIC_HINTS = {
    "Accuracy": "share of all predictions that were right",
    "AUC": "how well strokes are ranked above non-strokes",
    "Precision": "of flagged patients, how many had a stroke",
    "Recall": "of real strokes, how many were caught",
    "F1": "balance of precision and recall",
    "MCC": "balanced score from -1 to +1",
}
METRIC_COLORS = dict(zip(METRIC_HINTS, [TEAL, "#5B6C9D", "#C9A227", CORAL, "#8C5E58", DEEP]))

st.set_page_config(page_title="Stroke Risk Workbench", page_icon="🧠", layout="wide")

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .hero {{
        background: linear-gradient(120deg, {DEEP} 0%, {TEAL} 60%, #2A9D8F 100%);
        border-radius: 18px; padding: 1.6rem 2rem; margin-bottom: 1.2rem; color: #fff;
    }}
    .hero h1 {{ color: #fff; margin: 0; font-size: 2.1rem; letter-spacing: .3px; }}
    .hero p {{ margin: .4rem 0 0 0; color: #D8F0EC; font-size: 1.02rem; }}
    .pill {{
        display: inline-block; background: rgba(255,255,255,.16); border-radius: 999px;
        padding: .15rem .8rem; margin: .7rem .4rem 0 0; font-size: .82rem; color: #fff;
    }}
    .mcard {{
        background: #fff; border-radius: 12px; padding: .8rem .9rem .7rem .9rem;
        box-shadow: 0 1px 6px rgba(18,51,47,.12); min-height: 7.4rem;
    }}
    .mlabel {{ font-size: .78rem; letter-spacing: .8px; text-transform: uppercase; color: #5b6f6c; }}
    .mvalue {{ font-size: 1.85rem; font-weight: 700; color: {INK}; line-height: 1.25; }}
    .mhint {{ font-size: .72rem; color: #7d8f8c; line-height: 1.25; }}
    .callout {{
        background: {SAND}; border-left: 5px solid {CORAL}; border-radius: 8px;
        padding: .7rem 1rem; margin: .9rem 0; color: {INK};
    }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8fa3a0",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load the saved pipelines; if they cannot be unpickled (for example the
    cloud runs another scikit-learn version), retrain from the bundled
    training split so the app still works."""
    try:
        return {n: joblib.load(os.path.join(ROOT, "model", f)) for n, f in MODEL_FILES.items()}, False
    except Exception:
        X_train, y_train = oversample_minority(*split_xy(pd.read_csv(TRAIN_PATH)))
        fitted = build_models()
        for pipe in fitted.values():
            pipe.fit(X_train, y_train)
        return fitted, True


@st.cache_data(show_spinner=False)
def load_reference():
    """Training split, used only to offer sensible choices in the what-if tab."""
    return pd.read_csv(TRAIN_PATH)


def clean_upload(df):
    """Tolerate raw Kaggle quirks: 'N/A' strings in bmi and an id column."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=[ID_COL], errors="ignore")


def metric_card(label, value):
    shown = "n/a" if pd.isna(value) else f"{value:.3f}"
    return (
        f'<div class="mcard" style="border-top:4px solid {METRIC_COLORS[label]}">'
        f'<div class="mlabel">{label}</div><div class="mvalue">{shown}</div>'
        f'<div class="mhint">{METRIC_HINTS[label]}</div></div>'
    )


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(show_spinner="Scoring all five models...")
def score_all(data):
    X_, y_ = split_xy(data)
    table = pd.DataFrame({n: compute_metrics(m, X_, y_) for n, m in models.items()}).T
    table.index.name = "ML Model Name"
    return table


models, retrained = load_models()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 🧠 Workbench")
st.sidebar.markdown("**Step 1 · Load patient records**")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
use_sample = st.sidebar.checkbox("Use the bundled test_data.csv", value=uploaded is None)
st.sidebar.markdown("**Step 2 · Pick a model**")
model_name = st.sidebar.selectbox("Classifier", list(models.keys()))
st.sidebar.info(MODEL_BLURBS[model_name])
st.sidebar.caption(
    "Expected columns: " + ", ".join(FEATURE_COLS) + f". Add `{TARGET}` to get performance scores."
)

# --------------------------------------------------------------------- hero
st.markdown(
    f"""
    <div class="hero">
      <h1>🧠 Stroke Risk Workbench</h1>
      <p>Compare five machine-learning classifiers on held-out patient records.</p>
      <span class="pill">Selected model: {model_name}</span>
      <span class="pill">5 classifiers</span>
      <span class="pill">6 evaluation metrics</span>
    </div>
    """,
    unsafe_allow_html=True,
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
    st.warning("Upload a CSV in the sidebar, or tick the box to use the bundled test data.")
    st.stop()

data = clean_upload(data)
missing = [c for c in FEATURE_COLS if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

X, y = split_xy(data)
model = models[model_name]
has_labels = y is not None
pred = model.predict(X)
score = get_scores(model, X)

with st.expander(f"Preview data · {source} · {data.shape[0]} rows × {data.shape[1]} columns"):
    st.dataframe(data.head(10))

tab_perf, tab_curves, tab_board, tab_whatif, tab_about = st.tabs(
    ["📊 Performance", "📈 Curves & scores", "🏁 Leaderboard", "🧪 Patient what-if", "ℹ️ About"]
)

# -------------------------------------------------------------- performance
with tab_perf:
    st.subheader(f"{model_name}: results on this data")
    if not has_labels:
        st.warning(f"No `{TARGET}` column found, so scores cannot be computed. Predictions are shown instead.")
        st.write(pd.Series(pred).map(dict(enumerate(CLASS_NAMES))).value_counts().rename("patients"))
    else:
        metrics = compute_metrics(model, X, y)
        for col, label in zip(st.columns(6), METRIC_HINTS):
            col.markdown(metric_card(label, metrics[label]), unsafe_allow_html=True)

        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        st.markdown(
            f'<div class="callout">Out of <b>{tp + fn}</b> real stroke cases the model caught '
            f"<b>{tp}</b> and missed <b>{fn}</b>. It also raised <b>{fp}</b> false alarms among "
            f"<b>{tn + fp}</b> patients without a stroke.</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1.1])
        with left:
            st.markdown("**Confusion matrix**")
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            sns.heatmap(
                np.array([[tn, fp], [fn, tp]]), annot=True, fmt="d", cbar=False,
                cmap=sns.light_palette(TEAL, as_cmap=True), linewidths=3, linecolor="white",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            show_fig(fig)
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y, pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(3))

    out = data.copy()
    out["predicted_stroke"] = pred
    out["model_score"] = np.round(score, 4)
    st.download_button(
        "⬇️ Download predictions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------- curves
with tab_curves:
    if not has_labels or y.nunique() < 2:
        st.info(f"ROC and score plots need a `{TARGET}` column containing both classes.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y, score)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"AUC = {metrics['AUC']:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.12, color=TEAL)
            ax.plot([0, 1], [0, 1], ls="--", color="#9aaeab", label="Random guess")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(frameon=False, loc="lower right")
            show_fig(fig)
        with c2:
            st.markdown("**How the model scores each group**")
            fig, ax = plt.subplots(figsize=(5, 4))
            for cls, colour in zip([0, 1], [TEAL, CORAL]):
                ax.hist(score[(y == cls).to_numpy()], bins=25, alpha=0.6, density=True,
                        color=colour, label=CLASS_NAMES[cls])
            ax.set_xlabel("Model score (higher = more stroke-like)")
            ax.set_ylabel("Density")
            ax.legend(frameon=False)
            show_fig(fig)
        st.caption("The more the two histograms separate, the easier it is to tell strokes apart; the ROC curve summarises the same idea.")

# -------------------------------------------------------------- leaderboard
with tab_board:
    if not has_labels:
        st.info(f"The leaderboard needs a `{TARGET}` column.")
    else:
        board = score_all(data)
        st.subheader("All five models on this data")
        st.dataframe(board.round(4))

        rows = {
            m: {"Best model": board[m].idxmax(), "Score": board[m].max()}
            for m in board.columns
            if board[m].notna().any()  # AUC is undefined when only one class is present
        }
        st.markdown("**Best model for each metric**")
        st.dataframe(pd.DataFrame(rows).T)

        fig, ax = plt.subplots(figsize=(10, 4))
        board.T.plot(kind="bar", ax=ax, color=MODEL_COLORS, width=0.8, rot=0)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        show_fig(fig)
        st.caption("Look beyond accuracy: strokes are rare, so Recall, F1 and MCC show whether a model really finds them.")

# ----------------------------------------------------------------- what-if
with tab_whatif:
    st.subheader("Try a single patient")
    st.write(f"Enter details and see what **{model_name}** predicts. This is a classroom demo, not medical advice.")
    ref = load_reference()

    a, b, c = st.columns(3)
    with a:
        age = st.slider("Age", 1, 100, value=55)
        glucose = st.slider("Average glucose level", 50.0, 300.0, value=105.0)
        bmi = st.slider("BMI", 10.0, 70.0, value=28.0)
    with b:
        gender = st.selectbox("Gender", sorted(ref["gender"].unique()), index=0)
        married = st.selectbox("Ever married", sorted(ref["ever_married"].unique()), index=0)
        residence = st.selectbox("Residence", sorted(ref["Residence_type"].unique()), index=0)
    with c:
        work = st.selectbox("Work type", sorted(ref["work_type"].unique()), index=0)
        smoking = st.selectbox("Smoking status", sorted(ref["smoking_status"].unique()), index=0)
        hyper = st.checkbox("Hypertension", value=False)
        heart = st.checkbox("Heart disease", value=False)

    patient = pd.DataFrame(
        [{
            "age": age, "avg_glucose_level": glucose, "bmi": bmi,
            "hypertension": int(hyper), "heart_disease": int(heart),
            "gender": gender, "ever_married": married, "work_type": work,
            "Residence_type": residence, "smoking_status": smoking,
        }]
    )[FEATURE_COLS]

    verdict = int(model.predict(patient)[0])
    if verdict == 1:
        st.error("⚠️ The model flags this patient as **higher stroke risk**.")
    else:
        st.success("✅ The model does **not** flag this patient.")
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(patient)[0, 1])
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Model score: {prob:.1%}. Models were trained on a balanced (oversampled) set, "
            "so this is a relative risk score, not a real-world probability."
        )
    else:
        st.caption(f"Decision score: {float(model.decision_function(patient)[0]):.3f} (above 0 means stroke).")

# -------------------------------------------------------------------- about
with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
**Task:** binary classification, predicting whether a patient has had a stroke.

**Data:** Kaggle *Stroke Prediction Dataset*, 5,110 patients and 10 input features.
The data was split 80/20 (stratified, seed 42); this app evaluates on the 20% test part.

**Pipeline:** median imputation of missing BMI, standard scaling of numeric features,
one-hot encoding of categorical features, and random oversampling of stroke cases in the
*training* data only.

**Models:** Logistic Regression, Decision Tree (entropy), k-Nearest Neighbors, Gaussian
Naive Bayes and an RBF Support Vector Machine.

**Metrics:** Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient.
        """
    )
    st.caption("Built for the Machine Learning course, M.Tech (DSE), BITS Pilani WILP.")
