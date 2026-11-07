"""Shared helpers: preprocessing, the five classifiers, and the evaluation metrics.

Both train_models.py and the Streamlit app import from here so the pipeline
used for training is exactly the one used for evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

TARGET = "stroke"
ID_COL = "id"
RANDOM_STATE = 42

NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLS = ["hypertension", "heart_disease"]
CATEGORICAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_preprocessor():
    """Median-impute + scale numeric columns, one-hot encode categoricals."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def oversample_minority(X, y):
    """Data skewness removal by random oversampling.

    Only ~5% of patients had a stroke. The minority class is resampled (with
    replacement) until both classes are the same size. Apply this to the
    TRAINING split only; the test split is never resampled.
    """
    counts = y.value_counts()
    minority, majority = counts.idxmin(), counts.idxmax()
    X_min = X[y == minority]
    extra = resample(
        X_min,
        replace=True,
        n_samples=counts[majority] - counts[minority],
        random_state=RANDOM_STATE,
    )
    X_bal = pd.concat([X, extra])
    y_bal = pd.concat([y, pd.Series(minority, index=extra.index)])
    order = np.random.RandomState(RANDOM_STATE).permutation(len(X_bal))
    return X_bal.iloc[order].reset_index(drop=True), y_bal.iloc[order].reset_index(drop=True)


def build_models():
    """Return {name: untrained Pipeline(preprocessor + classifier)}."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        # entropy / information gain, depth capped to limit overfitting
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", random_state=RANDOM_STATE),
    }
    return {
        name: Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        for name, clf in classifiers.items()
    }


def split_xy(df):
    """Separate features from the target (target is optional in uploaded data)."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    return X, y


def get_scores(model, X):
    """Continuous score for the positive class, used for AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)  # SVC without probability=True


def compute_metrics(model, X, y):
    """Accuracy, AUC, Precision, Recall, F1 and MCC for one fitted pipeline."""
    pred = model.predict(X)
    try:
        auc = roc_auc_score(y, get_scores(model, X))
    except ValueError:  # only one class present in y
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
