"""Train the five classifiers on the 80% training split and evaluate them on
the 20% test split. Saves each fitted pipeline to model/*.joblib and the
metrics table to model/metrics_summary.csv.

Run from the project root:   python model/train_models.py
"""
import os
import sys

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from model.model_utils import (  # noqa: E402
    MODEL_FILES,
    build_models,
    compute_metrics,
    oversample_minority,
    split_xy,
)

TRAIN_PATH = os.path.join(HERE, "train_data.csv")
TEST_PATH = os.path.join(ROOT, "test_data.csv")


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Stroke cases -> train: {y_train.sum()}  test: {y_test.sum()}")

    # skewness removal: balance the TRAINING data only
    X_bal, y_bal = oversample_minority(X_train, y_train)
    print(f"After oversampling, train class counts: {y_bal.value_counts().to_dict()}")

    rows = {}
    for name, pipe in build_models().items():
        pipe.fit(X_bal, y_bal)
        joblib.dump(pipe, os.path.join(HERE, MODEL_FILES[name]))
        rows[name] = compute_metrics(pipe, X_test, y_test)
        print(f"trained + saved: {name}")

    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    table.to_csv(os.path.join(HERE, "metrics_summary.csv"))

    n_features = build_models()["kNN"].named_steps["prep"].fit(X_train).get_feature_names_out().size
    print(f"\nModel input features after encoding: {n_features}")
    print("\nTest-set metrics:\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
