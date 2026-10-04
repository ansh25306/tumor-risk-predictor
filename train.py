"""Train the tumour-risk classifier and save the model + metadata.

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository. A copy ships
inside scikit-learn (`sklearn.datasets.load_breast_cancer`), so no download is
needed and the run is reproducible offline.

Usage:
    python train.py

Outputs (in ./artifacts):
    model.joblib    scikit-learn Pipeline (StandardScaler + classifier)
    metadata.json   feature schema, valid ranges, example profiles, metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# (dataset column, key used by the app, label, help text)
FEATURES = [
    ("mean radius", "mean_radius", "Mean radius",
     "Average distance from the nucleus centre to its boundary."),
    ("mean texture", "mean_texture", "Mean texture",
     "Standard deviation of grey-scale values inside the nucleus."),
    ("mean area", "mean_area", "Mean area",
     "Average area of the cell nucleus."),
    ("mean smoothness", "mean_smoothness", "Mean smoothness",
     "Local variation in radius lengths."),
    ("mean compactness", "mean_compactness", "Mean compactness",
     "Perimeter^2 / area - 1.0."),
    ("mean concavity", "mean_concavity", "Mean concavity",
     "Severity of concave portions of the contour."),
    ("mean concave points", "mean_concave_points", "Mean concave points",
     "Number of concave portions of the contour."),
    ("mean symmetry", "mean_symmetry", "Mean symmetry",
     "Symmetry of the nucleus shape."),
]

CLASS_NAMES = ["Benign", "Malignant"]  # index 1 == positive class (malignant)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    raw = load_breast_cancer(as_frame=True)
    cols = [f[0] for f in FEATURES]
    X = raw.data[cols].copy()
    X.columns = [f[1] for f in FEATURES]
    # sklearn encodes malignant=0, benign=1; flip so 1 == malignant.
    y = (raw.target == 0).astype(int)
    return X, y


def candidate_models() -> dict[str, GridSearchCV | Pipeline]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    logreg = GridSearchCV(
        Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
        scoring="roc_auc",
        cv=cv,
    )
    forest = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ])
    return {"logistic_regression": logreg, "random_forest": forest}


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidate_models().items():
        if isinstance(estimator, GridSearchCV):
            estimator.fit(X_train, y_train)
            best = estimator.best_estimator_
            mean = float(estimator.best_score_)
            std = float(estimator.cv_results_["std_test_score"][estimator.best_index_])
        else:
            scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc")
            estimator.fit(X_train, y_train)
            best, mean, std = estimator, float(scores.mean()), float(scores.std())
        fitted[name] = best
        cv_results[name] = {"cv_roc_auc_mean": round(mean, 4), "cv_roc_auc_std": round(std, 4)}
        print(f"{name:20s} CV ROC-AUC = {mean:.4f} +/- {std:.4f}")

    # Pick on cross-validation only; the test set is touched once, for reporting.
    # Prefer the simpler, explainable logistic regression unless the forest wins
    # by more than one CV standard deviation (i.e. a difference beyond noise).
    lr, rf = cv_results["logistic_regression"], cv_results["random_forest"]
    rf_clearly_better = rf["cv_roc_auc_mean"] - lr["cv_roc_auc_mean"] > max(
        lr["cv_roc_auc_std"], rf["cv_roc_auc_std"]
    )
    chosen = "random_forest" if rf_clearly_better else "logistic_regression"
    model = fitted[chosen]
    print(f"Selected model: {chosen}")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(json.dumps(test_metrics, indent=2))

    # Refit the chosen pipeline on ALL data for the deployed model.
    deployed = clone(model).fit(X, y)

    benign, malignant = X[y == 0], X[y == 1]
    features_meta = []
    for _, key, label, help_text in FEATURES:
        features_meta.append({
            "key": key,
            "label": label,
            "help": help_text,
            "min": float(X[key].min()),
            "max": float(X[key].max()),
            "median": float(X[key].median()),
        })

    metadata = {
        "title": "Tumour Risk Predictor",
        "dataset": "Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository",
        "n_samples": int(len(X)),
        "class_names": CLASS_NAMES,
        "positive_class": "Malignant",
        "model_type": chosen,
        "features": features_meta,
        "examples": {
            "Typical benign profile": {k: float(benign[k].median()) for k in X.columns},
            "Typical malignant profile": {k: float(malignant[k].median()) for k in X.columns},
        },
        "cv": cv_results,
        "test_metrics": test_metrics,
        "trained_with": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "seed": SEED},
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(deployed, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model and metadata to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
