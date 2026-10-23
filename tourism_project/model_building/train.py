"""Job 3 - compare baselines, tune XGBoost, log everything to MLflow and save the model."""
import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
logging.getLogger("mlflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DEPLOYMENT_DIR = ROOT / "tourism_project" / "deployment"
REPORTS_DIR.mkdir(exist_ok=True)
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
EXPERIMENT = "Tourism_Wellness_Package_Prediction"


def load_splits():
    X_train = pd.read_csv(ROOT / "Xtrain.csv")
    X_test = pd.read_csv(ROOT / "Xtest.csv")
    y_train = pd.read_csv(ROOT / "ytrain.csv").squeeze("columns")
    y_test = pd.read_csv(ROOT / "ytest.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def build_pipeline(classifier, numeric_features, categorical_features):
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", classifier),
    ])


def main():
    X_train, X_test, y_train, y_test = load_splits()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    print("=" * 66)
    print("MODEL TRAINING")
    print("=" * 66)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Numeric features   ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1
        ),
    }

    print("\n" + "-" * 66)
    print("BASELINE COMPARISON (3-fold stratified CV, F1 on the buyer class)")
    print("-" * 66)
    comparison = []
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier, numeric_features, categorical_features)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=3, scoring="f1", n_jobs=-1
        )
        comparison.append({
            "model": name,
            "cv_f1_mean": scores.mean(),
            "cv_f1_std": scores.std(),
        })
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.log_param("model", name)
            mlflow.log_param("stage", "baseline")
            mlflow.log_metric("cv_f1_mean", float(scores.mean()))
            mlflow.log_metric("cv_f1_std", float(scores.std()))
        print(f"{name:<20} CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    comparison_df = pd.DataFrame(comparison).sort_values("cv_f1_mean", ascending=False)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n" + "-" * 66)
    print("HYPERPARAMETER TUNING (XGBoost, GridSearchCV, scoring='f1')")
    print("-" * 66)
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__scale_pos_weight": [1, 3],
    }
    search = GridSearchCV(
        build_pipeline(
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
            numeric_features,
            categorical_features,
        ),
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="xgboost_tuned") as run:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        print("Best parameters:", best_params)
        print(f"Best CV F1     : {search.best_score_:.4f}")

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
        }

        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("stage", "tuned")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model, name="model", serialization_format="cloudpickle"
        )

        joblib.dump(best_model, DEPLOYMENT_DIR / "model.joblib")
        (REPORTS_DIR / "model_metrics.json").write_text(
            json.dumps({"best_params": best_params, "metrics": metrics,
                        "mlflow_run_id": run.info.run_id}, indent=2)
        )

    print("\n" + "-" * 66)
    print("HELD-OUT TEST PERFORMANCE (threshold = 0.50)")
    print("-" * 66)
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

    print("\nModel saved to:", DEPLOYMENT_DIR / "model.joblib")
    print("MLflow run ID :", run.info.run_id)
    print("=" * 66)


if __name__ == "__main__":
    main()
