"""
MLflow experiment runs - XGBoost vs LightGBM for IEEE-CIS Fraud Detection.

Run:
    python -m src.train.train

Each run logs:
    - params: model type, all hyperparameters
    - metrics: auc_roc, auc_pr, f1, best_threshold, n_estimators_used
    - artifacts: shap_summary.png, confusion_matrix_05.png,
                 confusion_matrix_opt.png, pipeline.pkl
    - model: native XGBoost / LightGBM flavor (loadable as pyfunc)

After all runs, the best AUC-PR run is registered in the MLflow Model
Registry as 'fraudlens-detector' and transitioned to stage 'Production'.
"""

import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EXPERIMENT_NAME = "fraudlens-detection"
REGISTRY_MODEL_NAME = "fraudlens-detector"

# - Hyperparameter grids
# 12 XGBoost + 12 LightGBM = 24 named runs (README asks for 20+)

XGB_CONFIGS: list[dict[str, Any]] = [
    {"name": "xgb-baseline",     "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
    {"name": "xgb-deep",         "n_estimators": 1000, "max_depth": 8, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 1},
    {"name": "xgb-shallow",      "n_estimators": 1000, "max_depth": 4, "learning_rate": 0.05,  "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1},
    {"name": "xgb-slow-lr",      "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.01,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
    {"name": "xgb-fast-lr",      "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.10,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
    {"name": "xgb-regularized",  "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5,  "reg_alpha": 0.1, "reg_lambda": 2.0},
    {"name": "xgb-deep-reg",     "n_estimators": 1000, "max_depth": 8, "learning_rate": 0.03,  "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5},
    {"name": "xgb-wide",         "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.9, "colsample_bytree": 1.0, "min_child_weight": 1},
    {"name": "xgb-conservative", "n_estimators": 1000, "max_depth": 5, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10},
    {"name": "xgb-aggressive",   "n_estimators": 1000, "max_depth": 9, "learning_rate": 0.05,  "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 1},
    {"name": "xgb-l1",           "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,  "reg_alpha": 1.0, "reg_lambda": 1.0},
    {"name": "xgb-low-var",      "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.5, "colsample_bytree": 0.5, "min_child_weight": 20},
]

LGBM_CONFIGS: list[dict[str, Any]] = [
    {"name": "lgbm-baseline",     "n_estimators": 1000, "max_depth": 6,  "learning_rate": 0.05, "num_leaves": 63,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20},
    {"name": "lgbm-deep",         "n_estimators": 1000, "max_depth": 8,  "learning_rate": 0.05, "num_leaves": 127, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 20},
    {"name": "lgbm-shallow",      "n_estimators": 1000, "max_depth": 4,  "learning_rate": 0.05, "num_leaves": 31,  "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 20},
    {"name": "lgbm-slow-lr",      "n_estimators": 1000, "max_depth": 6,  "learning_rate": 0.01, "num_leaves": 63,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20},
    {"name": "lgbm-fast-lr",      "n_estimators": 1000, "max_depth": 6,  "learning_rate": 0.10, "num_leaves": 63,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20},
    {"name": "lgbm-regularized",  "n_estimators": 1000, "max_depth": 6,  "learning_rate": 0.05, "num_leaves": 63,  "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 2.0},
    {"name": "lgbm-many-leaves",  "n_estimators": 1000, "max_depth": -1, "learning_rate": 0.05, "num_leaves": 255, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 50},
    {"name": "lgbm-few-leaves",   "n_estimators": 1000, "max_depth": -1, "learning_rate": 0.05, "num_leaves": 31,  "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 100},
    {"name": "lgbm-dart",         "n_estimators": 300,  "max_depth": 6,  "learning_rate": 0.05, "num_leaves": 63,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20, "boosting_type": "dart"},
    {"name": "lgbm-conservative", "n_estimators": 1000, "max_depth": 5,  "learning_rate": 0.05, "num_leaves": 31,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 100},
    {"name": "lgbm-l1",           "n_estimators": 1000, "max_depth": 6,  "learning_rate": 0.05, "num_leaves": 63,  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20, "reg_alpha": 1.0},
    {"name": "lgbm-bagging",      "n_estimators": 1000, "max_depth": 6,  "learning_rate": 0.05, "num_leaves": 63,  "subsample": 0.6, "colsample_bytree": 0.6, "min_child_samples": 20, "subsample_freq": 5},
]


# - Data loading

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load processed features; run the pipeline first if needed."""
    parquet_path = PROCESSED_DIR / "train.parquet"
    if not parquet_path.exists():
        print("Processed parquet not found — running feature pipeline...")
        from src.train.features import build as build_features
        build_features("train")

    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c != "isFraud"]
    X = df[feature_cols]
    y = df["isFraud"]
    return X, y, feature_cols


def _make_splits(X: pd.DataFrame, y: pd.Series, val_size: float = 0.2):
    return train_test_split(X, y, test_size=val_size, random_state=42, stratify=y)


# - Metric helpers

def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 on the validation set."""
    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                 for t in thresholds]
    return float(thresholds[np.argmax(f1_scores)])


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc":        float(roc_auc_score(y_true, y_prob)),
        "auc_pr":         float(average_precision_score(y_true, y_prob)),
        "f1":             float(f1_score(y_true, y_pred, zero_division=0)),
        "best_threshold": float(threshold),
    }


# - Plot helpers

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    title: str,
) -> plt.Figure:
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    classes = ["Non-fraud", "Fraud"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11)
    fig.tight_layout()
    return fig


def _plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 20,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=True).tail(max_display)
    ax.barh(mean_abs.index, mean_abs.values, color="#3498db", alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — Top {max_display} (SHAP)")
    fig.tight_layout()
    return fig


# - Single-run training

def _run_xgb(
    config: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    experiment_id: str,
) -> str:
    run_name = config["name"]
    params = {k: v for k, v in config.items() if k != "name"}

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        mlflow.log_params({"model_type": "xgboost", **params})

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=50,
            verbosity=0,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        n_used = model.best_iteration + 1
        mlflow.log_param("n_estimators_used", n_used)

        y_prob = model.predict_proba(X_val)[:, 1]
        threshold = _find_best_threshold(y_val.values, y_prob)
        metrics = _compute_metrics(y_val.values, y_prob, threshold)
        mlflow.log_metrics(metrics)

        # Confusion matrices
        fig_05 = _plot_confusion_matrix(y_val.values, y_prob, 0.5, f"{run_name} — threshold 0.5")
        fig_opt = _plot_confusion_matrix(y_val.values, y_prob, threshold, f"{run_name} — threshold {threshold:.3f} (opt)")
        mlflow.log_figure(fig_05,  "confusion_matrix_05.png")
        mlflow.log_figure(fig_opt, "confusion_matrix_opt.png")
        plt.close("all")

        # SHAP on a random subsample (fast)
        sample_idx = np.random.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
        X_shap = X_val.iloc[sample_idx]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        fig_shap = _plot_shap_summary(shap_values, X_shap)
        mlflow.log_figure(fig_shap, "shap_summary.png")
        plt.close("all")

        # Model + pipeline artifact
        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(PROCESSED_DIR / "pipeline.pkl"), artifact_path="pipeline")

        print(f"  [{run_name}] AUC-PR={metrics['auc_pr']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}  "
              f"F1={metrics['f1']:.4f}  trees={n_used}")
        return run.info.run_id


def _run_lgbm(
    config: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    experiment_id: str,
) -> str:
    run_name = config["name"]
    params = {k: v for k, v in config.items() if k != "name"}
    is_dart = params.get("boosting_type") == "dart"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        mlflow.log_params({"model_type": "lightgbm", **params})

        model = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        fit_kwargs: dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "average_precision",
        }
        if not is_dart:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        model.fit(X_tr, y_tr, **fit_kwargs)

        n_used = model.best_iteration_ if (not is_dart and model.best_iteration_ > 0) else params["n_estimators"]
        mlflow.log_param("n_estimators_used", n_used)

        y_prob = model.predict_proba(X_val)[:, 1]
        threshold = _find_best_threshold(y_val.values, y_prob)
        metrics = _compute_metrics(y_val.values, y_prob, threshold)
        mlflow.log_metrics(metrics)

        fig_05 = _plot_confusion_matrix(y_val.values, y_prob, 0.5, f"{run_name} — threshold 0.5")
        fig_opt = _plot_confusion_matrix(y_val.values, y_prob, threshold, f"{run_name} — threshold {threshold:.3f} (opt)")
        mlflow.log_figure(fig_05,  "confusion_matrix_05.png")
        mlflow.log_figure(fig_opt, "confusion_matrix_opt.png")
        plt.close("all")

        sample_idx = np.random.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
        X_shap = X_val.iloc[sample_idx]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        # LightGBM TreeExplainer may return list [neg_class, pos_class] — take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fig_shap = _plot_shap_summary(shap_values, X_shap)
        mlflow.log_figure(fig_shap, "shap_summary.png")
        plt.close("all")

        mlflow.lightgbm.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(PROCESSED_DIR / "pipeline.pkl"), artifact_path="pipeline")

        print(f"  [{run_name}] AUC-PR={metrics['auc_pr']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}  "
              f"F1={metrics['f1']:.4f}  trees={n_used}")
        return run.info.run_id


# - Model registry promotion

def _promote_best(experiment_id: str) -> None:
    """Register the best AUC-PR run as 'Production' in the model registry."""
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.auc_pr DESC"],
        max_results=1,
    )
    if not runs:
        print("No completed runs found — skipping registry promotion.")
        return

    best_run = runs[0]
    best_metrics = best_run.data.metrics
    best_run_id = best_run.info.run_id
    model_uri = f"runs:/{best_run_id}/model"

    print(f"\nBest run: {best_run.data.tags.get('mlflow.runName', best_run_id)}")
    print(f"  AUC-PR  = {best_metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC = {best_metrics['auc_roc']:.4f}")
    print(f"  F1      = {best_metrics['f1']:.4f}")

    # Register
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_MODEL_NAME)
    print(f"\nRegistered as '{REGISTRY_MODEL_NAME}' version {mv.version}")

    # Transition to Production
    client.transition_model_version_stage(
        name=REGISTRY_MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Transitioned version {mv.version} → Production")

    # Tag the run for downstream retrieval
    client.set_tag(best_run_id, "registry_stage", "Production")
    client.set_tag(best_run_id, "registry_version", str(mv.version))


# - Entry point

def main() -> None:
    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    print("Loading features...")
    X, y, feature_cols = _load_data()
    X_tr, X_val, y_tr, y_val = _make_splits(X, y)

    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Train: {len(X_tr):,} rows  |  Val: {len(X_val):,} rows")
    print(f"Fraud rate (train): {y_tr.mean():.3%}  |  scale_pos_weight: {scale_pos_weight:.1f}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nMLflow experiment: '{EXPERIMENT_NAME}'  (id={experiment_id})")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}\n")

    print("=" * 60)
    print("XGBoost runs")
    print("=" * 60)
    for cfg in XGB_CONFIGS:
        try:
            _run_xgb(cfg, X_tr, y_tr, X_val, y_val, scale_pos_weight, experiment_id)
        except Exception as e:
            print(f"  [{cfg['name']}] FAILED: {e}")

    print("\n" + "=" * 60)
    print("LightGBM runs")
    print("=" * 60)
    for cfg in LGBM_CONFIGS:
        try:
            _run_lgbm(cfg, X_tr, y_tr, X_val, y_val, scale_pos_weight, experiment_id)
        except Exception as e:
            print(f"  [{cfg['name']}] FAILED: {e}")

    print("\n" + "=" * 60)
    print("Promoting best run → MLflow Model Registry")
    print("=" * 60)
    _promote_best(experiment_id)

    print(f"\nDone. Open MLflow UI:")
    print(f"  mlflow ui --backend-store-uri {ROOT / 'mlruns'} --port 5000")


if __name__ == "__main__":
    main()
