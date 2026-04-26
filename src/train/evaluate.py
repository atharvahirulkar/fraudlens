"""
Post-training evaluation: threshold tuning, SHAP analysis, feature importance.

Run after train.py has promoted a 'Production' model to the registry:
    python -m src.train.evaluate

Outputs (printed + saved as MLflow artifacts on a separate 'evaluation' run):
    - Precision-recall curve with optimal threshold marked
    - SHAP beeswarm plot (top 20 features)
    - SHAP waterfall plot for a sample fraud case
    - Feature importance bar chart (gain-based)
    - Full classification report
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import shap
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REGISTRY_MODEL_NAME = "fraudlens-detector"
EXPERIMENT_NAME = "fraudlens-detection"


# - Load best model + data

def load_production_model():
    """Load the Production-stage model from the MLflow registry."""
    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
    model_uri = f"models:/{REGISTRY_MODEL_NAME}/Production"
    print(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def load_native_model():
    """
    Load the underlying XGBoost / LightGBM object (needed for SHAP TreeExplainer).
    Inspects the Production run's model flavor to choose the right loader.
    """
    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
    client = MlflowClient()

    versions = client.get_latest_versions(REGISTRY_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("No Production model found in registry. Run train.py first.")

    mv = versions[0]
    run_id = mv.run_id
    artifact_uri = f"runs:/{run_id}/model"

    # Determine model flavor
    run = client.get_run(run_id)
    model_type = run.data.params.get("model_type", "")

    if model_type == "xgboost":
        model = mlflow.xgboost.load_model(artifact_uri)
    elif model_type == "lightgbm":
        model = mlflow.lightgbm.load_model(artifact_uri)
    else:
        # Fallback: try both
        try:
            model = mlflow.xgboost.load_model(artifact_uri)
        except Exception:
            model = mlflow.lightgbm.load_model(artifact_uri)

    return model, run, mv


def load_val_data() -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the same 80/20 val split used during training."""
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    feature_cols = [c for c in df.columns if c != "isFraud"]
    X = df[feature_cols]
    y = df["isFraud"]
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_val.reset_index(drop=True), y_val.reset_index(drop=True)


# - Threshold tuning

def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 1.0,
) -> dict:
    """
    Find the probability cutoff that maximises F-beta on the validation set.
    Returns the threshold, precision, recall, and F1 at that point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # F-beta = (1+b²) * P*R / (b²*P + R)
    b2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(
            (b2 * precision + recall) > 0,
            (1 + b2) * precision * recall / (b2 * precision + recall),
            0.0,
        )

    best_idx = np.argmax(fbeta[:-1])  # last threshold has no corresponding fbeta
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall":    float(recall[best_idx]),
        "f1":        float(fbeta[best_idx]) if beta == 1.0 else float(f1_score(y_true, (y_prob >= thresholds[best_idx]).astype(int), zero_division=0)),
    }


# - Plots

def plot_precision_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_info: dict,
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#3498db", lw=2, label=f"AUC-PR = {auc_pr:.4f}")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (random) = {baseline:.4f}")
    ax.scatter(
        threshold_info["recall"],
        threshold_info["precision"],
        color="#e74c3c", s=120, zorder=5,
        label=f"Optimal threshold = {threshold_info['threshold']:.3f}\n(F1={threshold_info['f1']:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Production Model")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC-ROC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Production Model")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 20,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Beeswarm — Top 20 Features (Production Model)")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    sample_idx: int = 0,
) -> plt.Figure:
    """Waterfall plot for a single prediction (shows contribution of each feature)."""
    explanation = explainer(X_sample.iloc[[sample_idx]])
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Sample {sample_idx} (Production Model)")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 30) -> plt.Figure:
    """Gain-based feature importance bar chart."""
    try:
        # XGBoost
        imp = model.get_booster().get_score(importance_type="gain")
        imp_series = pd.Series(imp).reindex(feature_names).fillna(0)
    except AttributeError:
        # LightGBM
        imp_series = pd.Series(
            model.feature_importances_,
            index=feature_names,
        )

    top = imp_series.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(top.index, top.values, color="#3498db", alpha=0.8)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Features — Gain-Based Importance")
    fig.tight_layout()
    return fig


# - Full evaluation report

def full_evaluation(save_to_mlflow: bool = True) -> dict:
    """
    Run complete evaluation of the Production model on the held-out val set.
    Logs all plots and metrics to a new MLflow run tagged 'evaluation'.
    """
    mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")

    print("Loading Production model...")
    native_model, prod_run, mv = load_native_model()
    model_type = prod_run.data.params.get("model_type", "unknown")
    run_name = prod_run.data.tags.get("mlflow.runName", prod_run.info.run_id)

    print(f"  Model   : {REGISTRY_MODEL_NAME} v{mv.version} ({model_type})")
    print(f"  Source  : {run_name}")

    print("Loading validation data...")
    X_val, y_val = load_val_data()

    print("Scoring validation set...")
    y_prob = native_model.predict_proba(X_val)[:, 1]
    if isinstance(y_prob, list):
        y_prob = y_prob[1]
    y_prob = np.array(y_prob)

    # Threshold tuning
    threshold_info = tune_threshold(y_val.values, y_prob)
    threshold = threshold_info["threshold"]
    y_pred = (y_prob >= threshold).astype(int)

    auc_pr  = average_precision_score(y_val, y_prob)
    auc_roc = roc_auc_score(y_val, y_prob)
    f1      = f1_score(y_val, y_pred, zero_division=0)

    print(f"\nValidation metrics:")
    print(f"  AUC-PR         : {auc_pr:.4f}")
    print(f"  AUC-ROC        : {auc_roc:.4f}")
    print(f"  F1 (opt thr)   : {f1:.4f}")
    print(f"  Threshold      : {threshold:.4f}")
    print(f"  Precision@thr  : {threshold_info['precision']:.4f}")
    print(f"  Recall@thr     : {threshold_info['recall']:.4f}")
    print(f"\nClassification report (threshold = {threshold:.3f}):")
    print(classification_report(y_val, y_pred, target_names=["Non-fraud", "Fraud"]))

    # SHAP on subsample
    print("Computing SHAP values (2000 sample)...")
    shap_idx = np.random.default_rng(42).choice(len(X_val), size=min(2000, len(X_val)), replace=False)
    X_shap = X_val.iloc[shap_idx].reset_index(drop=True)
    explainer = shap.TreeExplainer(native_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Pick a fraud case for waterfall
    fraud_shap_indices = np.where(y_val.values[shap_idx] == 1)[0]
    waterfall_idx = int(fraud_shap_indices[0]) if len(fraud_shap_indices) > 0 else 0

    # Generate all plots
    fig_pr = plot_precision_recall(y_val.values, y_prob, threshold_info)
    fig_roc = plot_roc(y_val.values, y_prob)
    fig_bee = plot_shap_beeswarm(shap_values, X_shap)
    fig_wf  = plot_shap_waterfall(explainer, X_shap, waterfall_idx)
    fig_imp = plot_feature_importance(native_model, X_val.columns.tolist())

    if save_to_mlflow:
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"evaluation-{run_name}", experiment_id=experiment.experiment_id) as eval_run:
            mlflow.set_tag("evaluation_of_run", prod_run.info.run_id)
            mlflow.set_tag("registry_version", str(mv.version))

            mlflow.log_metrics({
                "auc_pr":         auc_pr,
                "auc_roc":        auc_roc,
                "f1":             f1,
                "best_threshold": threshold,
                "precision_at_threshold": threshold_info["precision"],
                "recall_at_threshold":    threshold_info["recall"],
            })

            mlflow.log_figure(fig_pr,  "precision_recall_curve.png")
            mlflow.log_figure(fig_roc, "roc_curve.png")
            mlflow.log_figure(fig_bee, "shap_beeswarm.png")
            mlflow.log_figure(fig_wf,  "shap_waterfall.png")
            mlflow.log_figure(fig_imp, "feature_importance.png")

            print(f"\nEvaluation run logged: {eval_run.info.run_id}")

    plt.close("all")

    return {
        "auc_pr":    auc_pr,
        "auc_roc":   auc_roc,
        "f1":        f1,
        "threshold": threshold,
        "model_version": mv.version,
    }


if __name__ == "__main__":
    full_evaluation()
