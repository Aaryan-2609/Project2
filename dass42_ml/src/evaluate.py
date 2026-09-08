"""
Held-out test-set evaluation: accuracy, precision/recall/F1 (macro, since
classes are imbalanced and every severity level matters equally), one-vs-rest
ROC-AUC, and confusion matrices.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

import config

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str, target_scale: str) -> dict:
    y_pred = model.predict(X_test)
    labels = [l for l in config.SEVERITY_ORDER if l in y_test.unique()]

    metrics = {
        "model": model_name,
        "target": target_scale,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=labels)
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_test_bin, y_proba, average="macro", multi_class="ovr"
            )
        except Exception as e:  # some folds may have too few samples of a class
            logger.warning("ROC-AUC failed for %s/%s: %s", model_name, target_scale, e)
            metrics["roc_auc_ovr_macro"] = np.nan
    else:
        metrics["roc_auc_ovr_macro"] = np.nan

    metrics["classification_report"] = classification_report(
        y_test, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str, target_scale: str) -> str:
    labels = [l for l in config.SEVERITY_ORDER if l in y_test.unique()]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — {target_scale} confusion matrix (row-normalized)")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"confusion_{target_scale}_{model_name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def summarize_results(all_metrics: list) -> pd.DataFrame:
    """Flatten metric dicts (minus the nested classification_report) into a
    leaderboard DataFrame, sorted by f1_macro within each target."""
    rows = [{k: v for k, v in m.items() if k != "classification_report"} for m in all_metrics]
    df = pd.DataFrame(rows)
    df = df.sort_values(["target", "f1_macro"], ascending=[True, False]).reset_index(drop=True)
    return df
