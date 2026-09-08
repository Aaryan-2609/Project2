"""
Model explainability with SHAP (primary) and a LIME fallback for models SHAP
handles poorly (e.g. SVM with RBF kernel).
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config

logger = logging.getLogger(__name__)

TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}


def explain_with_shap(model, X_train: pd.DataFrame, X_sample: pd.DataFrame,
                       model_name: str, target_scale: str, clf_step_name: str = "clf") -> str:
    """
    Produces a global SHAP summary (beeswarm) plot of feature importance.
    Tree models get the fast TreeExplainer; everything else falls back to
    KernelExplainer on a small background sample (SHAP's model-agnostic but
    much slower explainer).
    """
    # Pipelines wrap the classifier as the 'clf' step + optional 'scaler'.
    estimator = model.named_steps[clf_step_name] if hasattr(model, "named_steps") else model
    if "scaler" in getattr(model, "named_steps", {}):
        X_train_t = model.named_steps["scaler"].transform(X_train)
        X_sample_t = model.named_steps["scaler"].transform(X_sample)
        X_train_t = pd.DataFrame(X_train_t, columns=X_train.columns)
        X_sample_t = pd.DataFrame(X_sample_t, columns=X_sample.columns)
    else:
        X_train_t, X_sample_t = X_train, X_sample

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample_t)
    else:
        background = shap.sample(X_train_t, min(100, len(X_train_t)), random_state=config.RANDOM_STATE)
        explainer = shap.KernelExplainer(estimator.predict_proba, background)
        shap_values = explainer.shap_values(X_sample_t.iloc[: min(50, len(X_sample_t))], nsamples=100)
        X_sample_t = X_sample_t.iloc[: min(50, len(X_sample_t))]

    # Multiclass: shap_values is a list of arrays (one per class) -- average
    # |SHAP| across classes for a single global importance plot.
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(mean_abs, X_sample_t, show=False, plot_type="bar")
    plt.title(f"SHAP global feature importance — {model_name} ({target_scale})")
    path = config.FIGURES_DIR / f"shap_{target_scale}_{model_name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary: %s", path)
    return str(path)


def explain_single_prediction_lime(model, X_train: pd.DataFrame, instance: pd.Series,
                                    class_names: list) -> str:
    """LIME explanation for one prediction -- useful in the API/dashboard for
    'why did the model predict this for THIS person' explanations."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train.values, feature_names=X_train.columns.tolist(),
        class_names=class_names, mode="classification",
        random_state=config.RANDOM_STATE,
    )
    exp = explainer.explain_instance(instance.values, model.predict_proba, num_features=10)
    return exp.as_html()
