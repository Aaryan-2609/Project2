"""
Trains and hyperparameter-tunes 7 model families for each of the 3 targets
(Depression / Anxiety / Stress), using stratified k-fold CV + RandomizedSearchCV.

Returns a results dict consumed by evaluate.py and used to pick + persist the
overall best model per target.
"""
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_distributions: dict
    needs_scaling: bool = False
    n_iter: int = 12


def _model_zoo(n_classes: int, random_state: int = config.RANDOM_STATE) -> list:
    """Model + search-space definitions. n_iter kept modest (10-15) so the
    full 7-models x 3-targets sweep finishes in reasonable time; increase for
    a more exhaustive search once the pipeline is validated end-to-end."""
    return [
        ModelSpec(
            "LogisticRegression",
            LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=random_state, multi_class="multinomial"),
            {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "saga"]},
            needs_scaling=True,
        ),
        ModelSpec(
            "RandomForest",
            RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [None, 8, 16, 24],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        ModelSpec(
            "XGBoost",
            XGBClassifier(objective="multi:softprob", num_class=n_classes,
                           eval_metric="mlogloss", random_state=random_state, n_jobs=-1,
                           tree_method="hist"),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
            },
        ),
        ModelSpec(
            "LightGBM",
            LGBMClassifier(objective="multiclass", num_class=n_classes,
                            class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1),
            {
                "n_estimators": [200, 400, 600],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
            },
        ),
        ModelSpec(
            "CatBoost",
            CatBoostClassifier(loss_function="MultiClass", auto_class_weights="Balanced",
                                random_state=random_state, verbose=False),
            {
                "iterations": [300, 500],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5],
            },
            n_iter=8,
        ),
        ModelSpec(
            "SVM",
            SVC(probability=True, class_weight="balanced", random_state=random_state),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
            needs_scaling=True,
            n_iter=8,
        ),
        ModelSpec(
            "NeuralNetwork",
            MLPClassifier(max_iter=500, early_stopping=True, random_state=random_state),
            {
                "hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 5e-4],
            },
            needs_scaling=True,
            n_iter=8,
        ),
    ]


def train_all_models(X_train, y_train, random_state: int = config.RANDOM_STATE) -> dict:
    """
    Runs stratified-CV RandomizedSearchCV for every model in the zoo.
    Returns {model_name: {"best_estimator": Pipeline, "best_params": dict,
                           "cv_best_score": float, "fit_seconds": float}}.
    """
    n_classes = y_train.nunique()
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=random_state)
    results = {}

    for spec in _model_zoo(n_classes, random_state):
        t0 = time.time()
        steps = []
        if spec.needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", spec.estimator))
        pipe = Pipeline(steps)

        param_dist = {f"clf__{k}": v for k, v in spec.param_distributions.items()}

        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=spec.n_iter,
            scoring="f1_macro", cv=cv, random_state=random_state, n_jobs=-1, refit=True,
        )
        logger.info("Tuning %s ...", spec.name)
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        results[spec.name] = {
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_best_f1_macro": search.best_score_,
            "fit_seconds": elapsed,
        }
        logger.info("%s: best CV f1_macro=%.4f (%.1fs)", spec.name, search.best_score_, elapsed)

    return results
