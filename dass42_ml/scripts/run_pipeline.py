"""
End-to-end orchestration: load -> clean -> label -> EDA -> feature-build ->
train+tune 7 models x 3 targets -> evaluate -> explain (SHAP) -> save best
model per target.

Run from the project root:  python scripts/run_pipeline.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `import config`, `import src...`

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src import data_loader, evaluate, explain, feature_engineering, imbalance, preprocessing, scoring, train
from src.feature_engineering import _tipi_scores
from src import eda as eda_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== 1. Load raw data ===")
    df_raw = data_loader.load_raw()

    logger.info("=== 2. Clean (missing values, duplicates, outliers, low-quality responses) ===")
    df_clean = preprocessing.run_cleaning_pipeline(df_raw)

    logger.info("=== 3. Official DASS-42 scoring + severity labels ===")
    df_labeled = scoring.add_severity_labels(df_clean)
    df_labeled.to_parquet(config.DATA_PROCESSED_DIR / "labeled_data.parquet")

    logger.info("=== 4. EDA ===")
    tipi_features = _tipi_scores(df_labeled)
    eda_module.run_eda(df_raw, df_labeled, tipi_features)

    leaderboard_rows = []
    best_per_target = {}

    for target_scale in config.TARGETS:
        logger.info("=== TARGET: %s ===", target_scale)
        X = feature_engineering.build_features(df_labeled, target_scale)
        y = feature_engineering.get_target(df_labeled, target_scale)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
        )

        logger.info("--- Class balance before SMOTE: %s ---", y_train.value_counts().to_dict())
        X_train_bal, y_train_bal = imbalance.apply_smote(X_train, y_train)

        results = train.train_all_models(X_train_bal, y_train_bal)

        target_metrics = []
        for model_name, res in results.items():
            model = res["best_estimator"]
            m = evaluate.evaluate_model(model, X_test, y_test, model_name, target_scale)
            m["cv_best_f1_macro"] = res["cv_best_f1_macro"]
            m["best_params"] = res["best_params"]
            target_metrics.append(m)
            evaluate.plot_confusion_matrix(model, X_test, y_test, model_name, target_scale)

        leaderboard_rows.extend(target_metrics)

        best = max(target_metrics, key=lambda m: m["f1_macro"])
        best_model_name = best["model"]
        best_model = results[best_model_name]["best_estimator"]
        best_per_target[target_scale] = best_model_name
        logger.info("Best model for %s: %s (test f1_macro=%.4f)",
                    target_scale, best_model_name, best["f1_macro"])

        # SHAP explanation for the winning model
        try:
            explain.explain_with_shap(
                best_model, X_train, X_test.sample(min(200, len(X_test)), random_state=config.RANDOM_STATE),
                best_model_name, target_scale,
            )
        except Exception as e:
            logger.warning("SHAP explanation skipped for %s/%s: %s", best_model_name, target_scale, e)

        # Persist best model + the exact feature column order it expects
        joblib.dump(
            {"model": best_model, "feature_names": list(X.columns), "target": target_scale,
             "model_name": best_model_name, "feature_mode": config.FEATURE_MODE},
            config.MODELS_DIR / f"best_model_{target_scale}.joblib",
        )

    leaderboard = evaluate.summarize_results(leaderboard_rows)
    leaderboard.to_csv(config.MODELS_DIR / "leaderboard.csv", index=False)
    with open(config.MODELS_DIR / "best_models.json", "w") as f:
        json.dump(best_per_target, f, indent=2)

    logger.info("=== DONE. Leaderboard: ===\n%s", leaderboard[
        ["target", "model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
