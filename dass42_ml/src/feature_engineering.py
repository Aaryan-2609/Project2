"""
Feature selection with the leakage-safe modes described in config.FEATURE_MODE
and the README's "Leakage prevention" section.

Every mode returns (X, feature_names) for a GIVEN target scale, since
'cross_scale_items' and 'item_subset' modes are target-dependent (the
excluded columns differ per target).
"""
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _tipi_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-score the appropriate TIPI items and average into 5 Big-Five
    trait scores -- standard TIPI scoring (Gosling, Rentfrow & Swann, 2003)."""
    def r(x):
        return 8 - x  # reverse score on a 1-7 scale

    out = pd.DataFrame(index=df.index)
    out["trait_extraversion"] = (df["TIPI1"] + r(df["TIPI6"])) / 2
    out["trait_agreeableness"] = (r(df["TIPI2"]) + df["TIPI7"]) / 2
    out["trait_conscientiousness"] = (df["TIPI3"] + r(df["TIPI8"])) / 2
    out["trait_emotional_stability"] = (r(df["TIPI4"]) + df["TIPI9"]) / 2
    out["trait_openness"] = (df["TIPI5"] + r(df["TIPI10"])) / 2
    return out


def _vocab_score(df: pd.DataFrame) -> pd.Series:
    real_words = [c for c in config.VCL_COLS if c not in config.VCL_FAKE_WORDS]
    return df[real_words].sum(axis=1).rename("vocab_score")


def _base_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Personality + vocabulary + demographics + response-time behavior.
    Contains ZERO DASS item columns -- this is the leakage-safe core used by
    every feature mode."""
    parts = [_tipi_scores(df), _vocab_score(df).to_frame()]

    demo = df[[c for c in config.DEMOGRAPHIC_COLS if c in df.columns]].copy()
    # one-hot encode categorical demographics; keep age/familysize numeric
    numeric_demo_cols = ["age", "familysize"]
    cat_demo_cols = [c for c in demo.columns if c not in numeric_demo_cols]
    demo = pd.get_dummies(demo, columns=cat_demo_cols, prefix=cat_demo_cols, dummy_na=False)
    parts.append(demo)

    timing = df[[c for c in config.TIMING_COLS if c in df.columns]].copy()
    parts.append(timing)

    features = pd.concat(parts, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    return features


def build_features(df: pd.DataFrame, target_scale: str, mode: str = None) -> pd.DataFrame:
    """
    Build the feature matrix for `target_scale` ('Depression'|'Anxiety'|'Stress')
    according to `mode` (defaults to config.FEATURE_MODE).
    """
    mode = mode or config.FEATURE_MODE
    base = _base_metadata_features(df)

    if mode == "metadata_only":
        features = base

    elif mode == "cross_scale_items":
        other_scales = [s for s in config.TARGETS if s != target_scale]
        other_item_cols = [f"Q{i}A" for s in other_scales for i in config.DASS_ITEMS[s]]
        features = pd.concat([base, df[other_item_cols]], axis=1)

    elif mode == "item_subset":
        # Use the first 7 of the 14 same-scale items (a short-form screener)
        # to predict severity computed from the full 14 -- the held-out 7
        # items are NEVER included, so there is no direct leakage.
        items = config.DASS_ITEMS[target_scale]
        subset_cols = [f"Q{i}A" for i in items[: len(items) // 2]]
        features = pd.concat([base, df[subset_cols]], axis=1)

    else:
        raise ValueError(f"Unknown FEATURE_MODE: {mode}")

    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)
    logger.info("build_features[%s / %s]: %s features", target_scale, mode, features.shape[1])
    return features


def get_target(df: pd.DataFrame, target_scale: str) -> pd.Series:
    return df[f"{target_scale}_Severity"]
