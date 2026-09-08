"""
Official DASS-42 scoring engine.

Reference: Lovibond, S.H. & Lovibond, P.F. (1995). Manual for the Depression
Anxiety Stress Scales. (2nd ed.) Sydney: Psychology Foundation.

IMPORTANT: DASS-42 raw subscale sums are used AS-IS for severity lookup.
Do NOT multiply by 2 -- that adjustment is only for the DASS-21 short form,
which is a common bug when people mix up 21- vs 42-item conventions.
"""
import numpy as np
import pandas as pd

import config


def _item_cols(scale: str) -> list[str]:
    return [f"Q{i}A" for i in config.DASS_ITEMS[scale]]


def compute_subscale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Depression_Score, Anxiety_Score, Stress_Score.

    Raw item responses are coded 1-4 in the Kaggle export; the official DASS
    key uses 0-3, so we subtract 1 before summing.
    """
    scores = pd.DataFrame(index=df.index)
    for scale in config.TARGETS:
        cols = _item_cols(scale)
        recoded = df[cols].sub(1)  # 1..4 -> 0..3
        if recoded.min().min() < 0 or recoded.max().max() > 3:
            raise ValueError(
                f"Unexpected item values for {scale}; expected raw 1-4 coding."
            )
        scores[f"{scale}_Score"] = recoded.sum(axis=1)
    return scores


def score_to_severity(score: int, scale: str) -> str:
    """Map a raw subscale score to its official severity label."""
    for label, (lo, hi) in config.SEVERITY_THRESHOLDS[scale].items():
        if lo <= score <= hi:
            return label
    raise ValueError(f"Score {score} out of range for scale {scale}")


def add_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_Score and *_Severity columns (the ML targets) to df."""
    scores = compute_subscale_scores(df)
    out = pd.concat([df, scores], axis=1)
    for scale in config.TARGETS:
        out[f"{scale}_Severity"] = out[f"{scale}_Score"].apply(
            lambda s, sc=scale: score_to_severity(s, sc)
        )
        out[f"{scale}_Severity"] = pd.Categorical(
            out[f"{scale}_Severity"], categories=config.SEVERITY_ORDER, ordered=True
        )
    return out
