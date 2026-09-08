"""
Cleaning: missing values, duplicates, outliers, and low-quality-response
filtering. Every function is pure (returns a new/copied DataFrame) and logs
how many rows/cols it affected, so the pipeline prints an auditable trail.
"""
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Exact duplicate rows across all 42 items + demographics are almost
    # certainly re-submissions, not genuinely identical people.
    df = df.drop_duplicates(subset=config.ALL_Q_ANSWER_COLS + config.DEMOGRAPHIC_COLS)
    logger.info("drop_duplicates: removed %s rows", before - len(df))
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - Rows missing ANY of the 42 DASS items are dropped: we cannot compute a
      valid official score with partial items, and imputing psychometric
      item responses would fabricate the label itself.
    - Metadata: many OpenPsychometrics fields use sentinel codes (0 or -1)
      for "prefer not to say" / not-applicable. We convert those sentinels to
      NaN, then impute (median for numeric, most-frequent/'Unknown' for
      categorical) rather than dropping rows, since these are FEATURE
      columns, not the label.
    """
    before = len(df)
    df = df.dropna(subset=config.ALL_Q_ANSWER_COLS)
    df = df[(df[config.ALL_Q_ANSWER_COLS] >= 1).all(axis=1) &
            (df[config.ALL_Q_ANSWER_COLS] <= 4).all(axis=1)]
    logger.info("handle_missing_values: dropped %s rows with incomplete/invalid DASS items",
                before - len(df))

    # Sentinel -> NaN for demographic/categorical fields that use 0 as "NA"
    sentinel_zero_cols = ["education", "religion", "orientation", "race"]
    for c in sentinel_zero_cols:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    numeric_meta = ["age", "familysize"] + config.TIMING_COLS
    for c in numeric_meta:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    categorical_meta = [c for c in config.DEMOGRAPHIC_COLS if c not in numeric_meta]
    for c in categorical_meta:
        if c in df.columns:
            mode = df[c].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else -1
            df[c] = df[c].fillna(fill_value)

    for c in config.TIPI_COLS + config.VCL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    return df.reset_index(drop=True)


def filter_low_quality_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove respondents who almost certainly weren't paying attention, which
    otherwise injects pure noise into both features and labels:
      1. VCL attention-check items (VCL6, VCL9 are fake words) claimed as
         'known' -> careless responder.
      2. Straight-lining: identical answer to all 42 DASS items (e.g. all 1s)
         is a known low-effort response pattern.
      3. Implausible total test time: < 60 seconds to answer 42 items (< ~1.4
         sec/item) is physically implausible for genuine reading.
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)

    fake_word_cols = [c for c in config.VCL_FAKE_WORDS if c in df.columns]
    if fake_word_cols:
        mask &= (df[fake_word_cols] == 0).all(axis=1)  # 0 = "did not check this word"

    mask &= df[config.ALL_Q_ANSWER_COLS].nunique(axis=1) > 1

    if "testelapse" in df.columns:
        mask &= df["testelapse"] >= 60

    df = df[mask]
    logger.info("filter_low_quality_responses: removed %s low-quality rows", before - len(df))
    return df.reset_index(drop=True)


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap (winsorize) implausible values in free-entry numeric fields instead
    of dropping the whole row, since only one column is usually the culprit.
    age: plausible self-report range for an anonymous web survey is 13-100.
    familysize: capped at 20 siblings (values like 99/999 appear in the raw data).
    testelapse/surveyelapse: capped at 99th percentile to remove people who
      left the tab open for hours/days without answering.
    """
    df = df.copy()
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=13, upper=100)
    if "familysize" in df.columns:
        df["familysize"] = df["familysize"].clip(upper=20)
    for c in config.TIMING_COLS:
        if c in df.columns:
            cap = df[c].quantile(0.99)
            df[c] = df[c].clip(upper=cap)
    logger.info("cap_outliers: winsorized age/familysize/timing columns")
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = filter_low_quality_responses(df)
    df = cap_outliers(df)
    logger.info("run_cleaning_pipeline: final shape %s", df.shape)
    return df
