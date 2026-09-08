"""
Load and lightly validate the raw OpenPsychometrics DASS-42 CSV.

The Kaggle export is tab- or comma-separated depending on the mirror, and
uses '-1' or blank strings for missing metadata -- we normalise both here.
"""
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_raw(path: Path = config.DATA_RAW) -> pd.DataFrame:
    """Load the raw DASS-42 CSV, auto-detecting the delimiter."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download it from Kaggle "
            f"(lucasgreenwell/depression-anxiety-stress-scales-responses) "
            f"and place it there as data.csv."
        )
    # sep=None + engine='python' autodetects comma vs tab, which the
    # different Kaggle mirrors of this dataset use inconsistently.
    df = pd.read_csv(path, sep=None, engine="python", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    logger.info("Loaded raw data: %s rows, %s columns", len(df), df.shape[1])
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if the 42 DASS answer columns aren't present -- everything
    downstream (scoring, labeling) depends on them."""
    missing = [c for c in config.ALL_Q_ANSWER_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected DASS item columns: {missing[:5]}... "
            f"({len(missing)} total). Check that the correct file was downloaded."
        )
