"""
Class-imbalance handling.

DASS severity classes are naturally imbalanced (community samples skew
towards 'Normal'/'Mild', 'Extremely Severe' is rare). We support two
complementary strategies:

1. class_weight='balanced' (used for every model that supports it) -- always
   applied, free, no synthetic data.
2. SMOTE oversampling of the TRAINING split only (never the test split, to
   avoid leaking synthetic neighbours of test points into training).
"""
import logging

import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    counts = y_train.value_counts()
    min_class_size = counts.min()
    # SMOTE needs k_neighbors < smallest class size
    k = max(1, min(5, min_class_size - 1))
    if min_class_size <= 1:
        logger.warning("A class has <=1 sample; skipping SMOTE for this split.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE: %s -> %s training rows. Class counts before: %s | after: %s",
        len(X_train), len(X_res), counts.to_dict(),
        pd.Series(y_res).value_counts().to_dict(),
    )
    return X_res, y_res
