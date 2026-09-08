"""
Unit tests for the official DASS-42 scoring logic. Run with: pytest tests/
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from src import scoring


def _make_row(all_value: int) -> pd.DataFrame:
    """Build a single-row DataFrame where every Q*A item = all_value (1-4)."""
    data = {f"Q{i}A": all_value for i in range(1, 43)}
    return pd.DataFrame([data])


def test_all_minimum_answers_yield_normal_everywhere():
    df = _make_row(1)  # recoded to 0 for every item -> subscale scores all 0
    scored = scoring.add_severity_labels(df)
    for scale in config.TARGETS:
        assert scored.loc[0, f"{scale}_Score"] == 0
        assert scored.loc[0, f"{scale}_Severity"] == "Normal"


def test_all_maximum_answers_yield_extremely_severe_everywhere():
    df = _make_row(4)  # recoded to 3 for every item -> subscale score = 14*3 = 42
    scored = scoring.add_severity_labels(df)
    for scale in config.TARGETS:
        assert scored.loc[0, f"{scale}_Score"] == 42
        assert scored.loc[0, f"{scale}_Severity"] == "Extremely Severe"


@pytest.mark.parametrize("scale,score,expected", [
    ("Depression", 9, "Normal"),
    ("Depression", 10, "Mild"),
    ("Depression", 13, "Mild"),
    ("Depression", 14, "Moderate"),
    ("Depression", 27, "Severe"),
    ("Depression", 28, "Extremely Severe"),
    ("Anxiety", 7, "Normal"),
    ("Anxiety", 8, "Mild"),
    ("Anxiety", 20, "Extremely Severe"),
    ("Stress", 14, "Normal"),
    ("Stress", 15, "Mild"),
    ("Stress", 34, "Extremely Severe"),
])
def test_boundary_thresholds(scale, score, expected):
    assert scoring.score_to_severity(score, scale) == expected


def test_item_key_counts_are_14_per_scale_and_disjoint():
    all_items = []
    for scale, items in config.DASS_ITEMS.items():
        assert len(items) == 14
        all_items.extend(items)
    assert sorted(all_items) == list(range(1, 43))  # covers all 42 items exactly once
