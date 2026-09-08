"""
Central configuration for the DASS-42 ML pipeline.
All paths, constants and the key leakage-prevention switch live here so the
rest of the codebase never hard-codes them.
"""
from pathlib import Path

# ------------------------------------------------------------------ paths --
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "data.csv"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

for d in (DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- randomness --
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# --------------------------------------------------------- feature design --
# "metadata_only"      -> TIPI + VCL + demographics + timing only (RECOMMENDED, zero leakage)
# "cross_scale_items"  -> other two scales' items + metadata (no same-scale items)
# "item_subset"        -> 7-of-14 same-scale items -> predict full-14 severity (short-form screener)
FEATURE_MODE = "metadata_only"

TARGETS = ["Depression", "Anxiety", "Stress"]

SEVERITY_ORDER = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

# ------------------------------------------------------- DASS-42 item keys --
DASS_ITEMS = {
    "Depression": [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
    "Anxiety":    [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
    "Stress":     [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39],
}

# Official raw DASS-42 severity cut-points: (min_inclusive, max_inclusive)
SEVERITY_THRESHOLDS = {
    "Depression": {
        "Normal": (0, 9), "Mild": (10, 13), "Moderate": (14, 20),
        "Severe": (21, 27), "Extremely Severe": (28, 42),
    },
    "Anxiety": {
        "Normal": (0, 7), "Mild": (8, 9), "Moderate": (10, 14),
        "Severe": (15, 19), "Extremely Severe": (20, 42),
    },
    "Stress": {
        "Normal": (0, 14), "Mild": (15, 18), "Moderate": (19, 25),
        "Severe": (26, 33), "Extremely Severe": (34, 42),
    },
}

# ---------------------------------------------------- columns to drop -------
# Pure data-collection artifacts, never psychological signal.
ALWAYS_DROP_COLS = ["uniquenetworklocation", "major", "screensize", "source"]

TIPI_COLS = [f"TIPI{i}" for i in range(1, 11)]
VCL_COLS = [f"VCL{i}" for i in range(1, 17)]
VCL_FAKE_WORDS = ["VCL6", "VCL9"]  # attention-check items, excluded from vocab score

DEMOGRAPHIC_COLS = [
    "education", "urban", "gender", "engnat", "age", "hand",
    "religion", "orientation", "race", "voted", "married", "familysize",
]

TIMING_COLS = ["introelapse", "testelapse", "surveyelapse"]

ALL_Q_ANSWER_COLS = [f"Q{i}A" for i in range(1, 43)]
ALL_Q_TIME_COLS = [f"Q{i}E" for i in range(1, 43)]
ALL_Q_ORDER_COLS = [f"Q{i}I" for i in range(1, 43)]
