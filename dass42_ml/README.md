# DASS-42 Mental State Classifier

Production-style ML system that predicts **Depression**, **Anxiety**, and **Stress** severity
(Normal / Mild / Moderate / Severe / Extremely Severe) from the OpenPsychometrics
**DASS-42** Kaggle dataset:
https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses

---

## 1. Dataset — column-by-column

The raw CSV (~39,775 rows) from OpenPsychometrics has these column groups:

| Group | Columns | Meaning |
|---|---|---|
| **DASS items** | `Q1A ... Q42A` | Answer (1–4) to each of the 42 DASS statements. 1 = "Did not apply to me at all", 4 = "Applied to me very much". These 42 items are the **only** columns the official DASS scoring key uses. |
| **Item metadata** | `Q1I ... Q42I` | Position (order index) in which that question was shown to the respondent (randomized per respondent). Not psychometric signal — pure UI/order metadata. |
| | `Q1E ... Q42E` | Milliseconds spent on that specific item ("response/elapsed time"). Behavioral metadata: can proxy engagement/attention but is *not* part of the official score. |
| **Personality (TIPI)** | `TIPI1 ... TIPI10` | Ten-Item Personality Inventory (1–7 Likert), 2 items per Big-Five trait (Extraversion, Agreeableness, Conscientiousness, Emotional Stability, Openness), one item of each pair reverse-scored. Genuinely independent predictor of mental state — safe, useful feature. |
| **Vocabulary check** | `VCL1 ... VCL16` | Word Checklist — real and fake words the respondent claims to know. `VCL6` and `VCL9` are deliberately fake words (infrequency/attention-check items) used to flag careless/inattentive responders. |
| **Demographics** | `education` (1=less than HS…4=graduate degree, 0=NA) | | |
| | `urban` (1=rural,2=suburban,3=urban) | | |
| | `gender` (1=male,2=female,3=other) | | |
| | `engnat` (1=yes,2=no) — English as native language | | |
| | `age` | self-reported, contains extreme outliers (e.g. 3, 900+) that must be cleaned | | |
| | `hand` (1=right,2=left,3=both) | | |
| | `religion` (1–12 categorical, 0=NA) | | |
| | `orientation` (1–5 categorical) | | |
| | `race` (1–6 categorical) | | |
| | `voted` (1=yes,2=no) — voted in last election | | |
| | `married` (1=never,2=currently,3=previously) | | |
| | `familysize` | number of siblings, contains implausible outliers | | |
| | `major` | free-text field, extremely high-cardinality, mostly unusable raw | | |
| **Survey behavior** | `screensize` (1=device≥900px,2=smaller) | | |
| | `uniquenetworklocation` (1=yes,2=no) — flags duplicate submissions from same network | | |
| | `country` | ISO country code | | |
| | `source` | how respondent found the survey (categorical code) | | |
| | `introelapse` | seconds spent on the intro page | | |
| | `testelapse` | total seconds spent answering the 42 DASS items | | |
| | `surveyelapse` | seconds spent on the demographic questions | | |

**Key modeling implication:** the *label we predict* is a deterministic function of
`Q1A...Q42A`. This is the single most important design decision in the whole
project — see §4 "Leakage Prevention" below before touching any model code.

---

## 2. Official DASS-42 scoring (implemented in `src/scoring.py`)

The 42 items split into three 14-item subscales (Lovibond & Lovibond, 1995).
Unlike the DASS-21 short form, **DASS-42 raw sums are used directly — no ×2
multiplier**. Response coding: `1→0, 2→1, 3→2, 4→3` (the raw Q*A columns are 1–4,
the official key is 0–3), then summed per subscale.

```
Depression items: 3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42
Anxiety items:     2, 4,  7,  9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41
Stress items:      1, 6,  8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39
```

Official severity cut-offs (raw DASS-42 scores):

| Severity | Depression | Anxiety | Stress |
|---|---|---|---|
| Normal | 0–9 | 0–7 | 0–14 |
| Mild | 10–13 | 8–9 | 15–18 |
| Moderate | 14–20 | 10–14 | 19–25 |
| Severe | 21–27 | 15–19 | 26–33 |
| Extremely Severe | 28+ | 20+ | 34+ |

---

## 3. Project layout

```
dass42_ml/
├── data/
│   ├── raw/                 # put data.csv here (downloaded from Kaggle)
│   └── processed/           # cleaned + labeled parquet files written here
├── src/
│   ├── data_loader.py       # load + validate raw CSV
│   ├── scoring.py           # official DASS-42 keys, scoring, severity labels
│   ├── preprocessing.py     # missing values, duplicates, outliers, attention-check filter
│   ├── feature_engineering.py # feature selection + LEAKAGE-SAFE feature modes
│   ├── imbalance.py         # class-imbalance handling (SMOTE / class_weight)
│   ├── eda.py                # EDA plots -> reports/figures/
│   ├── train.py               # trains + tunes all 7 models x 3 targets, MLflow-style logging
│   ├── evaluate.py            # metrics, confusion matrices, ROC-AUC
│   └── explain.py             # SHAP explainability
├── models/                    # saved best model(s) (.pkl / .joblib) + metadata.json
├── api/main.py                # FastAPI inference service
├── dashboard/app.py            # Streamlit dashboard
├── scripts/run_pipeline.py     # end-to-end orchestration (one command)
├── tests/test_scoring.py       # unit tests for the DASS scoring logic
├── config.py
└── requirements.txt
```

## 4. Leakage prevention — the most important decision in this project

The label for each target (e.g. Depression severity) is **literally computed by
summing 14 of the 42 feature columns.** If you feed all 42 `Q*A` items into a
model and ask it to predict Depression severity, the model doesn't learn
anything — it just re-derives a sum it was already given. That's not
"data leakage" in the classic train/test sense, it's *label-is-a-function-of-features*
leakage, and it's worse: it produces a 99.9%-accuracy model that is
scientifically meaningless and cannot be used on anyone who hasn't already
filled in the full DASS-42 (in which case you don't need a model — just sum
the items).

This pipeline supports two leakage-safe feature modes, set in `config.py`
(`FEATURE_MODE`):

- **`metadata_only` (default, recommended)** — features are TIPI personality
  scores, vocabulary-checklist score, demographics, and response-time
  behavior. **Zero DASS items are used as features.** This is the only mode
  that answers a genuinely useful question: *"can we screen for likely
  depression/anxiety/stress severity from personality + demographics alone,
  before someone takes the full test?"* Expect moderate (not perfect)
  performance — that's the honest, realistic result.
- **`cross_scale_items`** — to predict Depression severity, you may use the
  Anxiety-scale and Stress-scale items (28 items) plus metadata, but **never**
  the 14 Depression items themselves (and symmetrically for the other two
  targets). This tests whether one subscale's severity can be inferred from
  the *other two* subscales — a legitimate psychometric question (comorbidity
  structure) — while still guaranteeing zero direct leakage for the target
  being predicted.

`item_subset` mode (use e.g. 7 of the 14 items of the *same* scale to predict
the severity computed from all 14 — i.e. building a validated short-form
screener) is included as an opt-in third mode for completeness, since this is
a real, published use case (short-form DASS validation), but it is off by
default because it's easy to misuse as accidental leakage if the held-out
item set isn't disciplined.

Additionally: `uniquenetworklocation`, `Q*I` (item order) and `major` are
dropped — they encode data-collection artifacts, not psychology.

## 5. How to run

```bash
pip install -r requirements.txt
# Download the dataset from Kaggle and place it at data/raw/data.csv
# (kaggle.json credentials required: https://www.kaggle.com/docs/api)
kaggle datasets download -d lucasgreenwell/depression-anxiety-stress-scales-responses -p data/raw --unzip

python scripts/run_pipeline.py          # cleans, labels, trains, tunes, evaluates, explains, saves best model
uvicorn api.main:app --reload --port 8000
streamlit run dashboard/app.py
```
