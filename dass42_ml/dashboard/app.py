"""
Streamlit dashboard for the DASS-42 Mental State Classifier.

Two modes:
  - "Try it yourself": fills in the TIPI/demographic form, calls the FastAPI
    /predict endpoint, shows severity + probability bars for all 3 targets.
  - "Model performance": reads models/leaderboard.csv produced by
    scripts/run_pipeline.py and shows a comparison of all 7 models x 3 targets.

Run:  streamlit run dashboard/app.py
(Start the API first:  uvicorn api.main:app --port 8000)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import streamlit as st

import config

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DASS-42 Mental State Classifier", layout="wide")
st.title("🧠 DASS-42 Mental State Classifier")
st.caption(
    "Educational/research demo — predicts likely Depression, Anxiety and Stress "
    "severity bands from personality + demographic signals. **Not a diagnostic tool.**"
)

page = st.sidebar.radio("View", ["Try it yourself", "Model performance"])

SEVERITY_COLORS = {
    "Normal": "#2ecc71", "Mild": "#f1c40f", "Moderate": "#e67e22",
    "Severe": "#e74c3c", "Extremely Severe": "#8e44ad",
}

if page == "Try it yourself":
    st.subheader("Enter your info")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Big-Five personality (1=disagree strongly, 7=agree strongly)**")
        tipi_labels = [
            "Extraverted, enthusiastic", "Critical, quarrelsome", "Dependable, self-disciplined",
            "Anxious, easily upset", "Open to new experiences, complex", "Reserved, quiet",
            "Sympathetic, warm", "Disorganized, careless", "Calm, emotionally stable",
            "Conventional, uncreative",
        ]
        tipi_vals = [st.slider(f"I see myself as: {lab}", 1, 7, 4, key=f"tipi{i+1}")
                     for i, lab in enumerate(tipi_labels)]

    with col2:
        st.markdown("**Vocabulary & demographics**")
        vocab_score = st.slider("Vocabulary score (real words recognized, 0-14)", 0, 14, 7)
        age = st.number_input("Age", 13, 100, 25)
        gender = st.selectbox("Gender", [1, 2, 3], format_func=lambda x: {1: "Male", 2: "Female", 3: "Other"}[x])
        education = st.selectbox("Education", [1, 2, 3, 4],
                                  format_func=lambda x: {1: "< High school", 2: "High school", 3: "University", 4: "Graduate degree"}[x])
        married = st.selectbox("Marital status", [1, 2, 3], format_func=lambda x: {1: "Never married", 2: "Currently married", 3: "Previously married"}[x])
        familysize = st.number_input("Number of siblings", 0, 20, 1)

    with col3:
        st.markdown("**Other**")
        urban = st.selectbox("Living environment", [1, 2, 3], format_func=lambda x: {1: "Rural", 2: "Suburban", 3: "Urban"}[x])
        engnat = st.selectbox("English native language?", [1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x])
        hand = st.selectbox("Handedness", [1, 2, 3], format_func=lambda x: {1: "Right", 2: "Left", 3: "Both"}[x])
        religion = st.number_input("Religion code (1-12)", 1, 12, 1)
        orientation = st.number_input("Orientation code (1-5)", 1, 5, 1)
        race = st.number_input("Race code (1-6)", 1, 6, 1)
        voted = st.selectbox("Voted in last election?", [1, 2], format_func=lambda x: {1: "Yes", 2: "No"}[x])

    if st.button("Predict", type="primary"):
        payload = {
            "tipi": {f"tipi{i+1}": v for i, v in enumerate(tipi_vals)},
            "vocab_score": vocab_score, "education": education, "urban": urban,
            "gender": gender, "engnat": engnat, "age": age, "hand": hand,
            "religion": religion, "orientation": orientation, "race": race,
            "voted": voted, "married": married, "familysize": familysize,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"Could not reach the API at {API_URL}. Is `uvicorn api.main:app` running? ({e})")
        else:
            cols = st.columns(3)
            for col, key, label in zip(cols, ["depression", "anxiety", "stress"],
                                        ["Depression", "Anxiety", "Stress"]):
                with col:
                    pred = result[key]
                    st.metric(label, pred["severity"])
                    probs = pd.Series(pred["probabilities"]).reindex(config.SEVERITY_ORDER)
                    st.bar_chart(probs)
                    st.caption(f"Model: {pred['model_used']}")
            st.info(result["disclaimer"])

else:
    st.subheader("Model comparison leaderboard")
    lb_path = config.MODELS_DIR / "leaderboard.csv"
    if not lb_path.exists():
        st.warning("No leaderboard found yet — run `python scripts/run_pipeline.py` first.")
    else:
        lb = pd.read_csv(lb_path)
        target_pick = st.selectbox("Target", config.TARGETS)
        sub = lb[lb["target"] == target_pick].sort_values("f1_macro", ascending=False)
        st.dataframe(
            sub[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]]
            .set_index("model").style.format("{:.3f}"),
            use_container_width=True,
        )
        st.bar_chart(sub.set_index("model")["f1_macro"])

        fig_dir = config.FIGURES_DIR
        best_model = sub.iloc[0]["model"]
        cm_path = fig_dir / f"confusion_{target_pick}_{best_model}.png"
        if cm_path.exists():
            st.image(str(cm_path), caption=f"Confusion matrix — best model ({best_model})")
        shap_path = fig_dir / f"shap_{target_pick}_{best_model}.png"
        if shap_path.exists():
            st.image(str(shap_path), caption=f"SHAP feature importance — {best_model}")
