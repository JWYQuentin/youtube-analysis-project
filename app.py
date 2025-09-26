from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

from refined_model import DATA_CANDIDATES, TARGET_COLUMN, load_channels_dataframe

st.set_page_config(page_title="YouTube Subscriber Prediction – Demo App")
st.title("YouTube Subscriber Prediction – Demo App")


@st.cache_resource(show_spinner=False)
def load_model() -> Optional[object]:
    model_path = Path("models/refined_subscriber_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[dict]:
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return None


def load_dataframe(sample: Optional[str], uploaded) -> Optional[pd.DataFrame]:
    if sample:
        try:
            return load_channels_dataframe(Path(sample))
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Failed to load sample file '{sample}': {exc}")
            return None

    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Unable to read uploaded CSV: {exc}")
            return None
    return None


model = load_model()
if model is None:
    st.warning(
        "Model artifact not found. Please run `python refined_model.py` to train and save the model."
    )

available_samples = [path for path in DATA_CANDIDATES if Path(path).exists()]
selected_sample_label = None

if available_samples:
    selected_sample_label = st.selectbox(
        "Choose a built-in dataset",
        options=["None"] + available_samples,
        index=0,
    )
    selected_sample = (
        None if selected_sample_label == "None" else selected_sample_label
    )
else:
    st.info("No bundled CSV files were found in the repository.")
    selected_sample = None

uploaded_file = st.file_uploader("Or upload a CSV file", type="csv")

dataframe = load_dataframe(selected_sample, uploaded_file)

if dataframe is not None:
    st.subheader("Data preview")
    st.dataframe(dataframe.head())

metrics = load_metrics()
if metrics:
    st.subheader("Model metrics")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            st.metric(key.replace("_", " ").title(), f"{value:.4f}")
        else:
            st.metric(key.replace("_", " ").title(), str(value))

if st.button("Predict", type="primary"):
    if model is None:
        st.error("Model is not loaded. Train the model before making predictions.")
    elif dataframe is None:
        st.error("No data available for prediction.")
    else:
        features = dataframe.drop(columns=[TARGET_COLUMN], errors="ignore")
        try:
            predictions = model.predict(features)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Prediction failed: {exc}")
        else:
            results = dataframe.copy()
            results["predicted_subs"] = predictions
            st.success("Predictions generated!")
            st.dataframe(results.head())

            csv_buffer = StringIO()
            results.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download predictions",
                data=csv_buffer.getvalue(),
                file_name="subscriber_predictions.csv",
                mime="text/csv",
            )
else:
    st.caption("Click 'Predict' to generate subscriber predictions for the dataset above.")
