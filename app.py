"""Streamlit app for predicting YouTube channel subscribers."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/refined_subscriber_model.joblib")
FEATURE_COLUMNS = ["views", "videos", "channel_age_years", "views_per_video"]


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python main.py` to train and save the model first."
        )
    return joblib.load(path)


@st.cache_resource
def get_model():
    return load_model()


def make_single_prediction(model, features: dict) -> float:
    data = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = model.predict(data)[0]
    return float(np.clip(prediction, a_min=0, a_max=None))


def predict_from_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_COLUMNS)
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            "Uploaded file is missing required columns: " + ", ".join(sorted(missing))
        )
    predictions = model.predict(df[FEATURE_COLUMNS])
    df = df.copy()
    df["predicted_subs"] = np.clip(predictions, a_min=0, a_max=None)
    return df


def sidebar_inputs() -> dict:
    st.sidebar.header("Channel metrics")

    views = st.sidebar.number_input(
        "Total channel views",
        min_value=0.0,
        value=5_000_000.0,
        step=100_000.0,
        help="Total lifetime views across all channel videos.",
    )
    videos = st.sidebar.number_input(
        "Number of uploaded videos",
        min_value=1.0,
        value=250.0,
        step=1.0,
        help="Published videos on the channel.",
    )
    age_years = st.sidebar.number_input(
        "Channel age (years)",
        min_value=0.1,
        value=6.0,
        step=0.1,
        help="Age of the channel in years.",
    )

    views_per_video = views / videos if videos else 0.0

    st.sidebar.caption(
        "`views_per_video` is derived automatically as views divided by videos."
    )

    return {
        "views": views,
        "videos": videos,
        "channel_age_years": age_years,
        "views_per_video": views_per_video,
    }


def render_single_prediction(model) -> None:
    st.subheader("Estimate subscribers for a single channel")
    features = sidebar_inputs()

    if st.button("Predict subscribers", type="primary"):
        prediction = make_single_prediction(model, features)
        st.metric(
            "Predicted subscribers",
            f"{prediction:,.0f}",
            help="Estimated subscriber count using the refined gradient boosting model.",
        )

        st.write("### Model inputs")
        st.write(pd.DataFrame([features]))


def render_batch_prediction(model) -> None:
    st.subheader("Batch predictions from CSV")
    st.write(
        "Upload a CSV file with the columns `views`, `videos`, `channel_age_years`, "
        "and `views_per_video` to score multiple channels at once."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            predictions = predict_from_dataframe(model, df)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Could not generate predictions: {exc}")
        else:
            st.success("Predictions generated successfully")
            st.dataframe(predictions.head(20))

            csv_bytes = predictions.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions",
                data=csv_bytes,
                file_name="youtube_subscriber_predictions.csv",
                mime="text/csv",
            )


def main() -> None:
    st.set_page_config(
        page_title="YouTube Subscriber Predictor",
        page_icon="📈",
        layout="wide",
    )
    st.title("YouTube Subscriber Predictor")
    st.write(
        "This app uses a refined gradient boosting model trained on channel-level "
        "engagement metrics to estimate subscriber counts. Provide your channel's "
        "statistics or upload a CSV to get predictions."
    )

    try:
        model = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.expander("Model details", expanded=False):
        st.write(
            "The model is a `HistGradientBoostingRegressor` wrapped in a transformed "
            "target pipeline to model `log1p(subscribers)`. It uses normalized `views`, "
            "`videos`, `channel_age_years`, and `views_per_video` as predictors."
        )

    render_single_prediction(model)
    st.divider()
    render_batch_prediction(model)


if __name__ == "__main__":
    main()
