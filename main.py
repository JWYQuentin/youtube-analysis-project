"""Train the refined YouTube subscriber prediction model.

This script loads the engineered channel-level features, trains a
HistGradientBoostingRegressor wrapped in a TransformedTargetRegressor (to model
``log1p(subs)``), evaluates the model, and persists the trained artifact along
with evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor


DEFAULT_DATA_PATH = Path("youtube_channels_with_features.csv")
DEFAULT_MODEL_PATH = Path("models/refined_subscriber_model.joblib")
DEFAULT_METRICS_PATH = Path("results/metrics.json")

FEATURE_COLUMNS: List[str] = [
    "views",
    "videos",
    "channel_age_years",
    "views_per_video",
]
TARGET_COLUMN = "subs"


@dataclass
class TrainingArtifacts:
    """Container for trained objects and metadata."""

    model: TransformedTargetRegressor
    metrics: Dict[str, float]
    feature_columns: List[str]


def _log1p_array(x: np.ndarray) -> np.ndarray:
    """Apply ``log1p`` element-wise while preserving array shape."""

    return np.log1p(np.clip(x, a_min=0, a_max=None))


def _inverse_log1p_array(x: np.ndarray) -> np.ndarray:
    """Apply inverse ``log1p`` (``expm1``) element-wise."""

    return np.expm1(x)


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used across training and inference."""

    log_features = ["views", "videos", "views_per_video"]
    linear_features = ["channel_age_years"]

    log_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log",
                FunctionTransformer(
                    _log1p_array,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    linear_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log", log_pipeline, log_features),
            ("linear", linear_pipeline, linear_features),
        ]
    )

    return preprocessor


def build_model(random_state: int = 42) -> TransformedTargetRegressor:
    """Create the full modeling pipeline wrapped with log-target transformation."""

    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_depth=8,
                    max_iter=500,
                    learning_rate=0.05,
                    l2_regularization=0.01,
                    min_samples_leaf=20,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=_log1p_array,
        inverse_func=_inverse_log1p_array,
        check_inverse=False,
    )

    return model


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the feature dataset and validate required columns."""

    df = pd.read_csv(path)

    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The dataset at {path} is missing required columns: {', '.join(missing_cols)}"
        )

    return df


def evaluate_model(
    model: TransformedTargetRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute evaluation metrics on the holdout set."""

    y_pred = np.clip(model.predict(X_test), a_min=0, a_max=None)

    log_true = np.log1p(y_test)
    log_pred = np.log1p(y_pred)

    metrics = {
        "r2_log": float(r2_score(log_true, log_pred)),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_test, y_pred))),
        "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
    }

    return metrics


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingArtifacts:
    """Train the refined model and compute evaluation metrics."""

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    return TrainingArtifacts(model=model, metrics=metrics, feature_columns=FEATURE_COLUMNS)


def save_model(model: TransformedTargetRegressor, path: Path) -> None:
    """Persist the trained model pipeline to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Persist evaluation metrics as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the refined subscriber model")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV file containing engineered channel features.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model artifact (joblib).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to save the evaluation metrics JSON file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to allocate to the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splitting and model reproducibility.",
    )

    return parser.parse_args()


def main() -> Tuple[TransformedTargetRegressor, Dict[str, float]]:
    args = parse_args()

    df = load_dataset(args.train_data)
    artifacts = train_model(df, test_size=args.test_size, random_state=args.random_state)

    save_model(artifacts.model, args.model_out)
    save_metrics(artifacts.metrics, args.metrics_out)

    print("Training complete. Metrics:")
    for key, value in artifacts.metrics.items():
        print(f"  {key}: {value:.4f}")

    return artifacts.model, artifacts.metrics


if __name__ == "__main__":
    main()
