from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_CANDIDATES: Tuple[str, ...] = (
    "youtube_channels_with_features.csv",
    "YouTube_Combined_Cleaned (1).csv",
    "sample_youtube_channels.csv",
)
TARGET_COLUMN = "subs"
RANDOM_STATE = 42


def _read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def _merge_fragmented_rows(lines: List[str]) -> Tuple[List[str], List[str]]:
    if not lines:
        raise ValueError("The provided dataset is empty.")

    header = lines[0]
    expected_columns = header.count(",") + 1
    rows: List[str] = []
    current_parts: List[str] = []

    for raw_line in lines[1:]:
        if not raw_line:
            continue

        current_parts.extend(raw_line.split(","))

        while len(current_parts) >= expected_columns:
            row = current_parts[:expected_columns]
            rows.append(",".join(row))
            current_parts = current_parts[expected_columns:]

    if current_parts:
        padded = current_parts + [""] * (expected_columns - len(current_parts))
        rows.append(",".join(padded[:expected_columns]))

    return header.split(","), rows


def load_channels_dataframe(path: Path) -> pd.DataFrame:
    lines = _read_lines(path)
    columns, rows = _merge_fragmented_rows(lines)
    data: List[List[str]] = [row.split(",") for row in rows if row.strip()]
    df = pd.DataFrame(data, columns=columns)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace("\"", "", regex=False).str.strip()
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() > 0:
                df[col] = numeric_series

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        df["published_year"] = df["published"].dt.year
        df["published_month"] = df["published"].dt.month
        df["published_day"] = df["published"].dt.day
        df = df.drop(columns=["published"])

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def find_training_data() -> Tuple[Path, pd.DataFrame]:
    for candidate in DATA_CANDIDATES:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            df = load_channels_dataframe(candidate_path)
            if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].notna().any():
                return candidate_path, df
    raise FileNotFoundError(
        "Could not locate a dataset containing the 'subs' column."
    )


def build_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No features available to train the model.")

    preprocessor = ColumnTransformer(transformers=transformers)

    model = RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])

    feature_df = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = feature_df.select_dtypes(exclude=["object"]).columns.tolist()

    if "channel_ID" in categorical_columns:
        categorical_columns.remove("channel_ID")
    if "channel_title" in categorical_columns:
        categorical_columns.remove("channel_title")

    if feature_df.shape[0] < 5:
        X_train, X_test, y_train, y_test = feature_df, feature_df, df[TARGET_COLUMN], df[TARGET_COLUMN]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df,
            df[TARGET_COLUMN],
            test_size=min(0.2, max(1, int(0.2 * len(df))) / len(df)),
            random_state=RANDOM_STATE,
        )

    model = build_pipeline(numeric_columns, categorical_columns)
    model.fit(X_train, y_train)

    metrics: dict = {}
    if not X_test.empty:
        predictions = model.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

    return model, metrics


def _persist_artifacts(model: Pipeline, metrics: dict) -> None:
    """Create output directories (if needed) and persist the artifacts."""

    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "refined_subscriber_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path.resolve()}")

    if metrics:
        metrics_path = results_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics written to {metrics_path.resolve()}")


def main() -> None:
    """Train the model using the best-available CSV and persist the outputs."""

    data_path, df = find_training_data()
    print(f"Loaded training data from: {data_path}")
    model, metrics = train_and_evaluate(df)
    _persist_artifacts(model, metrics)


if __name__ == "__main__":
    main()
