"""Refined YouTube subscriber prediction model.

This script trains an improved regression model using gradient boosting and
additional feature engineering compared to the baseline notebook.  It prints
cross-validation metrics and hold-out performance along with the top features
(by split gain) for transparency.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_log_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import TransformedTargetRegressor

DATA_CANDIDATES = [
    Path("YouTube_Combined_Cleaned (1).csv"),
    Path("youtube_channels_with_features.csv"),
    Path("sample_youtube_channels.csv"),
]


def _load_dataset() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded dataset: {path} ({len(df):,} rows)")
            return df
    raise FileNotFoundError(
        "Could not find any of the expected datasets: "
        + ", ".join(str(p) for p in DATA_CANDIDATES)
    )


def _engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df['published'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
    df = df.dropna(subset=['published', 'views', 'videos', 'subs']).copy()

    now = pd.Timestamp.utcnow()
    df['channel_age_years'] = (now - df['published']).dt.days / 365.25

    # Derived, leakage-safe features.
    df['views_per_video'] = np.where(df['videos'] > 0, df['views'] / df['videos'], np.nan)
    df['uploads_per_year'] = np.where(df['channel_age_years'] > 0,
                                      df['videos'] / df['channel_age_years'],
                                      np.nan)
    df['log_views_growth'] = np.log1p(df['views']) / np.maximum(df['channel_age_years'], 0.1)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _build_pipeline(numeric_features: list[str]) -> Pipeline:
    log_transformer = FunctionTransformer(lambda x: np.log1p(np.clip(x, a_min=0, a_max=None)),
                                          validate=False)

    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("log", log_transformer),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    gbr_params = dict(
        learning_rate=0.06,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=25,
        l2_regularization=0.15,
        random_state=42,
    )

    gb_regressor = HistGradientBoostingRegressor(**gbr_params)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        (
            "regressor",
            TransformedTargetRegressor(
                regressor=gb_regressor,
                func=np.log1p,
                inverse_func=lambda x: np.expm1(np.clip(x, a_min=None, a_max=20)),
            ),
        ),
    ])

    return model


def _neg_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(y_pred, 0.0)
    return -math.sqrt(mean_squared_log_error(y_true, y_pred))


rmsle_scorer = make_scorer(_neg_rmsle, greater_is_better=True)


def main() -> None:
    raw_df = _load_dataset()
    df = _engineer_features(raw_df)

    feature_cols = [
        'views',
        'videos',
        'channel_age_years',
        'views_per_video',
        'uploads_per_year',
        'log_views_growth',
    ]

    df = df.dropna(subset=feature_cols + ['subs']).copy()

    X = df[feature_cols]
    y = df['subs']

    model = _build_pipeline(feature_cols)

    print("Running 5-fold cross-validation...")
    cv_results = cross_validate(
        model,
        X,
        y,
        scoring={'r2': 'r2', 'rmsle': rmsle_scorer},
        cv=5,
        n_jobs=-1,
        return_train_score=False,
    )

    mean_r2 = cv_results['test_r2'].mean()
    std_r2 = cv_results['test_r2'].std()
    mean_rmsle = -cv_results['test_rmsle'].mean()
    std_rmsle = cv_results['test_rmsle'].std()

    print(f"CV R2: {mean_r2:.3f} ± {std_r2:.3f}")
    print(f"CV RMSLE: {mean_rmsle:.3f} ± {std_rmsle:.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    check_is_fitted(model)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    holdout_r2 = r2_score(y_test, preds)
    holdout_rmsle = math.sqrt(mean_squared_log_error(y_test, preds))

    print("Hold-out R2:", round(holdout_r2, 3))
    print("Hold-out RMSLE:", round(holdout_rmsle, 3))

    # Extract feature importances from underlying estimator if available.
    inner_regressor = model.named_steps['regressor'].regressor_
    if hasattr(inner_regressor, 'feature_importances_'):
        importances = inner_regressor.feature_importances_
        feat_imp = (
            pd.Series(importances, index=feature_cols)
            .sort_values(ascending=False)
        )
        print("\nTop feature importances (split gain):")
        for name, val in feat_imp.items():
            print(f"  {name:<20} {val:.3f}")

    # Save hold-out predictions for inspection.
    output = pd.DataFrame({
        'channel_title': df.loc[X_test.index, 'channel_title'].values
        if 'channel_title' in df.columns else X_test.index,
        'actual_subs': y_test,
        'predicted_subs': preds,
    })
    output_path = Path('predictions_refined_model.csv')
    output.to_csv(output_path, index=False)
    print(f"Saved hold-out predictions to {output_path}")


if __name__ == "__main__":
    main()
