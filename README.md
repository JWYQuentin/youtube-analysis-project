YouTube Channel Analytics — Predicting Subscribers

Goal. Predict YouTube channel subscribers from public channel-level metrics (views, uploads, channel age) and simple title/video style signals.

Dataset

Columns: artist_name, channel_ID, channel_title, published, views, subs, videos

Repo ships a small sample: data/sample_youtube_channels.csv

Full dataset: add locally as YouTube_Combined_Cleaned (1).csv (not committed)

Methods

EDA: distributions, top channels, log–log relationships, correlations

Features (no leakage): channel_age_years, views_per_video, plus optional text/video aggregates (title length, !, ?, emojis, keywords, sentiment, language share, median video stats)

Models: Ridge / Lasso, Random Forest, HistGradientBoosting (with CV or RF OOB)

Evaluation: R² on log-scale target and RMSLE; also sMAPE/WAPE on raw scale; segment analysis by channel size
## Quickstart
```bash
# 1) install
pip install -r requirements.txt
# 2) place the full CSV next to notebooks (optional)
# 3) run notebooks in order
notebooks/01_eda.ipynb
notebooks/02_modeling_baseline.ipynb
notebooks/03_model_comparison.ipynb

