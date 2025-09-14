# YouTube Channel Analytics — Predicting Subscribers

**Goal.** Use public channel-level metrics (views, uploads, channel age, ratios) to predict subscriber counts and understand key drivers.

## Dataset
- Columns: `artist_name`, `channel_ID`, `channel_title`, `published`, `views`, `subs`, `videos`
- Repo ships a small sample: `data/sample_youtube_channels.csv`
- Full dataset: add locally as `YouTube_Combined_Cleaned (1).csv` (not committed)

## Methods
1. **EDA:** distributions, top channels, log-log relationships, correlations  
2. **Features:** `channel_age_years`, `views_per_video`, `subs_per_video`, `views_per_sub`, `growth_rate=subs/channel_age_years`  
3. **Models:** Linear, Ridge, Lasso, Random Forest with cross-validation / OOB  
4. **Evaluation:** R² and RMSE on held-out test set (log-scale target)

## Quickstart
```bash
# 1) install
pip install -r requirements.txt
# 2) place the full CSV next to notebooks (optional)
# 3) run notebooks in order
notebooks/01_eda.ipynb
notebooks/02_modeling_baseline.ipynb
notebooks/03_model_comparison.ipynb

