# YouTube Channel Analytics — Predicting Subscribers

**Goal.** Predict YouTube channel subscribers from public channel-level metrics (views, uploads, channel age) and simple title/video style signals.

## Dataset
- Columns: `artist_name`, `channel_ID`, `channel_title`, `published`, `views`, `subs`, `videos`
- Sample for reproducibility: `data/sample_youtube_channels.csv`
- Full dataset (not committed): `YouTube_Combined_Cleaned (1).csv`

## Methods
1. **EDA:** distributions, top channels, log–log relationships, correlations  
2. **Features (no leakage):** `channel_age_years`, `views_per_video`, plus optional title/video aggregates (length, `!`, `?`, emojis, keywords, sentiment, language share, median video stats)  
3. **Models:** Ridge/Lasso, Random Forest, HistGradientBoosting (CV or RF OOB)  
4. **Evaluation:** R² on log target (`log1p(subs)`) and **RMSLE**; also sMAPE/WAPE on raw scale; segment analysis by channel size

## Quickstart
```bash
# 1) install
pip install -r requirements.txt

# 2) place the full CSV next to notebooks (optional)
#    otherwise the repo sample will run

# 3) run notebooks in order
notebooks/01_eda.ipynb
notebooks/02_modeling_baseline.ipynb
notebooks/03_model_comparison.ipynb
notebooks/04_text_features.ipynb

# 4) (optional) demo app
streamlit run app.py
