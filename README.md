#Time-Series Forecasting for Daily Sales

A comprehensive forecasting pipeline that progresses from classical statistical baselines to deep learning and tree-based machine learning, predicting daily sales for a single store–item combination while explicitly modeling weekly seasonality, trend, and non-linear temporal patterns.

## Overview

The project forecasts daily sales using a systematic, escalating modeling approach — starting with simple heuristics and statistical time-series models, then moving to sequence-based deep learning, and finally to feature-engineered supervised ML. Each stage is benchmarked against the same held-out test period so that improvements can be attributed directly to the modeling choice.

**Modeling progression:**

1. **Statistical baselines** — Seasonal Naive, Holt-Winters Triple Exponential Smoothing, ARIMA/SARIMA, and SARIMAX (with weekday/month exogenous variables) to capture 7-day seasonality
2. **Deep learning** — an LSTM network trained on sliding windows of the raw sequence, without imposing stationarity
3. **Supervised ML** — Linear Regression and hyperparameter-tuned XGBoost trained on engineered lag and rolling-window features

## Dataset

Daily sales records (Kaggle-style store–item demand data) filtered to a single store and item for a focused, interpretable analysis:

- **Train:** January 2013 – September 2017
- **Test / validation:** last 3 months of 2017 (Oct–Dec)
- Engineered calendar features: year, month, day, weekday

## Key Findings

| Model | MAE | RMSE | MAPE / WMAPE |
|---|---|---|---|
| Seasonal Naive (baseline) | — | — | 31.80% |
| Holt-Winters (undamped) | — | — | 30.19% |
| Holt-Winters (damped) | — | — | 35.91% |
| SARIMA(6,1,1)(1,1,1,7) | — | — | 28.03% |
| SARIMAX (+ exog) | — | — | 28.09% |
| LSTM | 4.258 | 5.353 | 21.55% |
| Linear Regression (lag/rolling features) | 4.570 | 5.574 | 17.50% |
| XGBoost (tuned, lag/rolling features) | 4.676 | 5.609 | 17.90% |

**Takeaways:**
- Explicit weekly seasonality (`s=7`) and exogenous calendar variables move SARIMA/SARIMAX modestly ahead of Holt-Winters and the seasonal naive baseline.
- The LSTM, trained directly on the raw sequence without differencing, closes much of the remaining gap by learning non-linear dependencies the statistical models can't express.
- Reframing the problem as supervised regression — using lag features (`lag_1`, `lag_7`) and rolling statistics (`rolling_mean`, `rolling_max`, `rolling_min`) — gives the biggest jump in percentage-based accuracy. Linear Regression edges out tuned XGBoost on this dataset, underscoring how much of the signal is captured by the engineered features themselves rather than model complexity.

## Methodology

### 1. Exploratory Data Analysis
- Weekly sales distribution (boxplots by weekday) — sales rise through the week, peak Saturday, drop sharply Sunday
- Monthly and yearly aggregates — clear upward trend (~33% growth from 2013 to 2017) and within-year seasonality peaking in July

### 2. Statistical Models
- **Seasonal Naive:** uses the prior year's value for the same calendar date as the forecast
- **Holt-Winters Triple Exponential Smoothing:** ETS framework (Multiplicative Error, Additive Trend, Multiplicative Seasonality) selected from the decomposition plot; both damped and undamped trend variants evaluated
- **Stationarity testing:** rolling mean/std visualization, Augmented Dickey-Fuller test, and ACF/PACF diagnostics before and after first-order differencing
- **SARIMA(6,1,1)(1,1,1,7):** order selected from PACF/ACF cutoffs; seasonal terms added after ACF revealed significant spikes at lags 7, 14, 21, 28, 35 (weekly pattern)
- **SARIMAX:** same specification extended with weekday and month exogenous regressors
- Residual ACF/PACF diagnostics run after each statistical fit to check for remaining structure

### 3. Deep Learning — LSTM
- Data reshaped into a supervised sliding-window format (30-day lookback) and scaled with `MinMaxScaler`
- Single LSTM layer (70 units, ReLU) followed by Dense output — trained without manual differencing, letting the network learn trend and seasonality directly

### 4. Supervised Machine Learning
- Chronological (non-shuffled) train/test split to prevent leakage
- Feature engineering: `lag_1`, `lag_7`, `rolling_mean`, `rolling_max`, `rolling_min`, selected via `SelectKBest`
- **Linear Regression** as a causal-model baseline on the engineered features
- **XGBoost**, tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV) over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and `colsample_bytree`, optimizing for MAE

### Evaluation Metrics
- **MAE** — mean absolute error, in original sales units
- **RMSE** — root mean squared error, penalizes large errors more heavily
- **MAPE / WMAPE** — scale-independent percentage error, used as the primary metric for cross-model comparison

## Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Statistical modeling:** `statsmodels` (`ExponentialSmoothing`, `ARIMA`, `SARIMAX`, `adfuller`, `plot_acf`/`plot_pacf`)
- **Deep learning:** `tensorflow.keras` (`LSTM`, `Dense`, `Dropout`)
- **Machine learning:** `scikit-learn` (`LinearRegression`, `SelectKBest`, `MinMaxScaler`, metrics), `xgboost` (`XGBRegressor`, `RandomizedSearchCV`)

## Repository Structure

```
.
├── Time_Series_Project.ipynb   # Full analysis: EDA → statistical models → LSTM → supervised ML
└── README.md
```

## Getting Started

```bash
pip install pandas numpy seaborn matplotlib statsmodels scikit-learn xgboost tensorflow
```

Open `Time_Series_Project.ipynb` and run the cells sequentially — each modeling section is self-contained and builds its own train/test split from the shared cleaned dataset.

## Conclusion

While the LSTM captured the most complex sequential dynamics without relying on stationarity assumptions, carefully engineered lag and rolling-window features fed into simpler regression models (Linear Regression, XGBoost) delivered the strongest percentage-based accuracy — highlighting that, for this dataset, explicit feature engineering was at least as valuable as model complexity.
