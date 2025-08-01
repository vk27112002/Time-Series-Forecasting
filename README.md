# Time Series Forecasting of Atmospheric CO₂ Levels

This project focuses on the analysis and forecasting of atmospheric CO₂ concentration levels using classical time series modeling techniques. The data originates from the `statsmodels` dataset and is sourced from the Mauna Loa Observatory in Hawaii.

---

##  Objective

To model and forecast monthly atmospheric CO₂ levels using SARIMA after performing exploratory analysis and transformation for stationarity.

---

##  Dataset

- **Source**: Mauna Loa Observatory, Hawaii
- **Accessed via**: `statsmodels.datasets.co2`
- **Structure**: 526 rows (resampled from weekly to monthly)
- **Variable**: Atmospheric CO₂ concentration (in ppm)

---

##  Methodology

1. **Exploratory Data Analysis**
   - Visualized raw time series trends
   - Identified seasonality and patterns

2. **STL Decomposition**
   - Separated trend, seasonal, and residual components

3. **Detrending & Stationarity**
   - Applied differencing to make the series stationary
   - Verified using:
     - Augmented Dickey-Fuller (ADF) test
     - KPSS test

4. **Model Identification**
   - Used ACF and PACF plots to estimate AR and MA orders
   - Compared models using AIC and BIC scores

5. **Model Training & Forecasting**
   - Data split: 90% train / 10% test
   - Trained SARIMA model on training set
   - Forecasted the test set

6. **Model Evaluation**
   - Compared forecasted vs. actual values using:
     - **MAE**: 1.47  
     - **RMSE**: 1.54  
     - **R² Score**: 0.72

---

##  Author

Vaibhav Khare

---

*Developed as part of a time series modeling project using real-world atmospheric CO₂ data.*
