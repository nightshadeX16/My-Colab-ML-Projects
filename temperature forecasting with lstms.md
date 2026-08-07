# Temperature Forecasting with LSTM

A time-series forecasting model that predicts daily minimum temperature using a stacked LSTM network, evaluated with a full metrics suite and time-series cross-validation.

## Overview

This project builds a sequence-to-one LSTM model to forecast the next day's minimum temperature from the previous 30 days of readings. Beyond just training the model, it evaluates it properly on held-out data and includes a discussion of what the results actually mean, rather than reporting a single metric in isolation.

## Dataset

- Daily minimum temperatures in Melbourne, Australia (1981-1990)
- 3,650 daily records
- Source: [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) (`daily-min-temperatures.csv`)

## Approach

1. **Preprocessing** — normalized the temperature series with MinMaxScaler and converted it into supervised-learning sequences using a 30-day look-back window (30 days of history to predict day 31).
2. **Train/Test Split** — 80/20 chronological split (2,896 training sequences, 724 test sequences), preserving time order rather than shuffling, which matters for time-series data.
3. **Model Architecture** — a stacked LSTM: LSTM(50) -> LSTM(30) -> Dense(1), trained with the Adam optimizer and MSE loss.
4. **Inference & Evaluation** — ran the trained model on the untouched test set and computed a full metrics suite rather than relying on training loss alone.
5. **Cross-Validation** — repeated training and evaluation across 3 time-series cross-validation splits to confirm the results weren't specific to one lucky train/test split.

## Results (held-out test set)

| Metric | Value |
|---|---|
| RMSE | 2.17 degrees C |
| MAE | 1.72 degrees C |
| MAPE | 19.08% |
| SMAPE | 17.43% |
| R2 | 0.72 |
| Directional Accuracy | 46.06% |

Metrics were consistent across the 3-fold time-series cross-validation, confirming the results generalize rather than reflecting one favorable split.

## An Important Caveat

The RMSE and R2 look solid at first glance, but the directional accuracy (46%) is actually *below* random guessing (50%). This is a common issue with LSTM forecasts on smooth, highly autocorrelated series like daily temperature: the model can achieve a low error simply by predicting that tomorrow will look like today, without actually learning the underlying dynamics that drive temperature to rise or fall. In other words, low error alone does not mean the model has learned to forecast well — it's a reminder to always check more than one metric, especially for time-series problems where naive persistence can look deceptively good.

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn (MinMaxScaler)
- NumPy / pandas
- Matplotlib
- Google Colab

## Key Takeaways

- A single metric (like RMSE) can be misleading on time-series data; directional accuracy revealed a limitation that error metrics alone did not.
- Time-series cross-validation is important for confirming that results are not an artifact of a single train/test split.
- Proper preprocessing (chronological splitting, sequence windowing) is essential to avoid data leakage in time-series problems.
