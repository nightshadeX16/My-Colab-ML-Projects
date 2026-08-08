# Credit Card Fraud Detection

A comparison of classical machine learning models for detecting fraudulent credit card transactions on a highly imbalanced dataset.

## Overview

Credit card fraud detection is a binary classification problem made difficult by extreme class imbalance — fraudulent transactions are rare compared to legitimate ones, so a naive model can score high on accuracy while missing almost every actual fraud case. This project addresses that imbalance directly and compares several models to find the best trade-off between catching fraud and minimizing false positives.

## Dataset

- ~284,000 credit card transactions
- Only 98 labeled as fraudulent, reflecting real-world class imbalance
- Features are primarily anonymized/transformed numerical variables plus transaction amount

## Approach

1. **Exploratory Data Analysis** — examined the class distribution and confirmed the severity of the imbalance.
2. **Class Balancing** — applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to generate synthetic examples of the minority (fraud) class, avoiding the pitfalls of simple oversampling or undersampling.
3. **Model Training** — trained and compared five classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
4. **Evaluation** — assessed each model using accuracy, precision, recall, F1-score, and ROC-AUC, since accuracy alone is a poor indicator of performance on imbalanced data.

## Results

| Model | ROC-AUC | Notes |
|---|---|---|
| XGBoost | 0.978 | Best-performing model overall |
| Random Forest | High test accuracy (~0.9995) | Strong performance, slightly behind XGBoost on ROC-AUC |
| LightGBM | Competitive | Faster training than XGBoost |
| Decision Tree | Lower | More prone to overfitting on this dataset |
| Logistic Regression | Baseline | Useful benchmark, underperforms tree-based ensembles |

XGBoost was selected as the best model based on ROC-AUC, which better reflects performance on imbalanced classification than raw accuracy.

## Tech Stack

- Python
- scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn (SMOTE)
- pandas / NumPy
- Google Colab

## Key Takeaways

- Accuracy is a misleading metric on imbalanced datasets; ROC-AUC and F1-score give a much clearer picture of real model performance.
- SMOTE meaningfully improved the minority class recall compared to training on the raw imbalanced data.
- Gradient-boosted tree models (XGBoost, LightGBM) outperformed simpler baselines, consistent with their strong track record on structured/tabular data.
