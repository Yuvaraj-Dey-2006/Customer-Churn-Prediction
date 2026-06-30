![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn using the Telco Customer Churn dataset. The project includes data analysis, feature engineering, model comparison, hyperparameter tuning, threshold optimization, and performance evaluation.

Predict whether a telecom customer is likely to churn based on demographic, account, and service usage information.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run](#run)
- [Output](#output)
- [Results](#results)
- [Key Insights](#key-insights)
- [Best Performing Model](#best-performing-model)
- [Notes](#notes)
- [Requirements](#requirements)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Class imbalance handling using model-native weighting
- Logistic Regression, LightGBM, and CatBoost models
- Hyperparameter tuning with Optuna
- 5-Fold Stratified Cross Validation
- Decision threshold optimization (F1-based)
- ROC Curves and Confusion Matrices
- Automatic plot generation
- Rich terminal progress bar

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- CatBoost
- Optuna
- Matplotlib
- Seaborn
- Rich

## Models

- Logistic Regression
- LightGBM Classifier
- CatBoost Classifier

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Dataset

**Dataset:** Telco Customer Churn

> **Note:** The dataset is not included in this repository. Please download it from Kaggle due to its original licensing terms.

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the dataset inside a folder named `Datasets` in the project root:

```
Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Project Structure

```
Customer-Churn-Prediction/
|
├── Churn_Prediction.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── Plots/
    |
    ├── Data Distributions/
    ├── Models Handling Imbalanced Class/
    ├── Models Handling Balanced Class/
    └── Tuned Models Handling Balanced Class/
```

## Installation

```bash
git clone https://github.com/Yuvaraj-Dey-2006/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

pip install -r requirements.txt
```

## Run

```bash
python Churn_Prediction.py
```

## Output

The project generates:

- Data analysis
- Model comparison
- Hyperparameter tuning results
- Final evaluation metrics
- Automatically saved plots
- Best model selection
- Total execution time

## Results

### Final Tuned Model Performance

| Model                | Accuracy | Precision | Recall | F1     | ROC-AUC |
|----------------------|----------|-----------|--------|--------|---------|
| Logistic Regression  | 0.787    | 0.583     | 0.698  | 0.635  | 0.838   |
| LightGBM Classifier  | 0.757    | 0.529     | 0.773  | 0.628  | 0.834   |
| CatBoost Classifier  | 0.762    | 0.536     | 0.773  | 0.633  | 0.839   |

*(metrics computed on a held-out test set, using per-model thresholds tuned for F1 on a separate validation split)*

## Key Insights

- **SMOTE on one-hot encoded categorical data degrades model quality.** Generating synthetic minority samples in encoded categorical space produces fractional values that don't correspond to any real customer (e.g. 0.4 of a contract type), inflating false positives. Switching to native class weighting (`class_weight='balanced'`, `scale_pos_weight`, `auto_class_weights='Balanced'`) fixed this without fabricating data.
- **Threshold tuning must use a separate validation set, not the test set.** Picking a decision threshold using test-set labels — even just to compute the "optimal" F1 cutoff — leaks test information into the final evaluation. A held-out validation split (carved from training data) is used to select thresholds, keeping the test set fully untouched until final reporting.
- **ROC-AUC, not accuracy or a fixed 0.5 cutoff, should drive model comparison during tuning.** Optuna objectives were switched from ROC-AUC to F1 scoring, since F1 directly reflects the precision/recall tradeoff relevant to identifying both churners and non-churners correctly — the business-relevant outcome, not just ranking quality.
- **This dataset has a real ceiling around ROC-AUC ≈ 0.84.** Some customers share nearly identical profiles (same contract, tenure, charges, services) yet differ in outcome — the available features can't fully separate them. No amount of additional tuning closes this gap; doing so would require richer data (e.g. support interaction history, usage trends) not present in this dataset.

## Best Performing Model

Based on ROC-AUC after hyperparameter tuning and threshold optimization, **CatBoost Classifier** achieved the best overall performance.

## Notes

- CatBoost is trained using native categorical features.
- LightGBM uses `scale_pos_weight`.
- Logistic Regression uses `class_weight='balanced'`.
- Decision thresholds are optimized on a validation set to avoid data leakage.
- Instead of using the default 0.5 cutoff, decision thresholds are optimized to maximize the F1 score.

## Requirements

- Python 3.10 or later
- Dependencies listed in `requirements.txt`

## Future Improvements

- Model explainability using SHAP
- Feature importance visualization
- Model serialization
- Interactive prediction interface
- Deployment with Streamlit or Flask

## License

The source code in this repository is licensed under the MIT License.

The Telco Customer Churn dataset is not owned by this repository and remains subject to its original license and terms of use.