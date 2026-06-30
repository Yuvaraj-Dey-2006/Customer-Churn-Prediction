![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn using the Telco Customer Churn dataset. The project includes data analysis, feature engineering, model comparison, hyperparameter tuning, threshold optimization, and performance evaluation.

Predict whether a telecom customer is likely to churn based on demographic, account, and service usage information.

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

### ROC Curve
![ROC Curve](Plots/Tuned%20Models%20Handling%20Balanced%20Class/12_roc_curve_tuned_models.png)

### Confusion Matrix
![Confusion Matrix](Plots/Tuned%20Models%20Handling%20Balanced%20Class/13_confusion_matrix_tuned_models.png)

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