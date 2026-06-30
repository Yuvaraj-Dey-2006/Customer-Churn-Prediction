'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ================================= #
#       IMPORTING LIBRARIES         #
# ================================= #

# to get the execution time
import time
start_time = time.perf_counter() # Start measuring execution time

# data loading and cleaning
import numpy as np
import pandas as pd
import io  # used to capture df.info() output so it can be printed via console.print()
           # instead of writing directly to stdout, which conflicts with the live Progress bar

# ignoring un-necessary warnings
import warnings
warnings.filterwarnings('ignore')

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# for splitting dataset
from sklearn.model_selection import train_test_split

# reducing manual working using pipeline
# data scaling, encoding and preprocessing(removing: missing values, imbalance dataset)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ML models
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# hyperparameter tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# cross validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

# performance testing — precision_recall_curve added for threshold tuning
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, precision_recall_curve

# to add progress bar
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
# ===================================================== #
#       INITIALIZE CONSOLE & PROGRESS BAR               #
# ===================================================== #

# Rich console for colored output
console = Console()

# Total stages in the project
TOTAL_STEPS = 100

progress = Progress(
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(bar_width=60),
    TextColumn("[bold yellow]{task.percentage:>3.0f}%"),
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
    console=console,  # share the same Console as console.print() calls below —
                       # without this, Progress silently creates its own Console,
                       # and two competing live renderers causes the bar to be
                       # reprinted as a new line on every refresh instead of
                       # updating in place
)

progress.start()

# Creates a single live progress bar
task = progress.add_task(
    "Running Customer Churn Prediction...",
    total=TOTAL_STEPS
)

console.print(
    "\n[bold green]Program Started Successfully[/bold green]\n"
)
'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ========================== #
#       DATA LOADING         #
# ========================== #

df = pd.read_csv(r"Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# converting str to int
df['Churn'] = df['Churn'].map({"Yes": 1, "No": 0})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

console.print("\n________________________ INFORMATION OF DATASET ________________________\n")
# .info() writes directly to stdout by default, which breaks the live Progress
# render — capture it to a buffer and print that through console instead
info_buffer = io.StringIO()
df.info(buf=info_buffer)
console.print(info_buffer.getvalue())
console.print("________________________________________________________________________\n")

# missing value checking in dataset
def missing_values(df):
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        console.print("[bold green]No missing values found.[/bold green]")
    else:
        console.print(missing.sort_values(ascending=False))

console.print("\n______________________ MISSING VALUEs IN DATASET ______________________\n")
missing_values(df)
console.print("_______________________________________________________________________\n")

# Update progress after successful dataset loading
progress.update(task, completed=5)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ================================================== #
#       DATA VISUALIZATION AND DISTRIBUTIONS         #
# ================================================== #

# setting the plot to saving directory
output_folder_1 = Path("Plots/Data Distributions")
output_folder_1.mkdir(parents=True, exist_ok=True)

# avoiding the repetition of savefig
def save_plot(folder, filename):
    plt.tight_layout()
    plt.savefig(folder / filename, dpi=300, bbox_inches="tight")
    plt.close()

# avoiding any change to real dataset
plot_df = df.copy()

# bin values into discrete intervals
plot_df["tenure_group"] = pd.cut(
    plot_df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"],
    include_lowest=True
)

# Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(
    x="Churn",
    data=plot_df,
    palette=["#4CAF50", "#F44336"],
    ax=ax
)

ax.set_xticklabels(["No", "Yes"])
ax.set_title("Churn Distribution",fontsize=16, fontweight='bold')

total = len(plot_df)

for p in ax.patches:
    height = p.get_height()
    percentage = 100 * height / total

    ax.text(
        p.get_x() + p.get_width() / 2,
        height + 20,
        f"{height}   ({percentage:.1f}%)",
        ha="center"
    )

save_plot(output_folder_1, "01_Churn_Distribution.png")

# Numerical Feature Distributions
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for col, ax in zip(num_cols, axes):
    sns.histplot(
        plot_df[col],
        bins=30,
        kde=True,
        color="#2978B5",
        ax=ax
    )

    ax.set_title(f"{col} Distribution", fontsize=16, fontweight='bold')

save_plot(output_folder_1, "02_Numerical_Distributions.png")

# Numerical Features vs Churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    x="Churn",
    y="tenure",
    data=plot_df,
    palette=["#4CAF50", "#F44336"],
    ax=axes[0]
)

axes[0].set_xticklabels(["No", "Yes"])
axes[0].set_title("Tenure by Churn", fontsize=16, fontweight='bold')

sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=plot_df,
    palette=["#4CAF50", "#F44336"],
    ax=axes[1]
)

axes[1].set_xticklabels(["No", "Yes"])
axes[1].set_title("Monthly Charges by Churn", fontsize=16, fontweight='bold')

save_plot(output_folder_1, "03_Churn_vs_Numerical.png")

# Correlation Heatmap
corr_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

corr = plot_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True,
    vmin=-1,
    vmax=1,
    ax=ax
)

ax.set_title("Correlation Heatmap", fontsize=16, fontweight='bold')

save_plot(output_folder_1, "04_Correlation_Heatmap.png")

# Major Categorical Features vs Churn Rate
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(
    axes,
    ["Contract", "InternetService", "PaymentMethod"]
):

    rate = (
        plot_df.groupby(col)["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    sns.barplot(
        x=rate.index,
        y=rate.values,
        palette="viridis",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(f"{col} vs Churn Rate", fontsize=16, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

save_plot(output_folder_1, "05_Categorical_Churn_Rates.png")

# Important Category-wise Churn Rates
important_categories = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "tenure_group"
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, col in zip(axes.flatten(), important_categories):

    rate = (
        plot_df.groupby(col)["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    sns.barplot(
        x=rate.index,
        y=rate.values,
        palette="Spectral",
        ax=ax
    )

    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(f"Churn Rate by {col}", fontsize=16, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

save_plot(output_folder_1, "06_Important_Categories_Churn.png")

console.print("\n_________________________________________ DATA VISUALIZATION PLOTS CREATED _________________________________________\n")
console.print(f"Plots saved to: {output_folder_1.absolute()}")
console.print("____________________________________________________________________________________________________________________\n")

# Update progress after EDA completion
progress.update(task, completed=18)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ================================================== #
#               OUTLIER ANALYSIS                     #
# ================================================== #

numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numerical_cols:

    Q1 = plot_df[col].quantile(0.25)
    Q3 = plot_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = plot_df[
        (plot_df[col] < lower_bound) |
        (plot_df[col] > upper_bound)
    ]

    percentage = len(outliers) / len(plot_df) * 100

    print(f"{col:<15}: {len(outliers):>4} outliers ({percentage:.2f}%)")



# Boxplot for outlier analysis

plt.figure(figsize=(7, 4))

sns.boxplot(
    data=plot_df[numerical_cols],
    palette="Set2"
)

plt.title("Outlier Analysis", fontsize=15, fontweight="bold")
plt.xticks(rotation=20)

save_plot(output_folder_1, "07_Outlier_Analysis.png")

console.print("\n__________________________________________________ OUTLIER ANALYSIS __________________________________________________\n")
console.print(f"Outlier Plot saved to: {output_folder_1.absolute()}")
console.print("__________________________________________________________________________________________________________________________\n")

# Update progress after outlier analysis
progress.update(task, completed=25)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ================================= #
#       FEATURES AND TARGETS        #
# ================================= #

# separating of features and target
X = df.drop(columns=['customerID', 'gender', 'Churn'])
y = df['Churn']

# --- FEATURE ENGINEERING ---
# charge-per-tenure ratio: flags customers paying high relative to how long they've stayed
# (+1 avoids divide-by-zero for tenure=0 customers)
X['ChargePerTenure'] = X['MonthlyCharges'] / (X['tenure'] + 1)

# count of subscribed add-on services: fewer services = weaker lock-in = higher churn risk
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
X['TotalServices'] = (X[service_cols] == 'Yes').sum(axis=1)

# tenure x contract interaction: short tenure on a month-to-month contract is a much
# stronger churn signal than short tenure alone — captures that combined effect
X['TenureContractRisk'] = X['tenure'] * X['Contract'].map(
    {'Month-to-month': 1, 'One year': 0.5, 'Two year': 0.1}
)

# gives the percentage of target class values.
console.print("\n____________________ PERCENTAGE DISTRIBUTION OF TARGET CLASS ____________________\n")
console.print(y.value_counts(normalize=True) * 100 )
console.print("_________________________________________________________________________________\n")

# missing value checking in features
console.print("\n______________________ MISSING VALUEs IN FEATURE SET ______________________\n")
missing_values(X)
console.print("_______________________________________________________________________\n")

# splitting in training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# further splitting training data into fit/validation — validation is used ONLY for
# picking the decision threshold later, so the test set stays untouched until final
# reporting (using X_test/y_test to choose the threshold would be data leakage:
# the threshold would be reverse-fit to the exact data we're "evaluating" on)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

console.print("\n_______________________ X_train SET INFORMATION _______________________\n")
# same fix as df.info() above — capture to buffer, print via console
xtrain_info_buffer = io.StringIO()
X_train.info(buf=xtrain_info_buffer)
console.print(xtrain_info_buffer.getvalue())
console.print("_______________________________________________________________________\n")

# Update progress after feature preparation
progress.update(task, completed=30)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ======================================= #
#     DATA ENCODING AND PREPROCESSING     #
# ======================================= #

# separating string type and numerical type columns of X_train
categ_cols_X_train = X_train.select_dtypes(include=['object', 'string']).columns
num_cols_X_train = X_train.select_dtypes(exclude=['object', 'string']).columns

# scaling and encoding for logistic regression
preprocessor_logreg = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols_X_train),
                                               ('categ', OneHotEncoder(drop='first',
                                                                        handle_unknown='ignore'), categ_cols_X_train)
                                                     ])

# lightGBM doesn't need scaling.
# encoding of lightGBM classification
preprocessor_lgbm = ColumnTransformer(transformers=[('categ', OneHotEncoder(drop='first', handle_unknown='ignore'), categ_cols_X_train)],
                                      remainder='passthrough')

# Update progress after preprocessing objects are ready
progress.update(task, completed=35)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ========================================================= #
#      HOW MUCH BASE MODELS CAN HANDLE IMBALANCE CLASS      #
# ========================================================= #

# pipeline with unbalanced data for logistic regression
pipe_logreg_unbal = Pipeline([('preprocessor_logreg', preprocessor_logreg),
                        ('logreg_base', LogisticRegression(random_state=42, n_jobs=-1))
                       ])

# pipeline with unbalanced data for lightGBM classification
pipe_lgbm_unbal = Pipeline([('preprocessor_lgbm', preprocessor_lgbm),
                      ('lgbm_base', LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
                    ])

# CatBoost handles un-encoded, un-processed and imbalance data efficiently.
# model initialization with default parameters
cbc_base_unbal = CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False, thread_count=-1)

# fitting the base logistic regression model with unbalanced training data
pipe_logreg_unbal.fit(X_train, y_train)

# fitting the base lightgbm classification model with training data
pipe_lgbm_unbal.fit(X_train, y_train)

# fitting the base cat boost classification model with training data
cat_features = list(categ_cols_X_train)
cbc_base_unbal.fit(X_train, y_train, cat_features=cat_features)

# predictions of three base models
y_pred_logreg = pipe_logreg_unbal.predict(X_test)
y_pred_lgbm = pipe_lgbm_unbal.predict(X_test)
y_pred_cbc = cbc_base_unbal.predict(X_test)

# base model evaluation with accuracy, precision, recall, f1, and roc-auc
base_result_unbal = pd.DataFrame({
                    'Models': ['Logistic Regression', 'LightGBM Classifier', 'CatBoost Classifier'],
                    'ACCURACY': [
                        accuracy_score(y_test, y_pred_logreg),
                        accuracy_score(y_test, y_pred_lgbm),
                        accuracy_score(y_test, y_pred_cbc)
                    ],
                    'PRECISION': [
                        precision_score(y_test, y_pred_logreg),
                        precision_score(y_test, y_pred_lgbm),
                        precision_score(y_test, y_pred_cbc)
                    ],
                    'RECALL': [
                        recall_score(y_test, y_pred_logreg),
                        recall_score(y_test, y_pred_lgbm),
                        recall_score(y_test, y_pred_cbc)
                    ],
                    'F1': [
                        f1_score(y_test, y_pred_logreg),
                        f1_score(y_test, y_pred_lgbm),
                        f1_score(y_test, y_pred_cbc)
                    ],
                    'ROC-AUC': [
                        roc_auc_score(y_test, pipe_logreg_unbal.predict_proba(X_test)[:,1]),
                        roc_auc_score(y_test, pipe_lgbm_unbal.predict_proba(X_test)[:,1]),
                        roc_auc_score(y_test, cbc_base_unbal.predict_proba(X_test)[:,1])
                    ]
                                  })

console.print("\n______________________________________ BASE MODELS PERFORMANCE (IMBALANCED CLASS) ______________________________________\n")
console.print(base_result_unbal.round(5))
console.print("________________________________________________________________________________________________________________________\n")

# Roc Curve to determine how the base models (Imbalanced) separates 0 and 1
output_folder_2 = Path("Plots/Models Handling Imbalanced Class")
output_folder_2.mkdir(parents=True, exist_ok=True)


# Predicted probabilities
y_prob_logreg = pipe_logreg_unbal.predict_proba(X_test)[:, 1]
y_prob_lgbm = pipe_lgbm_unbal.predict_proba(X_test)[:, 1]
y_prob_cb = cbc_base_unbal.predict_proba(X_test)[:, 1]

# ROC data
fpr_logreg_unbal, tpr_logreg_unbal, _ = roc_curve(y_test, y_prob_logreg)
fpr_lgbm_unbal, tpr_lgbm_unbal, _ = roc_curve(y_test, y_prob_lgbm)
fpr_cb_unbal, tpr_cb_unbal, _ = roc_curve(y_test, y_prob_cb)

# AUC
auc_logreg = roc_auc_score(y_test, y_prob_logreg)
auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)
auc_cb = roc_auc_score(y_test, y_prob_cb)

# Plot
plt.figure(figsize=(18,10))

plt.plot(fpr_logreg_unbal, tpr_logreg_unbal,
         label=f"Logistic Regression (AUC = {auc_logreg:.4f})")

plt.plot(fpr_lgbm_unbal, tpr_lgbm_unbal,
         label=f"LightGBM (AUC = {auc_lgbm:.4f})")

plt.plot(fpr_cb_unbal, tpr_cb_unbal,
         label=f"CatBoost (AUC = {auc_cb:.4f})")

# Random classifier
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Baseline Models (Imbalanced Class)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

save_plot(output_folder_2, "01_roc_curve_baseline_imbalanced.png")

# visualized base model evaluation for confusion matrix
fig, axes = plt.subplots(1,3,figsize=(10, 7))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_logreg, cmap="Blues", ax=axes[0], colorbar=False, display_labels=['No', 'Yes'])
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lgbm, cmap="Blues", ax=axes[1], colorbar=False, display_labels=['No', 'Yes'])
axes[1].set_title("LightGBM Classification")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cbc, cmap="Blues", ax=axes[2], colorbar=False, display_labels=['No', 'Yes'])
axes[2].set_title("Cat Boost Classification")

plt.suptitle("CONFUSION MATRIX (Imbalanced Class)", fontsize=16, fontweight='bold', y=0.8)
plt.tight_layout()
save_plot(output_folder_2, "02_confusion_matrix_baseline_imbalanced.png")

console.print("\n___________________________________________________ IMBALANCED CLASS HANDLING ANALYSIS ___________________________________________________")
console.print(f"\nAnalysis Plots saved to: {output_folder_2.absolute()}")
console.print("__________________________________________________________________________________________________________________________________________\n")

# Update progress after imbalanced baseline models
progress.update(task, completed=50)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
# =========================================== #
#      BASE MODELS AFTER BALANCING CLASS      #
# =========================================== #

# pipeline with class-weighted logistic regression — replaces SMOTE; reweighting the
# loss function on real data avoids the synthetic-sample corruption SMOTE causes on
# one-hot encoded columns
pipe_logreg_bal = Pipeline([('preprocessor_logreg', preprocessor_logreg),
                        ('logreg_base', LogisticRegression(class_weight='balanced', random_state=42, n_jobs=-1))
                       ])

# scale_pos_weight = ratio of negative to positive class — LightGBM's native
# weighting mechanism, same purpose as class_weight='balanced' above
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# pipeline with class-weighted lightGBM classification
pipe_lgbm_bal = Pipeline([('preprocessor_lgbm', preprocessor_lgbm),
                      ('lgbm_base', LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1))
                    ])

# cat boost took raw data. It is it's native feature
cbc_base_bal = CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False, thread_count=-1, auto_class_weights='Balanced')

# fitting the base logistic regression model with balanced training data
pipe_logreg_bal.fit(X_train, y_train)

# fitting the base lightgbm classification model with training data
pipe_lgbm_bal.fit(X_train, y_train)

# fitting the base cat boost classification model with training data
cbc_base_bal.fit(X_train, y_train, cat_features=cat_features)

# predictions of three base models
y_pred_logreg = pipe_logreg_bal.predict(X_test)
y_pred_lgbm = pipe_lgbm_bal.predict(X_test)
y_pred_cbc = cbc_base_bal.predict(X_test)

# base model evaluation with accuracy, precision, recall, f1, and roc-auc
base_result_bal = pd.DataFrame({
                    'Models': ['Logistic Regression', 'LightGBM Classifier', 'CatBoost Classifier'],
                    'ACCURACY': [
                        accuracy_score(y_test, y_pred_logreg),
                        accuracy_score(y_test, y_pred_lgbm),
                        accuracy_score(y_test, y_pred_cbc)
                    ],
                    'PRECISION': [
                        precision_score(y_test, y_pred_logreg),
                        precision_score(y_test, y_pred_lgbm),
                        precision_score(y_test, y_pred_cbc)
                    ],
                    'RECALL': [
                        recall_score(y_test, y_pred_logreg),
                        recall_score(y_test, y_pred_lgbm),
                        recall_score(y_test, y_pred_cbc)
                    ],
                    'F1': [
                        f1_score(y_test, y_pred_logreg),
                        f1_score(y_test, y_pred_lgbm),
                        f1_score(y_test, y_pred_cbc)
                    ],
                    'ROC-AUC': [
                        roc_auc_score(y_test, pipe_logreg_bal.predict_proba(X_test)[:,1]),
                        roc_auc_score(y_test, pipe_lgbm_bal.predict_proba(X_test)[:,1]),
                        roc_auc_score(y_test, cbc_base_bal.predict_proba(X_test)[:,1])
                    ]
                                  })
console.print("\n______________________________________ BASE MODELS PERFORMANCE (BALANCED CLASS) ______________________________________\n")
console.print(base_result_bal.round(5))
console.print("______________________________________________________________________________________________________________________\n")

# Roc Curve to determine how the base models (Balanced) separates 0 and 1 
output_folder_3 = Path("Plots/Models Handling Balanced Class")
output_folder_3.mkdir(parents=True, exist_ok=True)

# Predicted probabilities
y_prob_logreg = pipe_logreg_bal.predict_proba(X_test)[:, 1]
y_prob_lgbm = pipe_lgbm_bal.predict_proba(X_test)[:, 1]
y_prob_cb = cbc_base_bal.predict_proba(X_test)[:, 1]

# ROC data
fpr_logreg_bal, tpr_logreg_bal, _ = roc_curve(y_test, y_prob_logreg)
fpr_lgbm_bal, tpr_lgbm_bal, _ = roc_curve(y_test, y_prob_lgbm)
fpr_cb_bal, tpr_cb_bal, _ = roc_curve(y_test, y_prob_cb)

# AUC
auc_logreg = roc_auc_score(y_test, y_prob_logreg)
auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)
auc_cb = roc_auc_score(y_test, y_prob_cb)

# Plot
plt.figure(figsize=(18,10))

plt.plot(fpr_logreg_bal, tpr_logreg_bal,
         label=f"Logistic Regression (AUC = {auc_logreg:.4f})")

plt.plot(fpr_lgbm_bal, tpr_lgbm_bal,
         label=f"LightGBM (AUC = {auc_lgbm:.4f})")

plt.plot(fpr_cb_bal, tpr_cb_bal,
         label=f"CatBoost (AUC = {auc_cb:.4f})")

# Random classifier
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Baseline Models (Balanced Class)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

save_plot(output_folder_3, "01_roc_curve_baseline_balanced.png")

# visualized base model evaluation for confusion matrix
fig, axes = plt.subplots(1,3,figsize=(10, 7))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_logreg, cmap="Blues", ax=axes[0], colorbar=False, display_labels=['No', 'Yes'])
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lgbm, cmap="Blues", ax=axes[1], colorbar=False, display_labels=['No', 'Yes'])
axes[1].set_title("LightGBM Classification")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cbc, cmap="Blues", ax=axes[2], colorbar=False, display_labels=['No', 'Yes'])
axes[2].set_title("Cat Boost Classification")

plt.suptitle("CONFUSION MATRIX (Balanced Class)", fontsize=16, fontweight='bold', y=0.8)
plt.tight_layout()
save_plot(output_folder_3 , "02_confusion_matrix_baseline_balanced.png")

console.print("\n___________________________________________________ BALANCED CLASS HANDLING ANALYSIS ___________________________________________________\n")
console.print(f"Analysis Plots saved to: {output_folder_3.absolute()}")
console.print("________________________________________________________________________________________________________________________________________\n")

# Update progress after balanced model analysis
progress.update(task, completed=65)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ================================================================= #
#       HYPERPARAMETER TUNING USING OPTUNA + STRATIFIED K-FOLD      #
# ================================================================= #

# cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# tuning logistic regression — scoring switched to F1 so tuning optimizes the
# precision/recall balance that matters for the business goal, not just ranking quality
def objective_logreg(trial):
    solver = trial.suggest_categorical("solver",["liblinear", "lbfgs"])

    if solver == "liblinear":
        penalty = trial.suggest_categorical("penalty",["l1", "l2"])
    else:
        penalty = "l2"
    
    logreg = LogisticRegression(C = trial.suggest_float("C", 1e-4, 100, log=True),
                                solver = solver,
                                penalty = penalty,
                                max_iter=1000,
                                class_weight='balanced',
                                random_state=42
                                )
    
    pipe_logreg_tune = Pipeline([('preprocessor', preprocessor_logreg),
                     ('logreg', logreg)
                     ])

    score = cross_val_score(pipe_logreg_tune, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
    return score

study_logreg = optuna.create_study(direction='maximize')
study_logreg.optimize(objective_logreg, n_trials=30)

# Logistic Regression tuning finished
progress.update(task, completed=72)

# tuning lightGBM Classification
def objective_lgbm(trial):
    lgbm = LGBMClassifier(n_estimators = trial.suggest_int('n_estimators',50, 200),
                          learning_rate = trial.suggest_float('learning_rate',0.01,0.2,log=True),
                          num_leaves = trial.suggest_int('num_leaves', 20, 100),
                          max_depth = trial.suggest_int('max_depth', 3, 10),
                          min_child_samples = trial.suggest_int('min_child_samples', 10, 50),
                          subsample = trial.suggest_float('subsample', 0.7, 1.0),
                          colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0),
                          scale_pos_weight=scale_pos_weight,
                          random_state=42,
                          verbose=-1
                          )
    
    pipe_lgbm_tune = Pipeline([('preprocessor', preprocessor_lgbm),
                     ('lgbm', lgbm)
                     ])
    
    score = cross_val_score(pipe_lgbm_tune, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
    return score

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=35)

# LightGBM tuning finished
progress.update(task, completed=80)

# tuned cat boost classification
def objective_cb(trial):
    cb = CatBoostClassifier(iterations = trial.suggest_int('iterations', 50, 200),
                            depth = trial.suggest_int('depth', 4, 6),
                            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                            l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10),
                            random_strength = trial.suggest_float('random_strength', 0, 5),
                            random_state = 42,
                            verbose = 0,
                            allow_writing_files = False,
                            thread_count=-1,
                            cat_features=cat_features,
                            auto_class_weights='Balanced'
                            )
    
    scores = []

    # inside working of cross_val_score(). This way it can also take str features.
    # named X_cv_val/y_cv_val (not X_val/y_val) to avoid confusion with the
    # separate held-out validation set used later for threshold tuning
    for train_idx, val_idx in cv.split(X_train, y_train):


        X_tr = X_train.iloc[train_idx] # set of train data from X_train used in validation
        X_cv_val = X_train.iloc[val_idx] # set of validation data from X_train used in validation

        y_tr = y_train.iloc[train_idx] # set of train data from y_train used in validation
        y_cv_val = y_train.iloc[val_idx] # set of validation data from y_train used in validation

        cb.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_cv_val, y_cv_val), early_stopping_rounds=50)

        # F1 instead of ROC-AUC: optimizes the precision/recall balance directly,
        # which is what determines correct Yes/No calls in the confusion matrix
        y_hat = cb.predict(X_cv_val)
        score = f1_score(y_cv_val, y_hat)
        scores.append(score)

    return np.mean(scores) # returns validation score which we use from study.best_value_

study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_cb, n_trials=35)

# CatBoost tuning finished
progress.update(task, completed=90)

console.print("\n_________________________________________________________________ BEST PARAMETERS SELECTED _________________________________________________________________\n")
console.print(f"Logistic Regression      : {study_logreg.best_params}")
console.print(f"LightGBM Classification  : {study_lgbm.best_params}")
console.print(f"Cat Boost Classification : {study_cb.best_params}")
console.print("____________________________________________________________________________________________________________________________________________________________\n")

tune_val = pd.DataFrame({
            'MODEL': ['Logistic Regression', 'LightGBM Classifier', 'CatBoost Classifier'],

            'VALIDATION SCORE': [
                                 study_logreg.best_value,
                                 study_lgbm.best_value,
                                 study_cb.best_value
                                ]
            })

console.print("\n______________________________________ VALIDATION SCORES ______________________________________\n")
console.print(tune_val.round(5))
console.print("_______________________________________________________________________________________________\n")

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# ===================================== #
#       TRAINING THE TUNED MODELS       #
# ===================================== #

# Training tuned Logistic Regression model (class_weight replaces SMOTE)
# fit on X_fit (not full X_train) — X_val is held back for threshold tuning below
model_logreg = Pipeline([('preprocessor', preprocessor_logreg),
                  ('logreg', LogisticRegression(**study_logreg.best_params, max_iter=1000,
                                                 class_weight='balanced', random_state=42))
                ])
model_logreg.fit(X_fit, y_fit)

# Training tuned LightGBM Classification model (scale_pos_weight replaces SMOTE)
model_lgbm = Pipeline([('preprocessor', preprocessor_lgbm),
                  ('lgbm', LGBMClassifier(**study_lgbm.best_params, scale_pos_weight=scale_pos_weight,
                                           random_state=42, verbose=-1))
                ])
model_lgbm.fit(X_fit, y_fit)

# Training tuned Cat Boost Classification model
model_cb = CatBoostClassifier(**study_cb.best_params, 
                              random_state=42, 
                              verbose=0,
                              auto_class_weights='Balanced',
                              allow_writing_files=False,
                              thread_count=-1
                              )
model_cb.fit(X_fit, y_fit, cat_features=cat_features)

# --- THRESHOLD TUNING (on validation set, NOT test set) ---
# Default 0.5 cutoff is arbitrary and doesn't target the business goal of getting
# both Yes and No predictions correct. This scans thresholds and picks the one that
# maximizes F1 (best precision/recall balance) for each model independently.
# Using X_val/y_val (held out from training, separate from X_test) avoids data
# leakage — picking the threshold using y_test directly would inflate the test
# score, since the cutoff would be reverse-fit to the exact labels we're scoring against.
def best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx]

# probabilities on the validation set, used only to pick each threshold
val_proba_logreg = model_logreg.predict_proba(X_val)[:,1]
val_proba_lgbm = model_lgbm.predict_proba(X_val)[:,1]
val_proba_cb = model_cb.predict_proba(X_val)[:,1]

threshold_logreg = best_threshold(y_val, val_proba_logreg)
threshold_lgbm = best_threshold(y_val, val_proba_lgbm)
threshold_cb = best_threshold(y_val, val_proba_cb)

console.print(f"\n[bold white on blue] OPTIMAL DECISION THRESHOLDS (F1-maximizing, chosen on validation set) [/bold white on blue]")
console.print(f"  Logistic Regression : {threshold_logreg:.3f}")
console.print(f"  LightGBM Classifier : {threshold_lgbm:.3f}")
console.print(f"  CatBoost Classifier : {threshold_cb:.3f}")

# probabilities on the untouched test set — used for final, honest reporting
pred_proba_logreg = model_logreg.predict_proba(X_test)[:,1]
pred_proba_lgbm = model_lgbm.predict_proba(X_test)[:,1]
pred_proba_cb = model_cb.predict_proba(X_test)[:,1]

# final class predictions: test-set probabilities cut at the validation-chosen threshold
pred_logreg = (pred_proba_logreg >= threshold_logreg).astype(int)
pred_lgbm = (pred_proba_lgbm >= threshold_lgbm).astype(int)
pred_cb = (pred_proba_cb >= threshold_cb).astype(int)

# final tuned model evaluation with accuracy, precision, recall, f1, and roc-auc
final_result = pd.DataFrame({
                    'Models': ['Logistic Regression', 'LightGBM Classifier', 'CatBoost Classifier'],
                    'ACCURACY': [
                        accuracy_score(y_test, pred_logreg),
                        accuracy_score(y_test, pred_lgbm),
                        accuracy_score(y_test, pred_cb)
                    ],
                    'PRECISION': [
                        precision_score(y_test, pred_logreg),
                        precision_score(y_test, pred_lgbm),
                        precision_score(y_test, pred_cb)
                    ],
                    'RECALL': [
                        recall_score(y_test, pred_logreg),
                        recall_score(y_test, pred_lgbm),
                        recall_score(y_test, pred_cb)
                    ],
                    'F1': [
                        f1_score(y_test, pred_logreg),
                        f1_score(y_test, pred_lgbm),
                        f1_score(y_test, pred_cb)
                    ],
                    'ROC-AUC': [
                        roc_auc_score(y_test, pred_proba_logreg),
                        roc_auc_score(y_test, pred_proba_lgbm),
                        roc_auc_score(y_test, pred_proba_cb)
                    ]
                                  })
console.print("\n________________________________________ TUNED MODEL PERFORMANCE (BALANCED CLASS) ________________________________________\n")
console.print(final_result.round(5))
console.print("__________________________________________________________________________________________________________________________\n")

# Final models trained and evaluated
progress.update(task, completed=98)

# Roc Curve to determine how the final models separates class 0 and 1
output_folder_4 = Path("Plots/Tuned Models Handling Balanced Class")
output_folder_4.mkdir(parents=True, exist_ok=True)

# ROC data
fpr_logreg_tuned, tpr_logreg_tuned, _ = roc_curve(y_test, pred_proba_logreg)
fpr_lgbm_tuned, tpr_lgbm_tuned, _ = roc_curve(y_test, pred_proba_lgbm)
fpr_cb_tuned, tpr_cb_tuned, _ = roc_curve(y_test, pred_proba_cb)

# AUC
roc_auc_logreg = roc_auc_score(y_test, pred_proba_logreg)
roc_auc_lgbm = roc_auc_score(y_test, pred_proba_lgbm)
roc_auc_cb = roc_auc_score(y_test, pred_proba_cb)

# Plot
plt.figure(figsize=(18,10))

plt.plot(fpr_logreg_tuned, tpr_logreg_tuned,
         label=f"Logistic Regression (AUC = {roc_auc_logreg:.4f})")

plt.plot(fpr_lgbm_tuned, tpr_lgbm_tuned,
         label=f"LightGBM (AUC = {roc_auc_lgbm:.4f})")

plt.plot(fpr_cb_tuned, tpr_cb_tuned,
         label=f"CatBoost (AUC = {roc_auc_cb:.4f})")

# Random classifier
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Tuned Models")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

save_plot(output_folder_4, "12_roc_curve_tuned_models.png")

# visualized base model evaluation for confusion matrix
fig, axes = plt.subplots(1,3,figsize=(10, 7))

ConfusionMatrixDisplay.from_predictions(y_test, pred_logreg, cmap="Blues", ax=axes[0], colorbar=False, display_labels=['No', 'Yes'])
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(y_test, pred_lgbm, cmap="Blues", ax=axes[1], colorbar=False, display_labels=['No', 'Yes'])
axes[1].set_title("LightGBM Classification")

ConfusionMatrixDisplay.from_predictions(y_test, pred_cb, cmap="Blues", ax=axes[2], colorbar=False, display_labels=['No', 'Yes'])
axes[2].set_title("Cat Boost Classification")

plt.suptitle("CONFUSION MATRIX (Tuned Models)", fontsize=16, fontweight='bold', y=0.8)
plt.tight_layout()
save_plot(output_folder_4, "13_confusion_matrix_tuned_models.png")

console.print("\n___________________________________________________TUNED MODEL CLASS HANDLING ANALYSIS ___________________________________________________\n")
console.print(f"\nAnalysis Plots saved to: {output_folder_4.absolute()}")
console.print("__________________________________________________________________________________________________________________________________________\n")

# Complete and immediately stop the progress bar — stopping right away (rather than
# after the BEST MODEL section below) prevents the live display from repainting a
# leftover 100% frame when later console.print()/console.rule() calls run
progress.update(task, completed=100, description="[bold green]Customer Churn Prediction Completed!")
progress.stop()

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

models = {
           "Logistic Regression": roc_auc_score(y_test, pred_proba_logreg),
           "LightGBM Classifier": roc_auc_score(y_test, pred_proba_lgbm),
           "Cat Boost Classifier": roc_auc_score(y_test, pred_proba_cb)
         }

best_model = max(models, key=models.get)

console.rule("[bold green]BEST MODEL SELECTED")

console.print(
    f"[bold cyan]Best Model:[/bold cyan] "
    f"[bold yellow]{best_model}[/bold yellow]"
)

console.print(
    f"[bold cyan]ROC-AUC:[/bold cyan] "
    f"[bold green]{models[best_model]:.5f}[/bold green]"
)

# Calculate total execution time
elapsed = time.perf_counter() - start_time

hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60

console.rule("[bold magenta]PROGRAM FINISHED")

console.print(
    f"[bold green]Execution Time:[/bold green] "
    f"{hours} hr {minutes} min {seconds:.2f} sec"
)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''