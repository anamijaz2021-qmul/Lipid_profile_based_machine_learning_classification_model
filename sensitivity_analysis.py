"""
PESSI-only sensitivity analysis
--------------------------------
Purpose: address Reviewer 2 comment 2/6 (inter-laboratory heterogeneity).
Restrict the dataset to the single-method (Friedewald / PESSI) subgroup,
removing the Evercare (direct-method) samples -- which, per the retro-coding
check, contain 0 Controls and are therefore a pure confound with outcome.

Methodology mirrors the manuscript's main pipeline as closely as possible:
- 80/20 stratified train/test split (per user's simplification choice)
- SMOTE applied to training set only
- StandardScaler
- GridSearchCV, 5-fold StratifiedKFold, scoring='roc_auc' (same grids as main script)
- Target encoding: Cases = 1, Controls = 0 (consistent with main pipeline)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 0. Locate / load the data file
# ---------------------------------------------------------------
# EDIT THIS if you already know the path on your machine, e.g.:
# DATA_PATH = "/content/Supplementary_Data_S1_with_Site.xlsx"       # Colab, after upload
# DATA_PATH = "C:/Users/you/Documents/Supplementary_Data_S1_with_Site.xlsx"  # Windows
# DATA_PATH = "./Supplementary_Data_S1_with_Site.xlsx"              # same folder as notebook
DATA_PATH = "Supplementary_Data_S1_with_Site.xlsx"

FILENAME = os.path.basename(DATA_PATH)

def _load_via_colab_upload():
    """If running in Google Colab, pop up a file picker and use whatever is uploaded."""
    from google.colab import files
    print("Please select 'Supplementary_Data_S1_with_Site.xlsx' from your computer...")
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No file was uploaded.")
    return next(iter(uploaded.keys()))

def resolve_data_path(path):
    # 1) Use the path as-is if it already exists.
    if os.path.exists(path):
        return path

    # 2) If we're in Google Colab, offer an upload widget.
    try:
        import google.colab  # noqa: F401
        print(f"'{path}' not found.")
        uploaded_name = _load_via_colab_upload()
        return uploaded_name
    except ImportError:
        pass  # not running in Colab

    # 3) Not in Colab and not found: fall back to a plain input() prompt,
    #    so this also works in a local Jupyter / terminal session.
    print(f"Could not find '{path}'.")
    typed = input(
        "Enter the full path to Supplementary_Data_S1_with_Site.xlsx "
        "(or press Enter to search the current folder): "
    ).strip()
    if typed:
        if os.path.exists(typed):
            return typed
        raise FileNotFoundError(f"File not found at the path you entered: {typed}")

    # 4) Last resort: look for a same-named file anywhere under the cwd.
    for root, _dirs, fnames in os.walk("."):
        if FILENAME in fnames:
            found = os.path.join(root, FILENAME)
            print(f"Found file at: {found}")
            return found

    raise FileNotFoundError(
        f"Could not locate '{FILENAME}'. Place it in the working directory, "
        f"update DATA_PATH at the top of this script, or upload it when prompted."
    )

DATA_PATH = resolve_data_path(DATA_PATH)
print(f"Loading data from: {DATA_PATH}\n")

# ---------------------------------------------------------------
# 1. Load data + retro-coded Center column, restrict to PESSI only
# ---------------------------------------------------------------
df = pd.read_excel(DATA_PATH)

pessi = df[df['Center'] == 'PESSI'].drop(columns=['Center']).copy()
print(f"PESSI-only subset: {pessi.shape[0]} samples")
print(pessi['Samples'].value_counts())
print()

# Mean-impute numeric missing values (mirrors main pipeline's imputation step)
num_cols = pessi.select_dtypes(include=['number']).columns
for col in num_cols:
    if pessi[col].isnull().any():
        n_missing = pessi[col].isnull().sum()
        pessi[col] = pessi[col].fillna(pessi[col].mean())
        print(f"Imputed {n_missing} missing value(s) in '{col}' with mean.")
print()

target_col = 'Samples'
X = pessi.drop(columns=[target_col])
y = pessi[target_col].map({'Cases': 1, 'Controls': 0})

# One-hot encode any remaining categoricals (none expected -- all numeric already)
X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------------------------
# 2. 80/20 stratified split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]}  (Cases={sum(y_train==1)}, Controls={sum(y_train==0)})")
print(f"Test:  {X_test.shape[0]}  (Cases={sum(y_test==1)}, Controls={sum(y_test==0)})")
print()

# ---------------------------------------------------------------
# 3. SMOTE on training set only
# ---------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE -- train: {X_train_res.shape[0]} "
      f"(Cases={sum(y_train_res==1)}, Controls={sum(y_train_res==0)})")
print()

# ---------------------------------------------------------------
# 3b. Build the Evercare set as a second, fully unseen holdout
#     (the model is trained ONLY on PESSI; Evercare is never touched
#     during training, tuning, or SMOTE)
# ---------------------------------------------------------------
evercare = df[df['Center'] == 'Evercare'].drop(columns=['Center']).copy()
for col in evercare.select_dtypes(include=['number']).columns:
    if evercare[col].isnull().any():
        evercare[col] = evercare[col].fillna(evercare[col].mean())

X_evercare = evercare.drop(columns=[target_col])
y_evercare = evercare[target_col].map({'Cases': 1, 'Controls': 0})
X_evercare = pd.get_dummies(X_evercare, drop_first=True)
# align columns exactly with training feature set (in case of dummy mismatches)
X_evercare = X_evercare.reindex(columns=X.columns, fill_value=0)

print(f"Evercare unseen holdout: {X_evercare.shape[0]} samples "
      f"(Cases={sum(y_evercare==1)}, Controls={sum(y_evercare==0)})")
print()

# ---------------------------------------------------------------
# 4. Model configs (same param grids as main manuscript pipeline)
# ---------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    'Logistic Regression': (
        Pipeline([('scaler', StandardScaler()),
                  ('clf', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))]),
        {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100], 'clf__penalty': ['l1', 'l2']}
    ),
    'Decision Tree': (
        Pipeline([('scaler', StandardScaler()),
                  ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        {'clf__max_depth': [None, 5, 10, 15, 20],
         'clf__min_samples_split': [2, 5, 10],
         'clf__min_samples_leaf': [1, 2, 4],
         'clf__criterion': ['gini', 'entropy']}
    ),
    'SVM (RBF)': (
        Pipeline([('scaler', StandardScaler()),
                  ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))]),
        {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': [0.001, 0.01, 0.1, 1]}
    ),
    'XGBoost': (
        Pipeline([('scaler', StandardScaler()),
                  ('clf', XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE))]),
        {'clf__n_estimators': [50, 100, 200],
         'clf__learning_rate': [0.01, 0.1, 0.2],
         'clf__max_depth': [3, 5, 7],
         'clf__subsample': [0.7, 0.8, 1.0],
         'clf__colsample_bytree': [0.7, 0.8, 1.0]}
    ),
    'LightGBM': (
        Pipeline([('scaler', StandardScaler()),
                  ('clf', LGBMClassifier(random_state=RANDOM_STATE, objective='binary', verbose=-1))]),
        {'clf__n_estimators': [50, 100, 200],
         'clf__learning_rate': [0.01, 0.1, 0.2],
         'clf__num_leaves': [20, 31, 40],
         'clf__max_depth': [-1, 5, 10],
         'clf__subsample': [0.7, 0.8, 1.0],
         'clf__colsample_bytree': [0.7, 0.8, 1.0]}
    ),
}

# ---------------------------------------------------------------
# 5. Fit + evaluate
# ---------------------------------------------------------------
results = []
best_params_log = {}

for name, (pipe, grid) in models.items():
    print(f"--- {name} ---")
    gs = GridSearchCV(pipe, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_res, y_train_res)
    best_params_log[name] = gs.best_params_
    print("Best params:", gs.best_params_)

    best_model = gs.best_estimator_

    def eval_partition(X_eval, y_eval, partition_label):
        y_pred = best_model.predict(X_eval)
        y_proba = best_model.predict_proba(X_eval)[:, 1]
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        auc = roc_auc_score(y_eval, y_proba)
        cm = confusion_matrix(y_eval, y_pred)
        print(f"[{partition_label}] Accuracy: {acc:.4f}  Precision: {prec:.4f}  "
              f"Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
        print(f"[{partition_label}] Confusion matrix:\n", cm)
        return {
            'Model': name, 'Partition': partition_label,
            'N': len(y_eval), 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'AUC': auc,
            'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1]
        }

    # Training partition = SMOTE-resampled training data (what the model was fit on)
    results.append(eval_partition(X_train_res, y_train_res, 'Train (SMOTE-resampled)'))
    # Test partition = untouched 20% held-out set (within PESSI)
    results.append(eval_partition(X_test, y_test, 'Test (held-out 20% PESSI)'))

    # ---- Evercare as a second, fully unseen holdout ----------------
    # NB: Evercare contains Cases only (n=257), no Controls. AUC/Precision/
    # Recall-for-Controls are therefore undefined. We report what IS
    # computable: how many of these unseen Case patients the PESSI-trained
    # model still correctly flags as Cases.
    y_pred_ev = best_model.predict(X_evercare)
    n_ev = len(y_evercare)
    correctly_flagged = int((y_pred_ev == 1).sum())   # predicted Case
    misclassified = int((y_pred_ev == 0).sum())        # predicted Control (wrong)
    acc_ev = accuracy_score(y_evercare, y_pred_ev)      # = recall/sensitivity here, single class
    print(f"[Evercare unseen holdout, Cases-only, N={n_ev}] "
          f"Correctly flagged as Case: {correctly_flagged}/{n_ev}  "
          f"(Accuracy=Sensitivity: {acc_ev:.4f})")
    print(f"[Evercare unseen holdout] Misclassified as Control (false negatives): {misclassified}")
    print("NOTE: AUC/Precision/F1 undefined -- Evercare holdout contains only the positive (Case) class.")

    results.append({
        'Model': name, 'Partition': 'Evercare (unseen, Cases-only holdout)',
        'N': n_ev, 'Accuracy': acc_ev, 'Precision': np.nan,
        'Recall': acc_ev, 'F1': np.nan, 'AUC': np.nan,
        'TN': np.nan, 'FP': np.nan, 'FN': misclassified, 'TP': correctly_flagged
    })
    print()

results_df = pd.DataFrame(results)
print("=" * 90)
print("SUMMARY -- PESSI-only sensitivity analysis (Train vs. Test)")
print("=" * 90)
print(results_df.to_string(index=False))

# Save outputs next to the input file's folder if /mnt/user-data/outputs doesn't exist
OUT_DIR = "/mnt/user-data/outputs"
if not os.path.isdir(OUT_DIR):
    OUT_DIR = os.path.dirname(os.path.abspath(DATA_PATH)) or "."

results_df.to_csv(os.path.join(OUT_DIR, "PESSI_only_sensitivity_results.csv"), index=False)

# Wide-format pivot for easy side-by-side comparison
pivot = results_df.pivot(index='Model', columns='Partition',
                          values=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
pivot.to_csv(os.path.join(OUT_DIR, "PESSI_only_sensitivity_results_wide.csv"))
print("\nWide-format table:")
print(pivot.to_string())

with open(os.path.join(OUT_DIR, "PESSI_only_best_params.txt"), 'w') as f:
    for name, params in best_params_log.items():
        f.write(f"{name}: {params}\n")

print(f"\nSaved results to: {OUT_DIR}")
