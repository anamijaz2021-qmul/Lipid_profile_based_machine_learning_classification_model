# Lipid Profile-Based Machine Learning Classification of Ischemic Heart Disease

This repository contains the full analysis code, data dictionary, and sensitivity
analyses supporting the manuscript *"Lipid Profile-Based Machine Learning
Classification of Ischemic Heart Disease Using Comparative Classification
Algorithms."* Five supervised machine learning classifiers are trained and
evaluated on routinely available lipid, anthropometric, and clinical variables
to distinguish ischemic heart disease (IHD) Cases from Controls.

---

## Overview
| **Study design** | Retrospective case-control |
| **Cohort** | 929 participants (710 Cases, 219 Controls) recruited from two tertiary care centres in Lahore, Pakistan 
| **Features** | Gender, Age, BMI, Diabetes, Hypertension, Smoking, Triglycerides (TG), VLDL, LDL, HDL |
| **Target** | `Samples` — Cases (IHD) vs. Controls |
| **Models** | Logistic Regression, Decision Tree, SVM (RBF kernel), XGBoost, LightGBM |
| **Partitioning** | Stratified 70% train / 20% test / 10% held-out validation |
| **Class imbalance handling** | SMOTE applied to the training partition only |
| **Hyperparameter tuning** | GridSearchCV, 5-fold stratified cross-validation, `scoring='roc_auc'` |
| **Interpretability** | SHAP (SHapley Additive exPlanations) |

---

## Repository structure

```
├── data/
│   └── Supplementary_Data_S1.xlsx        # De-identified analysis dataset (929 x 11)
├── Lipid_profile_ML_classification.py    # Primary pipeline: preprocessing, 5-model
│                                          #   training/tuning, evaluation, SHAP
├── sensitivity_analysis/
│   ├── pessi_sensitivity_analysis.py     # Centre-attribution + cross-platform
│   │                                     #   sensitivity analysis (see below)
│   └── PESSI_only_sensitivity_results.csv
├── requirements.txt
└── README.md


## Primary pipeline (`Lipid_profile_ML_classification.py`)

1. **Data loading & cleaning** — loads `Supplementary_Data_S1.xlsx`; missing values
   (numeric) are mean-imputed.
2. **Target encoding** — `Cases = 1`, `Controls = 0`. Encoding is verified
   programmatically via assertion checks prior to model training, and all five
   classifiers share a common preprocessing/encoding module to prevent divergent
   label handling across models.
3. **Splitting** — stratified split into 70% train / 20% test / 10% validation
   (`random_state=42`), preserving the overall Case:Control ratio in each partition.
4. **Resampling** — SMOTE applied to the training partition only; test and
   validation partitions retain the natural class distribution.
5. **Scaling** — `StandardScaler`, fit on the training partition only.
6. **Model training & tuning** — each of the five classifiers is tuned via
   `GridSearchCV` (5-fold stratified CV, `scoring='roc_auc'`) over a
   model-specific hyperparameter grid.
7. **Evaluation** — accuracy, precision, recall, F1, ROC AUC, and PR AUC are
   reported for all three partitions; confusion matrices, ROC curves, and
   calibration plots (Brier score) are generated per model.
8. **Statistical comparison** — pairwise McNemar's test across all model pairs,
   computed separately per partition. *Note: McNemar's test results on the
   SMOTE-resampled training partition are reported for completeness only, as
   synthetic minority-class samples lack a real paired counterpart; the test
   (20%) and validation (10%) partition results are the statistically valid
   comparisons.*
9. **Feature importance** — SHAP values computed for all five models.


Four of five classifiers retained high sensitivity on entirely unseen,
cross-platform data; Logistic Regression's linear boundary did not generalize
as well across the platform-associated shift in feature distributions.

---

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
matplotlib
seaborn
openpyxl
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Primary pipeline
python Lipid_profile_ML_classification.py

# Sensitivity analysis (requires data/Supplementary_Data_S1.xlsx)
python sensitivity_analysis/pessi_sensitivity_analysis.py
```

---

## Data availability

`Supplementary_Data_S1.xlsx` contains de-identified patient-level data (no
direct identifiers). Recruitment centre/site was not recorded as a variable in
the original dataset; the `Center` label used in the sensitivity analysis is a
retrospective attribution derived from the LDL/VLDL quantification method, as
described above, and is provided for transparency and reproducibility.

---

## Citation

If you use this code or data, please cite:

> [Author list]. Lipid Profile-Based Machine Learning Classification of
> Ischemic Heart Disease Using Comparative Classification Algorithms.
> [Journal, year, DOI — update upon acceptance]

---

## License

[Specify license, e.g., MIT, CC-BY-4.0 — add before publishing]

---

## Contact

For questions regarding this repository, please contact anamijaz2021@gmail.com.
