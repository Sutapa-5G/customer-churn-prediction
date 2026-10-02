# Customer Churn Prediction

## Overview

This project builds a machine learning pipeline to estimate whether a telecommunications
customer is likely to churn (stop using the service), based on demographic, account, and
service-related attributes.

Churn prediction is a widely used application of machine learning in subscription businesses:
acquiring a new customer is typically far more expensive than retaining an existing one, so
identifying at-risk customers in advance lets a retention team prioritize outreach. This project
does not claim to solve that business problem for any specific company — it demonstrates the
end-to-end methodology, on a public benchmark dataset, with particular attention to avoiding data
leakage and to being honest about what the results do and do not show.

**What this project demonstrates:**
- Exploratory data analysis grounded in the actual dataset
- A leakage-safe preprocessing and modeling pipeline (train/test split before any fitting,
  preprocessing fit only on training data, SMOTE confined to training folds)
- Comparison of three classification models with cross-validation and hyperparameter tuning
- Honest, multi-metric evaluation (not accuracy alone) on a held-out test set
- Model explainability with SHAP, with careful, verified claims about what the values represent
- A working Streamlit application for interactive, single-customer predictions

## Project Highlights

- 📊 **Exploratory Data Analysis** — target distribution, churn patterns by contract/tenure/
  charges/payment method, correlation analysis, all computed from the actual data (see
  `notebooks/01_data_exploration.ipynb`)
- 🧹 **Leakage-safe preprocessing** — stratified train/test split performed before any
  data-dependent fitting; imputers/scaler/encoder fit only on training data
- ⚖️ **Class imbalance handling with SMOTE** — applied only inside the training pipeline/CV folds,
  never to validation or test data
- 🤖 **Multiple model comparison** — Logistic Regression, Random Forest, XGBoost, trained through
  an identical pipeline architecture for a fair comparison
- 🔁 **Cross-validation** — stratified 5-fold (`shuffle=True, random_state=42`) on training data
  only, for both baseline and tuned models
- 🎛️ **Hyperparameter tuning** — `RandomizedSearchCV` over the full pipeline (preprocessing → SMOTE
  → classifier), scored on ROC-AUC, never touching the test set
- 📈 **Multi-metric evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, and full confusion
  matrices on a test set touched exactly once
- 🔍 **SHAP explainability** — global importance, beeswarm, and individual-prediction explanations,
  with the SHAP output space (log-odds, not probability) verified empirically rather than assumed
- 🖥️ **Interactive Streamlit application** — single-customer churn prediction with probability,
  SHAP-based explanation, and clear disclaimers

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                    # Original, unmodified dataset
│   └── processed/              # Intentionally empty — see "Data Leakage Prevention" below
│
├── notebooks/
│   ├── 01_data_exploration.ipynb            # EDA
│   ├── 02_data_preprocessing.ipynb          # Cleaning, pipeline, SMOTE architecture
│   ├── 03_model_training_evaluation.ipynb   # Training, CV, tuning, test evaluation
│   └── 04_model_explainability_shap.ipynb   # SHAP explainability
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # load/clean/split/preprocess + validation checks
│   ├── train.py                # end-to-end training script (python -m src.train)
│   ├── evaluate.py             # metrics, plots, cross-validation, threshold analysis
│   └── explain.py              # SHAP explainability (python -m src.explain)
│
├── models/
│   ├── churn_model.pkl          # Final fitted pipeline (preprocessing + SMOTE config + model)
│   └── model_metadata.json      # Selection criterion, hyperparameters, metrics, threshold
│
├── reports/
│   ├── figures/                 # All generated plots (EDA, confusion matrices, ROC, SHAP, ...)
│   ├── baseline_cv_results.csv
│   ├── tuned_cv_results.csv
│   ├── test_results.csv
│   ├── threshold_analysis.csv
│   ├── shap_feature_importance.csv
│   └── shap_importance_by_original_feature.csv
│
├── app/
│   └── app.py                   # Streamlit application
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Dataset

- **Name:** IBM Telco Customer Churn (a widely used public benchmark dataset)
- **Location in this repo:** `data/raw/Telco-Customer-Churn.csv`
- **Rows:** 7,043 · **Columns:** 21
- **Target variable:** `Churn` (`Yes`/`No`, encoded to `1`/`0` during preprocessing)
- **Class distribution:** No = 5,174 (73.46%), Yes = 1,869 (26.54%) — a moderately imbalanced
  target, which is why SMOTE is used during training

**Data quality, as actually found (Notebook 1 & 2):**
- No `NaN` values and no duplicate rows or duplicate `customerID` values were found.
- `TotalCharges` is stored as text and had **11 blank (whitespace) values**, none of which are
  caught by a plain `.isnull()` check. All 11 correspond to customers with `tenure == 0`.
  These were filled with `0`, based on the domain rule that `TotalCharges ≈ MonthlyCharges ×
  tenure`, which is exactly `0` for a brand-new customer — a fixed rule, not a statistic learned
  from the data, so it introduces no leakage risk between the train/test split.
- `customerID` is a unique identifier with no predictive value and is dropped before modeling; it
  is confirmed absent from the feature set by an explicit check in the pipeline.

No source URL is asserted here beyond the dataset's name, since none was independently verified as
part of this project; if you are sourcing the file yourself, place it at
`data/raw/Telco-Customer-Churn.csv`.

## Project Workflow

```
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning (TotalCharges → numeric, drop customerID, encode Churn)
   ↓
Train/Test Split (stratified, 80/20, random_state=42)
   ↓
Preprocessing (impute, scale, one-hot encode — fit on training data only)
   ↓
SMOTE on Training Data (inside the pipeline, never on validation/test data)
   ↓
Model Training (Logistic Regression, Random Forest, XGBoost)
   ↓
Cross-Validation & Hyperparameter Tuning (training data only)
   ↓
Test Evaluation (held-out test set, touched once)
   ↓
SHAP Explainability
   ↓
Streamlit Application
```

### Data leakage prevention

The central engineering principle of this project: **data-dependent preprocessing is fitted only
on training data, while SMOTE is applied inside the imbalanced-learn pipeline so synthetic samples
are not created from the test set.**

Concretely:
- The stratified train/test split happens **before** any imputer, scaler, or encoder is fit.
- The `ColumnTransformer` (imputation + `StandardScaler` + `OneHotEncoder`) is fit only on
  `X_train`; the test set is only ever `.transform()`-ed. This was verified directly — the fitted
  scaler's mean for `tenure` was checked to equal the train-only mean and not the full-dataset
  mean.
- SMOTE lives inside an `imblearn.pipeline.Pipeline`, which — unlike a plain scikit-learn
  pipeline — knows to apply resampling only during `.fit()`/`.fit_resample()` on training folds,
  and to skip it automatically during `.transform()`/`.predict()`. This was also verified directly:
  predicting on the test set produced exactly as many predictions as test rows, confirming SMOTE
  never resampled it.
- `data/processed/` is deliberately left empty: saving a globally-fit transformed dataset there
  and reusing it for modeling would silently leak test-set information into the encoder/scaler.
  The only artifact ever persisted is the pipeline **fitted on the training split**, in
  `models/churn_model.pkl`.
- Hyperparameter tuning (`RandomizedSearchCV`) was run over the **entire pipeline** (not just the
  classifier), so every candidate configuration was refit with preprocessing and SMOTE inside each
  cross-validation fold — the test set played no role in tuning.
- The classification threshold was selected using out-of-fold cross-validated predictions on the
  training set (`cross_val_predict`), not the test set.

## Modeling

Three classifiers were trained through an identical pipeline architecture
(`preprocessing → SMOTE → classifier`), so the comparison isolates the effect of the algorithm
itself:

- **Logistic Regression** — a linear baseline; coefficients are directly interpretable, which
  matters when a retention team needs to understand why a customer was flagged.
- **Random Forest** — an ensemble of decision trees that captures non-linear relationships and
  feature interactions without much tuning effort.
- **XGBoost** — gradient boosting, typically strong on tabular data, with more hyperparameters and
  lower direct interpretability (addressed here with SHAP).

**Evaluation used multiple metrics, not accuracy alone:** Accuracy, Precision, Recall, F1, and
ROC-AUC, plus full confusion matrices. With a ~26.5% churn rate, a model predicting "no churn" for
every customer would score ~73.5% accuracy while catching zero churners — which is why ROC-AUC
(threshold-independent, robust to imbalance) was used as the primary model-selection metric, and
why precision/recall/F1 are reported alongside it rather than accuracy alone.

**Cross-validation:** stratified 5-fold (`StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)`), run on the training set only, with SMOTE re-applied inside each fold via the
pipeline.

## Results

All figures below come directly from `reports/tuned_cv_results.csv` and `reports/test_results.csv`
(generated by `src/train.py` and reproduced in `notebooks/03_model_training_evaluation.ipynb`) —
nothing here is hand-entered or estimated.

### Cross-validation results (5-fold, tuned models, training data only)

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7556 | 0.5265 | 0.7860 | 0.6304 | 0.8451 |
| Random Forest | 0.7774 | 0.5626 | 0.7284 | 0.6346 | 0.8433 |
| XGBoost | 0.7934 | 0.5954 | 0.6916 | 0.6398 | 0.8470 |

### Test-set results (held out, evaluated once, threshold = 0.50)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7367 | 0.5025 | 0.7941 | 0.6155 | 0.8394 |
| Random Forest | 0.7637 | 0.5416 | 0.7139 | 0.6159 | 0.8372 |
| XGBoost | 0.7736 | 0.5599 | 0.6872 | 0.6170 | 0.8427 |

Confusion matrices for all three models and a combined ROC curve are in `reports/figures/`
(`confusion_matrix_*.png`, `roc_curve_comparison.png`).

### Model selection

**Selection criterion (declared before results were examined):** highest mean 5-fold
cross-validation ROC-AUC on the training data.

Under that criterion, tuned XGBoost achieved the highest score — **0.8470** — versus Logistic
Regression's **0.8451** and Random Forest's **0.8433**. The difference is small: it is smaller than
XGBoost's own fold-to-fold standard deviation (±0.0102), so **the three models showed similar
discrimination performance**; this should not be read as XGBoost being clearly or universally
superior. XGBoost was retained as the final model because it achieved the highest mean CV ROC-AUC
under the predefined selection criterion — not because it dominated the others.

It's also worth noting that **Logistic Regression had higher test recall** (0.7941) than XGBoost
(0.6872) — meaning it missed fewer actual churners (77 false negatives vs. 117). Which model is
preferable in practice depends on the operational objective and the relative cost of a missed
churner (false negative) versus a wasted retention offer (false positive) — a trade-off this
project does not attempt to resolve, since no real business cost figures are available for this
public dataset.

### Threshold analysis

Threshold behavior was investigated using **out-of-fold cross-validated predictions on the
training set** (never the test set), for the selected model:

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.30 | 0.4914 | 0.8415 | 0.6205 |
| 0.40 | 0.5381 | 0.7645 | 0.6317 |
| 0.50 | 0.5953 | 0.6916 | **0.6399** |
| 0.60 | 0.6381 | 0.5826 | 0.6091 |
| 0.70 | 0.6937 | 0.4515 | 0.5470 |

**0.50 maximized F1** among the thresholds tested and is the default used by the app and in the
reported test metrics above. Lower thresholds trade precision for recall — e.g., 0.30 catches more
churners (recall 0.84) at the cost of many more false positives (precision drops to 0.49). Which
threshold is "right" is a business decision (how expensive is a missed churner vs. a wasted
retention offer?), not something this project decides on the dataset's behalf.

## SHAP Explainability

- **SHAP version:** 0.52.0
- **Model explained:** XGBoost (the selected final model)
- **Explainer:** `TreeExplainer` (chosen because the classifier is genuinely tree-based; a
  non-tree model would use the model-agnostic `shap.Explainer` instead — see `src/explain.py`)
- **Transformed features:** 45 (from 19 original columns, after one-hot encoding)
- **Rows explained:** 500, sampled reproducibly (`random_state=42`) from the held-out test set,
  used strictly post-hoc — no test labels were used, and nothing here fed back into model
  selection, tuning, or thresholding

**Important:** for this XGBoost model, SHAP values were empirically verified to be in **raw
log-odds (margin) space, not probability** (`sum(SHAP values) + base_value` was checked to equal
the model's raw margin output, and its sigmoid was checked to equal the predicted probability, for
every explained row). A SHAP value of `+0.40` means "+0.40 log-odds toward churn" — it is **not**
"+40% probability."

**Top 10 features by mean absolute SHAP value** (i.e., the features the model relied on most
heavily when producing predictions — see `reports/shap_feature_importance.csv` and
`reports/figures/shap_global_importance.png` / `shap_beeswarm.png`):

1. `Contract_Month-to-month`
2. `tenure`
3. `OnlineSecurity_No`
4. `InternetService_Fiber optic`
5. `PaymentMethod_Electronic check`
6. `TechSupport_No`
7. `MonthlyCharges`
8. `PaperlessBilling_No`
9. `Contract_Two year`
10. `StreamingMovies_Yes`

These are features the model relied on strongly when producing its predictions — **not** causal
factors. SHAP explains model behavior on this dataset; it cannot distinguish a genuine driver from
a proxy or a correlated variable, and establishing causation would require a controlled experiment,
not observational data like this. Three distinct claim types are kept separate throughout this
project: *observed dataset relationships* (Notebook 1), *model behavior* (Notebook 4, this
section), and *causal claims* (not supported by this project).

## Running the Streamlit App

### Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Launch

From the **project root**:

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. If `models/churn_model.pkl` is missing, regenerate it
with `python -m src.train` first (this repo already includes the trained model, so this step is
only needed if you retrain).

### What it does

The app loads the already-fitted pipeline from `models/churn_model.pkl` for inference only — it
never retrains or overwrites the model on startup. You fill in one customer's attributes
(demographics, account information, and services — grouped to mirror the dataset's structure), and
the app returns an estimated churn probability and predicted class at the project's documented
threshold of 0.50, along with an optional SHAP-based explanation of that specific prediction.

No personally identifying information is collected or required; `customerID` was dropped before
modeling and is never requested. All computation runs locally.

**A note on interpretation, reflected throughout the app:** the prediction is a statistical
estimate from historical patterns in a public benchmark dataset, not a certainty about any
individual customer, and the SHAP explanation describes model behavior, not causation.

## Reproducibility

- `random_state=42` is used consistently for the train/test split, cross-validation folds, SMOTE,
  hyperparameter search, and SHAP row sampling.
- All notebooks execute top-to-bottom without manual intervention and were verified to run with
  zero errors and zero warnings (including with `FutureWarning`/`DeprecationWarning` escalated to
  exceptions where checked).
- `src/train.py` (`python -m src.train`) and `src/explain.py` (`python -m src.explain`) reproduce
  the saved model and SHAP outputs respectively from the raw CSV, end to end.
- The raw dataset is never modified by any script or notebook in this repository; this was
  explicitly re-verified (file hash unchanged) after every step of this project, including after
  building and testing the Streamlit app.

### Repository conventions (see `.gitignore` for full reasoning)

The dataset, trained model, and all report figures/tables are **intentionally committed** to this
repository (not gitignored), so that cloning it is enough to explore the results and run the app
without first re-running the training pipeline. Standard ephemeral artifacts (`__pycache__/`,
`.ipynb_checkpoints/`, virtual environments, OS/IDE files) are ignored as usual.

## Limitations

- **No time dimension.** This is a static, single-snapshot dataset. A production churn system
  would use time-based validation (train on the past, validate on the future) to avoid
  inadvertently learning from information that wouldn't be available at prediction time.
- **Public benchmark, not a specific business's customers.** Patterns learned here (e.g., around
  contract type or payment method) may not transfer to another company's customer base, pricing,
  or market conditions.
- **Model performance is not decisively separated.** The three models showed similar CV ROC-AUC
  performance, and the observed differences were small relative to fold-to-fold variation; a
  different reasonable analyst could defensibly pick Logistic Regression for its higher recall and
  interpretability instead of XGBoost.
- **No cost-sensitive analysis.** The threshold and model comparisons do not incorporate a real
  ratio between retention-offer cost and customer lifetime value, because no such figures exist
  for this public dataset — a real deployment would need them to choose an operating point.
- **`RandomizedSearchCV` samples the hyperparameter space** rather than exhaustively searching it,
  by design (kept computationally reasonable rather than maximizing score).
- **SHAP explains the model, not the world.** It faithfully describes what a model relies on, even
  if the model itself is imperfect; it cannot confirm causation, and its credit allocation among
  correlated features is not unique.

## Future Improvements

- Incorporate time-based (rolling-origin) validation if longitudinal data becomes available.
- Add a cost-sensitive evaluation once real retention-offer cost and customer lifetime value
  figures are available, to choose a threshold on business terms rather than F1 alone.
- Expand hyperparameter search (e.g., Bayesian optimization) if a larger compute budget is
  acceptable.
- Add model monitoring / drift detection for a deployed setting.
- Extend the Streamlit app with batch (CSV upload) predictions in addition to the single-customer
  form.

## License

This project's original source code is released under the MIT License — see `LICENSE`. The
bundled dataset is the public IBM Telco Customer Churn dataset, included here for reproducibility
of this educational/portfolio project rather than relicensed.
