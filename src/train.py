"""
train.py

End-to-end, leakage-safe training script for the Customer Churn Prediction
project.

Run from the project root with:

    python -m src.train

Workflow enforced by this script:

    raw data
      -> clean_data()
      -> X / y separation
      -> stratified train/test split          (test set sealed from here on)
      -> [per model] preprocessing -> SMOTE -> classifier
      -> stratified 5-fold CV on TRAINING data only   (model selection)
      -> hyperparameter search on TRAINING data only  (tuning)
      -> threshold selection from out-of-fold TRAINING predictions
      -> ONE final evaluation on the untouched test set
      -> save fitted pipeline + metadata + results

The test set is never used to fit preprocessing, to resample with SMOTE, to
cross-validate, to tune hyperparameters, or to choose the decision threshold.
It is touched exactly once, at the very end, to produce the reported
generalization estimate.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

# Support both `python -m src.train` (package-relative) and direct execution
try:
    from src import data_preprocessing as dp
    from src import evaluate as ev
except ImportError:  # pragma: no cover - fallback when run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data_preprocessing as dp
    import evaluate as ev


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# The single, pre-declared criterion used to select the final model.
# Declared BEFORE any results are seen, and applied to cross-validation
# results on the training data only - never to test-set results.
SELECTION_METRIC = "roc_auc_mean"
SELECTION_CRITERION_DESCRIPTION = (
    "Highest mean ROC-AUC across stratified 5-fold cross-validation on the "
    "training set. ROC-AUC is chosen because it is threshold-independent and "
    "robust to the class imbalance present in this dataset, unlike accuracy. "
    "Selection uses cross-validation results only; the test set plays no part."
)

THRESHOLDS_TO_EVALUATE = (0.30, 0.40, 0.50, 0.60, 0.70)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_baseline_models() -> dict:
    """Return the three baseline classifiers with sensible initial parameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,           # high enough for reliable convergence on scaled data
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def get_param_distributions(model_name: str) -> dict:
    """Hyperparameter search space per model, keyed for the full pipeline.

    Note the `classifier__` prefix: the search is run over the COMPLETE
    pipeline (preprocessing -> SMOTE -> classifier), so every candidate
    configuration is refit with preprocessing and SMOTE applied inside each
    CV fold.
    """
    if model_name == "Logistic Regression":
        # Note: scikit-learn 1.8 deprecated the explicit `penalty` parameter
        # (it is being replaced by `l1_ratio`/`C`). We therefore tune only `C`,
        # which controls regularisation strength and is the parameter that
        # actually matters here. This keeps the search warning-free and
        # forward-compatible.
        return {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        }
    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [8, 12, 16, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        }
    if model_name == "XGBoost":
        return {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
    raise ValueError(f"No parameter distribution defined for model: {model_name}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 78)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1-4. Load, clean, split ------------------------------------------------
    print("\n[1/9] Loading and cleaning data...")
    raw_df = dp.load_data(RAW_DATA_PATH)
    df_clean = dp.clean_data(raw_df)
    X, y = dp.split_features_target(df_clean)
    X_train, X_test, y_train, y_test = dp.create_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    numerical_features, categorical_features = dp.get_feature_types(X_train)

    print(f"      Raw shape:      {raw_df.shape}")
    print(f"      X_train shape:  {X_train.shape}   churn rate: {y_train.mean():.4f}")
    print(f"      X_test shape:   {X_test.shape}   churn rate: {y_test.mean():.4f}")
    print(f"      Numerical features:   {len(numerical_features)}")
    print(f"      Categorical features: {len(categorical_features)}")
    print("      >>> The test set is now sealed and will not be used again until final evaluation.")

    # -- 5. Build pipelines -----------------------------------------------------
    print("\n[2/9] Building pipelines (preprocessing -> SMOTE -> classifier)...")
    baseline_models = get_baseline_models()
    pipelines = {
        name: dp.build_pipeline(
            preprocessor=dp.build_preprocessor(numerical_features, categorical_features),
            classifier=model,
            use_smote=True,
            random_state=RANDOM_STATE,
        )
        for name, model in baseline_models.items()
    }
    for name in pipelines:
        print(f"      - {name}")

    # -- 6. Baseline cross-validation (TRAINING DATA ONLY) ---------------------
    print(f"\n[3/9] Baseline {N_SPLITS}-fold stratified cross-validation (training data only)...")
    baseline_cv_results = []
    for name, pipeline in pipelines.items():
        print(f"      Cross-validating {name}...")
        result = ev.cross_validate_model(
            pipeline, X_train, y_train,
            n_splits=N_SPLITS, random_state=RANDOM_STATE, model_name=name,
        )
        baseline_cv_results.append(result)

    baseline_cv_df = ev.compare_models(baseline_cv_results)
    print("\n      Baseline CV results (mean +/- std):")
    for row in baseline_cv_results:
        print(
            f"        {row['model']:<22} "
            f"acc={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f}  "
            f"prec={row['precision_mean']:.4f}  "
            f"rec={row['recall_mean']:.4f}  "
            f"f1={row['f1_mean']:.4f}  "
            f"auc={row['roc_auc_mean']:.4f}+/-{row['roc_auc_std']:.4f}"
        )

    # -- 7. Hyperparameter tuning (TRAINING DATA ONLY) -------------------------
    print("\n[4/9] Hyperparameter tuning via RandomizedSearchCV (training data only)...")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tuned_pipelines = {}
    tuned_cv_results = []
    best_params_per_model = {}

    for name, pipeline in pipelines.items():
        print(f"      Tuning {name}...")
        param_dist = get_param_distributions(name)

        # Cap n_iter at the total number of available combinations, otherwise
        # RandomizedSearchCV warns that the search space is smaller than n_iter.
        total_combinations = int(np.prod([len(v) for v in param_dist.values()]))
        n_iter = min(10, total_combinations)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,                # kept computationally reasonable
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        tuned_pipelines[name] = search.best_estimator_
        best_params_per_model[name] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in search.best_params_.items()
        }
        print(f"        best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"        best params:     {search.best_params_}")

        # Re-run the full metric suite on the tuned pipeline, same CV protocol
        tuned_result = ev.cross_validate_model(
            search.best_estimator_, X_train, y_train,
            n_splits=N_SPLITS, random_state=RANDOM_STATE, model_name=name,
        )
        tuned_cv_results.append(tuned_result)

    tuned_cv_df = ev.compare_models(tuned_cv_results)
    print("\n      Tuned CV results (mean +/- std):")
    for row in tuned_cv_results:
        print(
            f"        {row['model']:<22} "
            f"acc={row['accuracy_mean']:.4f}  "
            f"prec={row['precision_mean']:.4f}  "
            f"rec={row['recall_mean']:.4f}  "
            f"f1={row['f1_mean']:.4f}  "
            f"auc={row['roc_auc_mean']:.4f}+/-{row['roc_auc_std']:.4f}"
        )

    # -- 8. Model selection using the PRE-DECLARED criterion --------------------
    print(f"\n[5/9] Selecting final model by criterion: {SELECTION_METRIC} (CV, training data)...")
    best_row = max(tuned_cv_results, key=lambda r: r[SELECTION_METRIC])
    selected_model_name = best_row["model"]
    selected_pipeline = tuned_pipelines[selected_model_name]
    print(f"      Selected: {selected_model_name} "
          f"({SELECTION_METRIC} = {best_row[SELECTION_METRIC]:.4f})")

    # Did tuning actually help for the selected model?
    baseline_for_selected = next(r for r in baseline_cv_results if r["model"] == selected_model_name)
    tuning_delta = best_row[SELECTION_METRIC] - baseline_for_selected[SELECTION_METRIC]
    print(f"      Tuning change in CV {SELECTION_METRIC} for {selected_model_name}: {tuning_delta:+.4f}")

    # -- 9. Threshold analysis using OUT-OF-FOLD TRAINING predictions ----------
    print("\n[6/9] Threshold analysis using out-of-fold CV predictions (training data only)...")
    oof_proba = cross_val_predict(
        selected_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold_df = ev.threshold_analysis(y_train, oof_proba, thresholds=THRESHOLDS_TO_EVALUATE)
    print(threshold_df.to_string(index=False))

    best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    selected_threshold = float(best_f1_row["threshold"])
    print(f"      Threshold maximizing out-of-fold F1: {selected_threshold:.2f} "
          f"(F1={best_f1_row['f1']:.4f}, precision={best_f1_row['precision']:.4f}, "
          f"recall={best_f1_row['recall']:.4f})")
    print("      >>> Threshold chosen from TRAINING out-of-fold predictions, not the test set.")

    # -- 10-12. Fit final models on full training data, evaluate ONCE on test ---
    print("\n[7/9] Final fit on full training data + single evaluation on the sealed test set...")
    test_results = []
    model_probas = {}

    for name, tuned_pipeline in tuned_pipelines.items():
        tuned_pipeline.fit(X_train, y_train)
        y_pred = tuned_pipeline.predict(X_test)
        y_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
        model_probas[name] = (y_test.values, y_proba)

        metrics = ev.calculate_classification_metrics(y_test, y_pred, y_proba, model_name=name)
        test_results.append(metrics)

        ev.plot_confusion_matrix(
            y_test, y_pred, model_name=name,
            save_path=FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png",
        )

    test_df = ev.compare_models(test_results)
    print("\n      Test-set results (default threshold 0.50):")
    print(test_df.to_string(index=False))

    ev.plot_roc_curves(model_probas, save_path=FIGURES_DIR / "roc_curve_comparison.png")
    ev.plot_cv_comparison(tuned_cv_df, save_path=FIGURES_DIR / "cv_comparison.png")

    # Selected model, evaluated at both the default and the chosen threshold
    selected_fitted = tuned_pipelines[selected_model_name]
    selected_proba = selected_fitted.predict_proba(X_test)[:, 1]
    metrics_default = ev.calculate_classification_metrics(
        y_test, (selected_proba >= 0.50).astype(int), selected_proba,
        model_name=f"{selected_model_name} @0.50",
    )
    metrics_selected_threshold = ev.calculate_classification_metrics(
        y_test, (selected_proba >= selected_threshold).astype(int), selected_proba,
        model_name=f"{selected_model_name} @{selected_threshold:.2f}",
    )
    print(f"\n      Selected model at threshold 0.50:  {_fmt(metrics_default)}")
    print(f"      Selected model at threshold {selected_threshold:.2f}:  {_fmt(metrics_selected_threshold)}")

    # -- 13-15. Save pipeline, metadata, results -------------------------------
    print("\n[8/9] Saving fitted pipeline, metadata and results...")
    joblib.dump(selected_fitted, MODEL_PATH)
    print(f"      Model saved to: {MODEL_PATH}")

    fitted_preprocessor = selected_fitted.named_steps["preprocessor"]
    transformed_names = ev.get_transformed_feature_names(fitted_preprocessor)

    metadata = {
        "model_name": selected_model_name,
        "selection_metric": SELECTION_METRIC,
        "selection_criterion": SELECTION_CRITERION_DESCRIPTION,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_n_splits": N_SPLITS,
        "selected_threshold": selected_threshold,
        "threshold_selection_method": (
            "Maximised F1 over out-of-fold cross-validated predictions on the training set."
        ),
        "original_feature_names": list(X_train.columns),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "n_transformed_features": len(transformed_names),
        "best_hyperparameters": best_params_per_model.get(selected_model_name, {}),
        "baseline_cv_results": baseline_cv_results,
        "tuned_cv_results": tuned_cv_results,
        "test_results_default_threshold": test_results,
        "selected_model_test_metrics_at_0.50": metrics_default,
        "selected_model_test_metrics_at_selected_threshold": metrics_selected_threshold,
        "class_balance": {
            "train": {int(k): int(v) for k, v in y_train.value_counts().items()},
            "test": {int(k): int(v) for k, v in y_test.value_counts().items()},
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"      Metadata saved to: {METADATA_PATH}")

    baseline_cv_df.to_csv(REPORTS_DIR / "baseline_cv_results.csv", index=False)
    tuned_cv_df.to_csv(REPORTS_DIR / "tuned_cv_results.csv", index=False)
    test_df.to_csv(REPORTS_DIR / "test_results.csv", index=False)
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    print(f"      Result tables saved to: {REPORTS_DIR}")

    # -- Verify the saved artifact round-trips ---------------------------------
    print("\n[9/9] Verifying the saved model loads and predicts...")
    reloaded = joblib.load(MODEL_PATH)
    check_pred = reloaded.predict(X_test.head(5))
    check_proba = reloaded.predict_proba(X_test.head(5))[:, 1]
    print(f"      Reloaded model predictions (first 5): {check_pred.tolist()}")
    print(f"      Reloaded model probabilities (first 5): {np.round(check_proba, 4).tolist()}")

    print("\n" + "=" * 78)
    print(f"DONE. Final model: {selected_model_name} | criterion: {SELECTION_METRIC} (CV on training data)")
    print("=" * 78)


def _fmt(metrics: dict) -> str:
    return (
        f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} "
        f"[tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}]"
    )


if __name__ == "__main__":
    main()
