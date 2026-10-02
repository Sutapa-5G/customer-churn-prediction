"""
explain.py

SHAP explainability for the final selected Customer Churn Prediction model.

Run from the project root with:

    python -m src.explain

IMPORTANT METHODOLOGICAL NOTES
------------------------------
1. This module is strictly POST-HOC. It loads the already-fitted pipeline from
   `models/churn_model.pkl` and never refits preprocessing, never retrains the
   model, and never writes to the model artifact.

2. The preprocessing transformer is used in `transform`-only mode. It was fit
   on the training split during Step 4; calling `.transform()` on test rows
   applies those already-learned parameters and introduces no leakage.

3. SMOTE is part of the saved pipeline but is a resampling step. imblearn
   pipelines skip resamplers outside of `fit`, so SMOTE plays no role in
   producing the explanations here. We deliberately reach into the
   `preprocessor` and `classifier` steps directly so SHAP operates on exactly
   the matrix the classifier sees.

4. `y_test` is NEVER used. Nothing in this module is selected, tuned, or
   optimised using test labels. The test features are used only as a sample of
   realistic inputs to explain an already-finalised model.

5. SHAP explains THE MODEL, not the world. A high SHAP importance means the
   feature strongly influences this model's predictions. It is not evidence
   that the feature causes churn.

6. For an XGBoost binary classifier, TreeExplainer returns values in the
   model's RAW MARGIN (log-odds) space, not probability. This is verified at
   runtime by `verify_shap_additivity()`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from src import data_preprocessing as dp
    from src import evaluate as ev
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data_preprocessing as dp
    import evaluate as ev


RANDOM_STATE = 42
N_EXPLAIN_SAMPLE = 500  # rows sampled for global explanation (see justification below)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Loading & inspection
# ---------------------------------------------------------------------------
def load_model(model_path: str | Path = MODEL_PATH):
    """Load the saved, already-fitted pipeline. Read-only; never refits."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def inspect_pipeline(pipeline) -> dict:
    """Return a structured description of the saved pipeline for verification."""
    steps = {name: type(step).__name__ for name, step in pipeline.steps}
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    info = {
        "pipeline_class": type(pipeline).__name__,
        "steps": steps,
        "classifier_type": type(classifier).__name__,
        "classifier_module": type(classifier).__module__,
        "is_tree_based": hasattr(classifier, "get_booster")
        or hasattr(classifier, "estimators_"),
        "smote_included": "smote" in steps,
        "n_original_features": int(preprocessor.n_features_in_),
        "n_transformed_features": len(preprocessor.get_feature_names_out()),
    }
    return info


def transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the FITTED preprocessor in transform-only mode.

    This does not refit anything. SMOTE is bypassed entirely by accessing the
    preprocessor step directly, so the returned matrix is exactly what the
    classifier consumes at prediction time.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    # OneHotEncoder may produce a sparse matrix; SHAP needs dense.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


def get_feature_names(pipeline) -> list[str]:
    """Recover transformed feature names programmatically from the fitted pipeline.

    Delegates to the helper already written in `evaluate.py` rather than
    duplicating the logic.
    """
    return ev.get_transformed_feature_names(pipeline.named_steps["preprocessor"])


def get_feature_mapping(
    pipeline, numerical_features: list[str], categorical_features: list[str]
) -> pd.DataFrame:
    """Map each transformed feature back to its original business feature."""
    return ev.map_transformed_to_original(
        get_feature_names(pipeline), numerical_features, categorical_features
    )


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------
def create_shap_explainer(pipeline):
    """Create the SHAP explainer appropriate for the ACTUAL saved classifier.

    TreeExplainer is used only when the classifier is genuinely tree-based
    (XGBoost, RandomForest, etc.). If a future run selects a non-tree model
    (e.g. Logistic Regression), this falls back to the model-agnostic
    `shap.Explainer`, rather than forcing an incompatible explainer.
    """
    classifier = pipeline.named_steps["classifier"]
    is_tree = hasattr(classifier, "get_booster") or hasattr(classifier, "estimators_")

    if is_tree:
        return shap.TreeExplainer(classifier), "TreeExplainer"
    return shap.Explainer(classifier), type(shap.Explainer(classifier)).__name__


def calculate_shap_values(explainer, X_transformed: np.ndarray, feature_names: list[str]):
    """Compute SHAP values and attach feature names.

    Returns a `shap.Explanation` object. For binary XGBoost classifiers the
    `.values` are in RAW MARGIN (log-odds) units — see
    `verify_shap_additivity()` which checks this empirically rather than
    assuming it.
    """
    explanation = explainer(X_transformed)
    explanation.feature_names = list(feature_names)
    return explanation


def verify_shap_additivity(
    explanation, pipeline, X_transformed: np.ndarray, row: int = 0, tol: float = 1e-4
) -> dict:
    """Empirically determine the output space of the SHAP values.

    SHAP's additivity property states:
        sum(shap_values) + base_value == model output for that row

    By comparing that sum against both the raw margin and the predicted
    probability, we establish which space the values live in instead of
    guessing. Mislabelling log-odds as "probability change" is a common and
    misleading error, so this check is run explicitly.
    """
    classifier = pipeline.named_steps["classifier"]

    shap_sum = float(explanation.values[row].sum() + explanation.base_values[row])
    margin = float(classifier.predict(X_transformed[row : row + 1], output_margin=True)[0])
    proba = float(classifier.predict_proba(X_transformed[row : row + 1])[0, 1])
    sigmoid_of_sum = float(1.0 / (1.0 + np.exp(-shap_sum)))

    matches_margin = bool(np.isclose(shap_sum, margin, atol=tol))
    matches_proba = bool(np.isclose(shap_sum, proba, atol=tol))

    return {
        "shap_sum_plus_base": shap_sum,
        "raw_margin": margin,
        "predicted_probability": proba,
        "sigmoid_of_shap_sum": sigmoid_of_sum,
        "matches_raw_margin": matches_margin,
        "matches_probability": matches_proba,
        "sigmoid_matches_probability": bool(np.isclose(sigmoid_of_sum, proba, atol=tol)),
        "output_space": "raw margin (log-odds)" if matches_margin
                        else ("probability" if matches_proba else "undetermined"),
    }


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
def compute_global_importance(
    explanation, feature_mapping: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Build the global importance table from mean absolute SHAP values.

    `mean_abs_shap` measures the average magnitude of a feature's influence on
    the model's output — i.e. how much the model relies on it. `mean_shap`
    records the average signed direction. Neither is a causal quantity.
    """
    values = explanation.values
    names = list(explanation.feature_names)

    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    )

    if feature_mapping is not None:
        importance = importance.merge(
            feature_mapping[["transformed_feature", "original_feature", "category"]],
            left_on="feature",
            right_on="transformed_feature",
            how="left",
        ).drop(columns=["transformed_feature"])

    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_global_feature_importance(
    importance_df: pd.DataFrame, top_n: int = 20, save_path: Optional[str | Path] = None
):
    """Horizontal bar chart of the top-N features by mean absolute SHAP value."""
    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|  (average influence on model output, log-odds)")
    ax.set_ylabel("Transformed feature")
    ax.set_title(
        f"Global SHAP Feature Importance — Top {top_n}\n"
        "Features that most strongly influence the model's predictions"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_beeswarm(
    explanation, max_display: int = 20, save_path: Optional[str | Path] = None
):
    """SHAP beeswarm plot: importance, direction of influence, and distribution.

    Uses the modern `shap.plots.beeswarm` interface (SHAP >= 0.40) rather than
    the deprecated `shap.summary_plot`.
    """
    fig = plt.figure(figsize=(10, max(6, max_display * 0.38)))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(
        "SHAP Beeswarm — influence and direction per feature\n"
        "(positive = pushes model output toward the churn class)",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Individual explanation
# ---------------------------------------------------------------------------
def explain_individual(
    explanation,
    pipeline,
    X_original: pd.DataFrame,
    X_transformed: np.ndarray,
    row: int,
    top_k: int = 8,
) -> dict:
    """Explain a single prediction, without exposing identifying information.

    `customerID` was dropped before modelling and is never referenced here.
    The row is labelled generically as "Example Customer".
    """
    classifier = pipeline.named_steps["classifier"]

    proba = float(classifier.predict_proba(X_transformed[row : row + 1])[0, 1])
    predicted_class = int(proba >= 0.5)

    contributions = pd.DataFrame(
        {
            "feature": list(explanation.feature_names),
            "shap_value": explanation.values[row],
            "feature_value": X_transformed[row],
        }
    ).sort_values("shap_value", ascending=False)

    positive = contributions[contributions["shap_value"] > 0].head(top_k)
    negative = contributions[contributions["shap_value"] < 0].tail(top_k).iloc[::-1]

    return {
        "label": "Example Customer",
        "row_index_in_sample": row,
        "characteristics": X_original.iloc[row].to_dict(),
        "predicted_probability": proba,
        "predicted_class": predicted_class,
        "base_value_log_odds": float(explanation.base_values[row]),
        "sum_shap_log_odds": float(explanation.values[row].sum()),
        "top_positive_contributors": positive.reset_index(drop=True),
        "top_negative_contributors": negative.reset_index(drop=True),
        "all_contributions": contributions.reset_index(drop=True),
    }


def plot_individual_explanation(
    explanation, row: int, max_display: int = 14, save_path: Optional[str | Path] = None
):
    """Waterfall plot showing how one prediction was built up from the base value."""
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation[row], max_display=max_display, show=False)
    plt.title(
        "SHAP Explanation — Example Customer\n"
        "How each feature moved this prediction (log-odds) from the base value",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_shap_outputs(importance_df: pd.DataFrame, reports_dir: str | Path = REPORTS_DIR) -> Path:
    """Persist the global SHAP importance table to CSV."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "shap_feature_importance.csv"
    importance_df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    matplotlib.use("Agg")  # headless backend for CLI use

    print("=" * 78)
    print("CUSTOMER CHURN PREDICTION — SHAP EXPLAINABILITY")
    print("=" * 78)
    print(f"SHAP version: {shap.__version__}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and inspect the saved model -------------------------------------
    print("\n[1/8] Loading and inspecting the saved pipeline...")
    pipeline = load_model()
    info = inspect_pipeline(pipeline)
    for key, value in info.items():
        print(f"      {key}: {value}")

    # 2. Rebuild the exact same split (deterministic, random_state=42) --------
    print("\n[2/8] Reproducing the data split (random_state=42)...")
    raw_df = dp.load_data(RAW_DATA_PATH)
    df_clean = dp.clean_data(raw_df)
    X, y = dp.split_features_target(df_clean)
    X_train, X_test, y_train, y_test = dp.create_train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    numerical_features, categorical_features = dp.get_feature_types(X_train)
    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}")
    print("      NOTE: y_test is not used anywhere in this module.")

    # 3. Feature names --------------------------------------------------------
    print("\n[3/8] Recovering transformed feature names...")
    feature_names = get_feature_names(pipeline)
    feature_mapping = get_feature_mapping(pipeline, numerical_features, categorical_features)
    print(f"      Transformed features: {len(feature_names)}")

    # 4. Select explanation dataset ------------------------------------------
    print(f"\n[4/8] Sampling {N_EXPLAIN_SAMPLE} rows from X_test (random_state={RANDOM_STATE})...")
    sample_size = min(N_EXPLAIN_SAMPLE, len(X_test))
    X_explain = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
    X_explain_transformed = transform_features(pipeline, X_explain)
    print(f"      Explanation matrix: {X_explain_transformed.shape}")

    assert X_explain_transformed.shape[1] == len(feature_names), (
        "Transformed column count does not match feature name count."
    )
    print("      Dimension check passed.")

    # 5. Explainer + SHAP values ---------------------------------------------
    print("\n[5/8] Creating explainer and computing SHAP values...")
    explainer, explainer_name = create_shap_explainer(pipeline)
    print(f"      Explainer: {explainer_name}")
    explanation = calculate_shap_values(explainer, X_explain_transformed, feature_names)
    print(f"      SHAP values shape: {explanation.values.shape}")

    additivity = verify_shap_additivity(explanation, pipeline, X_explain_transformed, row=0)
    print(f"      Output space: {additivity['output_space']}")
    print(f"        sum(shap)+base = {additivity['shap_sum_plus_base']:.6f}")
    print(f"        raw margin     = {additivity['raw_margin']:.6f}")
    print(f"        probability    = {additivity['predicted_probability']:.6f}")
    print(f"        sigmoid(sum)   = {additivity['sigmoid_of_shap_sum']:.6f}")

    # 6. Global importance ----------------------------------------------------
    print("\n[6/8] Computing global importance...")
    importance_df = compute_global_importance(explanation, feature_mapping)
    csv_path = save_shap_outputs(importance_df)
    print(f"      Saved: {csv_path}")
    print("\n      Top 10 features by mean |SHAP| (model influence, not causation):")
    for i, row in importance_df.head(10).iterrows():
        print(f"        {i+1:>2}. {row['feature']:<42} {row['mean_abs_shap']:.5f}")

    plot_global_feature_importance(
        importance_df, top_n=20, save_path=FIGURES_DIR / "shap_global_importance.png"
    )
    plt.close("all")
    plot_shap_beeswarm(explanation, max_display=20, save_path=FIGURES_DIR / "shap_beeswarm.png")
    plt.close("all")
    print(f"      Saved: {FIGURES_DIR / 'shap_global_importance.png'}")
    print(f"      Saved: {FIGURES_DIR / 'shap_beeswarm.png'}")

    # Aggregate back to original business features
    agg = (
        importance_df.groupby("original_feature", dropna=False)["mean_abs_shap"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    agg.to_csv(REPORTS_DIR / "shap_importance_by_original_feature.csv", index=False)
    print(f"      Saved: {REPORTS_DIR / 'shap_importance_by_original_feature.csv'}")

    # 7. Individual explanation ----------------------------------------------
    print("\n[7/8] Explaining one example customer (deterministic: row 0 of sample)...")
    individual = explain_individual(
        explanation, pipeline, X_explain, X_explain_transformed, row=0
    )
    print(f"      Predicted churn probability: {individual['predicted_probability']:.4f}")
    print(f"      Predicted class: {individual['predicted_class']} "
          f"({'Churn' if individual['predicted_class'] else 'No churn'})")
    print("      Top positive contributors (toward churn):")
    for _, r in individual["top_positive_contributors"].head(5).iterrows():
        print(f"        {r['feature']:<42} {r['shap_value']:+.5f}")
    print("      Top negative contributors (away from churn):")
    for _, r in individual["top_negative_contributors"].head(5).iterrows():
        print(f"        {r['feature']:<42} {r['shap_value']:+.5f}")

    plot_individual_explanation(
        explanation, row=0, save_path=FIGURES_DIR / "shap_individual_example.png"
    )
    plt.close("all")
    print(f"      Saved: {FIGURES_DIR / 'shap_individual_example.png'}")

    # 8. Verify predictions still match the saved model ------------------------
    print("\n[8/8] Verifying predictions reproduce the saved model...")
    pipeline_proba = pipeline.predict_proba(X_explain)[:, 1]
    direct_proba = pipeline.named_steps["classifier"].predict_proba(X_explain_transformed)[:, 1]
    max_diff = float(np.max(np.abs(pipeline_proba - direct_proba)))
    print(f"      Max |pipeline_proba - direct_proba| = {max_diff:.10f}")
    assert max_diff < 1e-6, "Explanation path diverges from the saved pipeline!"
    print("      Predictions reproduce exactly.")

    print("\n" + "=" * 78)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("Reminder: SHAP explains MODEL BEHAVIOUR, not causation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
