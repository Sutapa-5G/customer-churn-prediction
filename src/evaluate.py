"""
evaluate.py

Reusable model evaluation utilities for the Customer Churn Prediction project.

All functions here compute metrics from actual model predictions — nothing is
hardcoded or assumed. Plotting functions derive their annotations (e.g. ROC-AUC
values shown in the legend) from the same computed arrays they plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# Metric names used consistently for cross-validation scoring
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

CLASS_NAMES = ["No Churn (0)", "Churn (1)"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calculate_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: Optional[Sequence] = None,
    model_name: str = "model",
) -> dict:
    """Compute the standard classification metric suite from real predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).
    y_proba : array-like, optional
        Predicted probabilities for the positive class. Required for ROC-AUC;
        if None, ROC-AUC is returned as NaN rather than being guessed.
    model_name : str
        Label used in the returned dict.

    Returns
    -------
    dict
        Keys: model, accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    model_name: str = "model",
) -> dict:
    """Run stratified k-fold cross-validation on the TRAINING data only.

    The `pipeline` passed in must be a full imblearn pipeline
    (preprocessing -> SMOTE -> classifier). Because the whole pipeline is
    handed to `cross_validate`, scikit-learn refits every step inside each
    fold: the preprocessor is fit on the training fold only, SMOTE resamples
    the training fold only, and the validation fold is merely transformed and
    predicted on. No information from a validation fold can reach the
    preprocessing statistics or the synthetic SMOTE samples — this is what
    prevents leakage during model selection.

    Returns
    -------
    dict
        Mean and standard deviation for each metric, e.g.
        {"model": ..., "accuracy_mean": ..., "accuracy_std": ..., ...}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=1,
        return_train_score=False,
    )

    summary = {"model": model_name}
    for metric in CV_SCORING:
        scores = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(scores))
        summary[f"{metric}_std"] = float(np.std(scores))

    return summary


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------
def compare_models(results: list[dict], sort_by: Optional[str] = None) -> pd.DataFrame:
    """Turn a list of metric dicts into a tidy comparison DataFrame.

    Parameters
    ----------
    results : list[dict]
        Output of `calculate_classification_metrics` or `cross_validate_model`.
    sort_by : str, optional
        Column to sort by (descending). If None, original order is preserved
        — deliberately the default, so no implicit ranking is imposed.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results)
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    model_name: str,
    save_path: Optional[str | Path] = None,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
):
    """Plot (and optionally save) a labelled confusion matrix for one model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig, cm


def plot_roc_curves(
    model_probas: dict[str, tuple[Sequence, Sequence]],
    save_path: Optional[str | Path] = None,
):
    """Plot ROC curves for multiple models on a single axis.

    Parameters
    ----------
    model_probas : dict
        Mapping of model_name -> (y_true, y_proba). AUC values shown in the
        legend are computed here from these arrays, never passed in or
        hardcoded.
    save_path : str | Path, optional
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (y_true, y_proba) in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_value = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_value:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig


def plot_cv_comparison(
    cv_df: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
    save_path: Optional[str | Path] = None,
):
    """Grouped bar chart of mean CV scores per model, with std-dev error bars."""
    plot_rows = []
    for _, row in cv_df.iterrows():
        for metric in metrics:
            plot_rows.append(
                {
                    "model": row["model"],
                    "metric": metric,
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                }
            )
    plot_df = pd.DataFrame(plot_rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x="metric", y="mean", hue="model", palette="Set2", ax=ax)

    # Overlay std-dev error bars aligned to each bar
    for bar_container, model_name in zip(ax.containers, plot_df["model"].unique()):
        model_subset = plot_df[plot_df["model"] == model_name].set_index("metric")
        for bar, metric in zip(bar_container, metrics):
            std = model_subset.loc[metric, "std"]
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=std,
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1,
            )

    ax.set_title("Cross-Validation Mean Scores by Model (error bars = std. dev. across folds)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", loc="lower right")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------
def threshold_analysis(
    y_true: Sequence,
    y_proba: Sequence,
    thresholds: Sequence[float] = (0.30, 0.40, 0.50, 0.60, 0.70),
) -> pd.DataFrame:
    """Compute precision/recall/F1 at several classification thresholds.

    Parameters
    ----------
    y_true : array-like
        True labels. For honest threshold selection these should be
        out-of-fold (cross-validated) labels from the TRAINING data, not the
        test set.
    y_proba : array-like
        Predicted positive-class probabilities corresponding to `y_true`.
    thresholds : sequence of float

    Returns
    -------
    pd.DataFrame with columns: threshold, precision, recall, f1, tn, fp, fn, tp
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature name mapping (for the SHAP stage)
# ---------------------------------------------------------------------------
def get_transformed_feature_names(fitted_preprocessor) -> list[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.

    Uses scikit-learn's `get_feature_names_out()`, which correctly expands
    one-hot encoded columns into their individual category columns. This is
    the reliable mapping from original features to transformed features that
    the SHAP stage needs in order to label its plots meaningfully.
    """
    return list(fitted_preprocessor.get_feature_names_out())


def map_transformed_to_original(
    transformed_names: Sequence[str],
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Build an explicit transformed-feature -> original-feature mapping table.

    ColumnTransformer prefixes output names with the transformer name
    ("num__" / "cat__"). For one-hot columns the remainder is
    "<original_feature>_<category>". This function resolves each transformed
    name back to the original column it came from.
    """
    rows = []
    for name in transformed_names:
        if name.startswith("num__"):
            stripped = name[len("num__"):]
            original = stripped if stripped in numerical_features else stripped
            kind = "numerical"
            category = None
        elif name.startswith("cat__"):
            stripped = name[len("cat__"):]
            # Match the longest categorical feature name that prefixes this string
            original = None
            category = None
            for feature in sorted(categorical_features, key=len, reverse=True):
                if stripped.startswith(feature + "_"):
                    original = feature
                    category = stripped[len(feature) + 1:]
                    break
            if original is None:
                original = stripped
            kind = "categorical (one-hot)"
        else:
            original = name
            kind = "unknown"
            category = None

        rows.append(
            {
                "transformed_feature": name,
                "original_feature": original,
                "type": kind,
                "category": category,
            }
        )

    return pd.DataFrame(rows)
