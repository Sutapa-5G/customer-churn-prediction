"""
data_preprocessing.py

Reusable, leakage-safe data preprocessing utilities for the Customer Churn
Prediction project.

Design principles followed throughout this module:
- Cleaning steps that use a fixed domain rule (e.g. "no billing history yet"
  implies TotalCharges = 0) are applied identically to every row and do not
  depend on any statistic computed from the data, so they are safe to apply
  before the train/test split.
- Any transformation that is *learned from data* (imputation statistics,
  scaling parameters, encoding categories) is only ever `.fit()` on the
  training set. The test set must only ever be `.transform()`-ed.
- SMOTE (class-imbalance resampling) is only ever applied to the training
  data, and only inside a pipeline, never to the full dataset or the test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV as-is (no modification).

    Parameters
    ----------
    path : str | Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, domain-driven cleaning to the raw dataframe.

    This function does NOT fit anything on the data (no learned statistics
    such as mean/median from the full dataset), so it is safe to call before
    the train/test split. Specifically it:

    1. Converts `TotalCharges` from text to numeric, turning blank/whitespace
       strings into NaN.
    2. Fills the resulting NaNs in `TotalCharges` with 0. Investigation (see
       the accompanying notebook) confirmed that every blank `TotalCharges`
       value belongs to a customer with `tenure == 0`, i.e. a brand-new
       customer who has not yet been billed. Since MonthlyCharges * tenure
       for these customers is exactly 0, filling with 0 reflects their real
       billing history rather than guessing a statistic — it is a fixed
       domain rule, not a data-dependent imputation, so it introduces no
       leakage risk between train and test.
    3. Drops `customerID` (a unique identifier with no predictive value).
    4. Encodes the target `Churn` as 0 ("No") / 1 ("Yes").

    The input dataframe is not modified in place; a cleaned copy is returned.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe as returned by `load_data`.

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the dataframe.
    """
    df_clean = df.copy()

    # 1-2. TotalCharges: text -> numeric, blanks -> NaN -> 0
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)

    # 3. Drop identifier column
    if ID_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[ID_COLUMN])

    # 4. Encode target
    # Note: newer pandas versions (2.x/3.x with the string-dtype backend) store
    # text columns as a native "str"/StringDtype rather than classic "object"
    # dtype, so we check "is this non-numeric" instead of "dtype == object" to
    # remain correct across pandas versions.
    if TARGET_COLUMN in df_clean.columns and not pd.api.types.is_numeric_dtype(
        df_clean[TARGET_COLUMN]
    ):
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)

    return df_clean


# ---------------------------------------------------------------------------
# Feature / target separation
# ---------------------------------------------------------------------------
def split_features_target(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        A cleaned dataframe (output of `clean_data`).
    target_column : str
        Name of the target column.

    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.Series]
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Create a stratified train/test split.

    Stratifying on `y` preserves the original class proportions in both the
    training and test sets, which matters here because churn is imbalanced.

    This split must happen BEFORE any data-dependent preprocessing (fitting
    imputers, scalers, or encoders) is performed, so that no information
    about the test set's distribution can influence how the training data is
    transformed. That is what "preventing data leakage" means in practice.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Feature type identification
# ---------------------------------------------------------------------------
def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Automatically detect numerical vs categorical feature columns.

    Numerical features are those with a numeric dtype (this naturally
    includes `tenure`, `MonthlyCharges`, `TotalCharges`, and `SeniorCitizen`,
    since the latter is stored as 0/1 integers in this dataset).
    Categorical features are those with an object/string dtype.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe (must not contain the target column).

    Returns
    -------
    (numerical_features, categorical_features) : tuple[list[str], list[str]]
    """
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return numerical_features, categorical_features


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for numerical + categorical preprocessing.

    Numerical features: median imputation (a safety net for any future
    missing values) followed by StandardScaler.
    Categorical features: most-frequent imputation followed by
    OneHotEncoder(handle_unknown="ignore") so unseen categories at inference
    time do not crash the pipeline.

    IMPORTANT: the returned transformer is unfitted. It must be `.fit()`
    only on training data; call `.transform()` (not `.fit_transform()`) on
    the test data.

    Parameters
    ----------
    numerical_features : list[str]
    categorical_features : list[str]

    Returns
    -------
    ColumnTransformer
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Full pipeline (preprocessing [-> SMOTE] [-> classifier])
# ---------------------------------------------------------------------------
def build_pipeline(
    preprocessor: ColumnTransformer,
    classifier: Optional[object] = None,
    use_smote: bool = True,
    random_state: int = 42,
) -> ImbPipeline:
    """Assemble the full leakage-safe pipeline: preprocessing -> SMOTE -> model.

    Using `imblearn.pipeline.Pipeline` (not plain `sklearn.pipeline.Pipeline`)
    is essential: imblearn's pipeline knows that a resampler like SMOTE
    should only run during `.fit()` on training folds, and is automatically
    skipped during `.transform()`/`.predict()` on new data (e.g. the test
    set, or a single customer in the Streamlit app). This guarantees SMOTE
    can never touch test/validation/inference data, even if this pipeline is
    reused inside cross-validation.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted preprocessor from `build_preprocessor`.
    classifier : object, optional
        A scikit-learn-compatible classifier. If None, the pipeline ends
        after preprocessing (+ SMOTE if enabled) — useful for demonstrating
        or unit-testing the preprocessing/resampling architecture without
        training a real model.
    use_smote : bool
        Whether to include a SMOTE resampling step.
    random_state : int
        Random state for SMOTE (and for the classifier, if it accepts one).

    Returns
    -------
    imblearn.pipeline.Pipeline
    """
    steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    if classifier is not None:
        steps.append(("classifier", classifier))

    return ImbPipeline(steps=steps)


# ---------------------------------------------------------------------------
# Data quality / leakage validation checks
# ---------------------------------------------------------------------------
def validate_no_missing_values(df: pd.DataFrame, allowed_columns: Optional[list[str]] = None) -> None:
    """Assert there are no NaN values, except in explicitly allowed columns."""
    allowed_columns = allowed_columns or []
    missing = df.isnull().sum()
    unexpected_missing = missing[(missing > 0) & (~missing.index.isin(allowed_columns))]
    assert unexpected_missing.empty, f"Unexpected missing values found:\n{unexpected_missing}"


def validate_target_values(y: pd.Series, expected_values: set = frozenset({0, 1})) -> None:
    """Assert the target only contains the expected encoded values."""
    actual_values = set(y.unique())
    assert actual_values.issubset(expected_values), (
        f"Target contains unexpected values: {actual_values - expected_values}"
    )


def validate_no_duplicate_ids(raw_df: pd.DataFrame, id_column: str = ID_COLUMN) -> None:
    """Assert the identifier column has no duplicate values in the raw data."""
    duplicate_count = raw_df[id_column].duplicated().sum()
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate {id_column} values."


def validate_no_id_leakage(X: pd.DataFrame, id_column: str = ID_COLUMN) -> None:
    """Assert the identifier column is not present among the model features."""
    assert id_column not in X.columns, f"{id_column} must not be present in the feature set X."


def validate_dtypes(
    X: pd.DataFrame, numerical_features: list[str], categorical_features: list[str]
) -> None:
    """Assert every column in X is accounted for as either numerical or categorical."""
    all_declared = set(numerical_features) | set(categorical_features)
    all_actual = set(X.columns)
    assert all_declared == all_actual, (
        f"Column/type mismatch. Declared but missing from X: {all_declared - all_actual}. "
        f"Present in X but not typed: {all_actual - all_declared}."
    )
    for col in numerical_features:
        assert pd.api.types.is_numeric_dtype(X[col]), f"{col} was expected to be numeric."
    for col in categorical_features:
        assert not pd.api.types.is_numeric_dtype(X[col]), f"{col} was expected to be categorical."


def validate_no_infinite_values(X: pd.DataFrame, numerical_features: list[str]) -> None:
    """Assert no +/- infinity values exist in the numerical features."""
    for col in numerical_features:
        n_inf = np.isinf(X[col]).sum()
        assert n_inf == 0, f"Found {n_inf} infinite value(s) in column '{col}'."


def validate_no_unexpected_negatives(
    X: pd.DataFrame, non_negative_columns: list[str]
) -> None:
    """Assert columns that should logically never be negative (e.g. tenure, charges) aren't."""
    for col in non_negative_columns:
        if col in X.columns:
            n_negative = (X[col] < 0).sum()
            assert n_negative == 0, f"Found {n_negative} negative value(s) in column '{col}'."


def validate_categories_known(
    train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_features: list[str]
) -> dict[str, set]:
    """Report any categories present in the test set but absent from the training set.

    This is informational (returns the differences) rather than a hard
    assertion, because OneHotEncoder(handle_unknown="ignore") is specifically
    configured to handle this situation safely at transform time. It is
    still useful to surface for awareness.
    """
    unseen_report: dict[str, set] = {}
    for col in categorical_features:
        train_categories = set(train_df[col].unique())
        test_categories = set(test_df[col].unique())
        unseen = test_categories - train_categories
        if unseen:
            unseen_report[col] = unseen
    return unseen_report


def run_all_validations(
    raw_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
) -> None:
    """Convenience function running the full validation suite. Raises on failure."""
    validate_no_duplicate_ids(raw_df)
    validate_no_id_leakage(X)
    validate_target_values(y)
    validate_dtypes(X, numerical_features, categorical_features)
    validate_no_missing_values(X)
    validate_no_infinite_values(X, numerical_features)
    validate_no_unexpected_negatives(
        X, non_negative_columns=["tenure", "MonthlyCharges", "TotalCharges"]
    )
