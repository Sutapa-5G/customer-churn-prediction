"""
app.py — Customer Churn Prediction (Streamlit)

Run from the project root:

    streamlit run app/app.py

This application performs INFERENCE ONLY. It loads the pipeline that was fitted
and saved in Step 4 (`models/churn_model.pkl`) and never retrains, refits, or
writes to it. All preprocessing is delegated to the saved pipeline, so the app
contains no duplicated preprocessing logic — this guarantees the exact same
transformations used at training time are applied at inference time.

Privacy: the app collects no names, emails, phone numbers, or customer IDs.
`customerID` was dropped before modelling and is never requested. Nothing is
sent to any external service; all computation is local.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

# Documented decision threshold from Step 4 (maximised F1 on out-of-fold
# training predictions). Kept fixed for the primary prediction.
DEFAULT_THRESHOLD = 0.50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn_app")


# ---------------------------------------------------------------------------
# Cached loaders (model loaded once, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    """Load the saved pipeline once per session. Read-only."""
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python -m src.train` from the project root first."
        )
    pipeline = joblib.load(MODEL_PATH)

    required_steps = {"preprocessor", "classifier"}
    actual_steps = set(dict(pipeline.steps))
    if not required_steps.issubset(actual_steps):
        raise ValueError(
            f"Loaded model is missing expected steps. Found: {sorted(actual_steps)}"
        )
    return pipeline


@st.cache_resource(show_spinner=False)
def load_metadata() -> dict:
    """Load training metadata (optional; the app degrades gracefully without it)."""
    import json

    if not METADATA_PATH.exists():
        return {}
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except Exception:
        logger.warning("Could not parse model_metadata.json", exc_info=True)
        return {}


@st.cache_resource(show_spinner=False)
def load_shap_explainer(_pipeline):
    """Build the SHAP explainer once, reusing the Step 5 implementation.

    The leading underscore tells Streamlit not to hash the pipeline argument
    (it is already a cached resource).
    """
    import explain as ex

    explainer, name = ex.create_shap_explainer(_pipeline)
    return explainer, name


# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}


def humanize_feature(transformed_name: str, mapping_lookup: dict) -> str:
    """Turn a transformed feature name into a readable label.

    'cat__Contract_Month-to-month' -> 'Contract: Month-to-month'
    'num__tenure'                  -> 'Tenure (months)'
    """
    info = mapping_lookup.get(transformed_name)
    if info is None:
        return transformed_name

    original, category = info
    pretty = FEATURE_LABELS.get(original, original)
    if category is None or (isinstance(category, float) and pd.isna(category)):
        return pretty
    return f"{pretty}: {category}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_inputs(data: dict, expected_categories: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid).

    Invalid input is reported, never silently corrected.
    """
    errors: list[str] = []

    if data["tenure"] < 0:
        errors.append("Tenure must be 0 or greater.")
    if data["MonthlyCharges"] < 0:
        errors.append("Monthly charges must be 0 or greater.")
    if data["TotalCharges"] < 0:
        errors.append("Total charges must be 0 or greater.")

    for column, allowed in expected_categories.items():
        value = data.get(column)
        if value is None:
            errors.append(f"{FEATURE_LABELS.get(column, column)} is required.")
        elif value not in allowed:
            errors.append(
                f"{FEATURE_LABELS.get(column, column)} has an unexpected value "
                f"'{value}'. Allowed: {', '.join(map(str, allowed))}."
            )

    # Consistency checks (these mirror how the dataset itself is structured)
    if data["tenure"] == 0 and data["TotalCharges"] > 0:
        errors.append(
            "A customer with 0 months of tenure has not been billed yet, "
            "so total charges should be 0."
        )
    if data["PhoneService"] == "No" and data["MultipleLines"] != "No phone service":
        errors.append(
            "Multiple lines must be 'No phone service' when phone service is 'No'."
        )
    if data["InternetService"] == "No":
        internet_addons = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        bad = [c for c in internet_addons if data[c] != "No internet service"]
        if bad:
            labels = ", ".join(FEATURE_LABELS.get(c, c) for c in bad)
            errors.append(
                f"When internet service is 'No', these must be 'No internet service': {labels}."
            )

    return errors


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📊",
        layout="wide",
    )

    st.title("Customer Churn Prediction")
    st.caption("Predict the likelihood that a customer will churn.")

    # --- Load model -------------------------------------------------------
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        logger.error("Failed to load model", exc_info=True)
        st.error(
            "The model file could not be loaded. It may be missing or corrupted. "
            "Try regenerating it by running `python -m src.train` from the project root."
        )
        with st.expander("Technical details (for developers)"):
            st.code(traceback.format_exc())
        st.stop()

    metadata = load_metadata()

    # Derive the exact expected columns and category values from the FITTED
    # pipeline itself, so the form can never drift from what the model expects.
    preprocessor = pipeline.named_steps["preprocessor"]
    numerical_features = list(preprocessor.transformers_[0][2])
    categorical_features = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    expected_categories = {
        col: list(cats) for col, cats in zip(categorical_features, encoder.categories_)
    }
    expected_columns = list(metadata.get("original_feature_names")
                            or (categorical_features + numerical_features))

    # --- Sidebar ----------------------------------------------------------
    render_sidebar(metadata, pipeline)

    # --- Input form -------------------------------------------------------
    st.header("Customer Information")
    st.caption(
        "Enter the customer's attributes below. No personally identifying "
        "information is collected or required."
    )

    with st.form("customer_form"):
        st.subheader("Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", expected_categories["gender"])
        with c2:
            senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        with c3:
            partner = st.selectbox("Partner", expected_categories["Partner"])
        with c4:
            dependents = st.selectbox("Dependents", expected_categories["Dependents"])

        st.divider()
        st.subheader("Account Information")
        a1, a2, a3 = st.columns(3)
        with a1:
            tenure = st.slider("Tenure (months)", 0, 72, 12,
                               help="How many months the customer has been with the company.")
            contract = st.selectbox("Contract", expected_categories["Contract"])
        with a2:
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0, max_value=1000.0,
                value=70.0, step=0.05, format="%.2f",
            )
            paperless = st.selectbox("Paperless billing", expected_categories["PaperlessBilling"])
        with a3:
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, max_value=100000.0,
                value=840.0, step=0.05, format="%.2f",
                help="Total amount billed to date. Should be 0 for a brand-new customer.",
            )
            payment_method = st.selectbox("Payment method", expected_categories["PaymentMethod"])

        st.divider()
        st.subheader("Services")
        st.caption(
            "Add-on options are only available when the underlying service is active; "
            "they are set automatically otherwise."
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone service", expected_categories["PhoneService"])
            if phone_service == "No":
                multiple_lines = "No phone service"
                st.caption("Multiple lines: No phone service (set automatically)")
            else:
                multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

            internet_service = st.selectbox(
                "Internet service", expected_categories["InternetService"]
            )

        has_internet = internet_service != "No"
        addon_options = ["No", "Yes"]

        with s2:
            if has_internet:
                online_security = st.selectbox("Online security", addon_options)
                online_backup = st.selectbox("Online backup", addon_options)
                device_protection = st.selectbox("Device protection", addon_options)
            else:
                online_security = online_backup = device_protection = "No internet service"
                st.caption("Online security: No internet service")
                st.caption("Online backup: No internet service")
                st.caption("Device protection: No internet service")

        with s3:
            if has_internet:
                tech_support = st.selectbox("Tech support", addon_options)
                streaming_tv = st.selectbox("Streaming TV", addon_options)
                streaming_movies = st.selectbox("Streaming movies", addon_options)
            else:
                tech_support = streaming_tv = streaming_movies = "No internet service"
                st.caption("Tech support: No internet service")
                st.caption("Streaming TV: No internet service")
                st.caption("Streaming movies: No internet service")

        st.divider()
        submitted = st.form_submit_button("Predict Churn", type="primary", width="stretch")

    if not submitted:
        st.info("Fill in the customer details above and select **Predict Churn**.")
        render_disclaimer()
        return

    # --- Assemble the input row ------------------------------------------
    input_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    # --- Validate ---------------------------------------------------------
    errors = validate_inputs(input_data, expected_categories)
    if errors:
        st.error("Please correct the following before predicting:")
        for err in errors:
            st.write(f"- {err}")
        return

    # Single-row DataFrame with exactly the columns the pipeline expects
    input_df = pd.DataFrame([input_data])[expected_columns]

    # --- Predict ----------------------------------------------------------
    try:
        # The saved pipeline performs all preprocessing internally.
        probability = float(pipeline.predict_proba(input_df)[0, 1])
        predicted_class = int(probability >= DEFAULT_THRESHOLD)
    except Exception:
        logger.error("Prediction failed", exc_info=True)
        st.error(
            "Something went wrong while generating the prediction. "
            "Please check the input values and try again."
        )
        with st.expander("Technical details (for developers)"):
            st.code(traceback.format_exc())
        return

    render_prediction(probability, predicted_class)
    render_shap_section(pipeline, input_df, probability)
    render_disclaimer()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_prediction(probability: float, predicted_class: int) -> None:
    st.header("Prediction Result")

    label = "Churn" if predicted_class == 1 else "No Churn"
    pct = probability * 100

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted churn probability", f"{pct:.1f}%")
        if predicted_class == 1:
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
        st.caption(f"Based on the project's documented threshold of {DEFAULT_THRESHOLD:.2f}.")

    with right:
        st.write("**Churn probability**")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"{pct:.1f}% likelihood of churn · {100 - pct:.1f}% likelihood of retention"
        )

        if probability >= 0.70:
            band = "High estimated likelihood of churn"
        elif probability >= DEFAULT_THRESHOLD:
            band = "Moderate estimated likelihood of churn"
        elif probability >= 0.30:
            band = "Lower estimated likelihood of churn"
        else:
            band = "Low estimated likelihood of churn"
        st.info(
            f"{band}. This is a statistical estimate from historical data, "
            "not a certainty about this individual customer."
        )


def render_shap_section(pipeline, input_df: pd.DataFrame, probability: float) -> None:
    st.header("Why did the model make this prediction?")

    with st.expander("Show explanation (SHAP)", expanded=False):
        try:
            import explain as ex
            import evaluate as ev

            explainer, explainer_name = load_shap_explainer(pipeline)

            transformed = ex.transform_features(pipeline, input_df)
            feature_names = ex.get_feature_names(pipeline)
            explanation = ex.calculate_shap_values(explainer, transformed, feature_names)

            preprocessor = pipeline.named_steps["preprocessor"]
            numerical_features = list(preprocessor.transformers_[0][2])
            categorical_features = list(preprocessor.transformers_[1][2])
            mapping = ev.map_transformed_to_original(
                feature_names, numerical_features, categorical_features
            )
            mapping_lookup = {
                r["transformed_feature"]: (r["original_feature"], r["category"])
                for _, r in mapping.iterrows()
            }

            contributions = pd.DataFrame({
                "feature": feature_names,
                "shap_value": explanation.values[0],
            })
            contributions["label"] = contributions["feature"].map(
                lambda n: humanize_feature(n, mapping_lookup)
            )

            pushed_toward = (
                contributions[contributions["shap_value"] > 0]
                .sort_values("shap_value", ascending=False).head(5)
            )
            pushed_away = (
                contributions[contributions["shap_value"] < 0]
                .sort_values("shap_value").head(5)
            )

            st.write(
                "These are the factors that most influenced **this particular prediction**."
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Pushed the prediction toward churn**")
                if pushed_toward.empty:
                    st.caption("No features pushed this prediction toward churn.")
                else:
                    for _, r in pushed_toward.iterrows():
                        st.write(f"- {r['label']}")
            with c2:
                st.markdown("**Pushed the prediction away from churn**")
                if pushed_away.empty:
                    st.caption("No features pushed this prediction away from churn.")
                else:
                    for _, r in pushed_away.iterrows():
                        st.write(f"- {r['label']}")

            st.divider()
            st.caption(
                "SHAP explains the model's prediction; it does not establish that a "
                "feature causes churn."
            )

            with st.expander("Technical detail: SHAP contribution values"):
                st.caption(
                    f"Explainer: {explainer_name}. For this XGBoost model, SHAP values are "
                    "in **log-odds (raw margin)** units, not probability. They sum with the "
                    "model's base value to produce the raw output for this customer. A value "
                    "of +0.40 therefore means '+0.40 log-odds toward churn', which is not the "
                    "same as a 40 percentage-point change in probability."
                )
                display = contributions.reindex(
                    contributions["shap_value"].abs().sort_values(ascending=False).index
                ).head(15)
                st.dataframe(
                    display[["label", "shap_value"]].rename(
                        columns={"label": "Feature", "shap_value": "SHAP value (log-odds)"}
                    ),
                    hide_index=True,
                    width="stretch",
                )

        except Exception:
            logger.error("SHAP explanation failed", exc_info=True)
            st.warning(
                "The explanation could not be generated for this input. "
                "The prediction above is still valid."
            )
            with st.expander("Technical details (for developers)"):
                st.code(traceback.format_exc())


def render_sidebar(metadata: dict, pipeline) -> None:
    with st.sidebar:
        st.header("About this model")

        classifier_name = type(pipeline.named_steps["classifier"]).__name__
        st.markdown(
            f"""
- **Model:** XGBoost (`{classifier_name}`)
- **Dataset:** IBM Telco Customer Churn (public benchmark)
- **Validation:** 5-fold stratified cross-validation
- **SMOTE:** applied during training/CV only — never to validation or test data
- **Primary selection metric:** ROC-AUC
- **CV ROC-AUC:** 0.8470
- **Decision threshold:** {DEFAULT_THRESHOLD:.2f}
"""
        )

        st.warning(
            "This is a **demonstration / portfolio project** built on a public "
            "benchmark dataset. It is not a production model, and it is not trained "
            "on any real company's current customers."
        )

        with st.expander("More technical details"):
            if metadata:
                st.markdown(
                    f"""
- Random state: `{metadata.get('random_state', 'n/a')}`
- CV folds: `{metadata.get('cv_n_splits', 'n/a')}`
- Original features: `{len(metadata.get('original_feature_names', []))}`
- Transformed features: `{metadata.get('n_transformed_features', 'n/a')}`
- Threshold selection: {metadata.get('threshold_selection_method', 'n/a')}
"""
                )
                test_metrics = metadata.get("selected_model_test_metrics", {})
                if test_metrics:
                    st.markdown("**Held-out test set performance**")
                    st.markdown(
                        f"""
- Accuracy: `{test_metrics.get('accuracy', float('nan')):.4f}`
- Precision: `{test_metrics.get('precision', float('nan')):.4f}`
- Recall: `{test_metrics.get('recall', float('nan')):.4f}`
- F1: `{test_metrics.get('f1', float('nan')):.4f}`
- ROC-AUC: `{test_metrics.get('roc_auc', float('nan')):.4f}`
"""
                    )
            else:
                st.caption("Training metadata file not found.")

        st.divider()
        st.caption("No personal data is collected. All computation runs locally.")


def render_disclaimer() -> None:
    st.divider()
    st.caption(
        "**Disclaimer.** This model provides a statistical prediction based on historical "
        "data. It does not establish causation and should not be treated as a certainty. "
        "Performance measured on this public dataset may not generalize to another "
        "company's customers, whose behaviour, pricing, and market conditions may differ."
    )


if __name__ == "__main__":
    main()
