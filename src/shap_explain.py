from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def explain_model(model_path, X):
    """
    Generate SHAP feature importance values when SHAP is available.
    """

    try:
        import shap
    except ImportError:
        return {
            "status": "SHAP is not installed",
            "feature_importance": {}
        }

    model = joblib.load(Path(model_path))

    # Get the Random Forest model from the pipeline
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("random_forest", model)
    else:
        estimator = model

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        importance = abs(shap_values[0]).mean(axis=0)
    else:
        importance = abs(shap_values).mean(axis=0)

    feature_importance = pd.Series(
        importance,
        index=X.columns
    ).sort_values(ascending=False)

    return {
        "status": "success",
        "feature_importance": feature_importance.to_dict()
    }


if __name__ == "__main__":
    print("SHAP explanation module ready.")
