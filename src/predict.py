from pathlib import Path
import joblib

from feature_engineering import prepare_single_input


def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def top_k_recommendations(model, values, k=3):
    X = prepare_single_input(values)

    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    ranked = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    )[:k]

    return [
        {
            "crop": crop,
            "confidence": float(probability * 100)
        }
        for crop, probability in ranked
    ]
