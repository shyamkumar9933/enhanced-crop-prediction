from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from feature_engineering import prepare_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "crop_data.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = prepare_features(df)
    y = df["crop"]

    return X, y


def build_model():
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=155,
                max_depth=10,
                min_samples_split=9,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return model


def main():
    print("Loading dataset...")

    X, y = load_data()

    print(f"Samples: {len(X)}")
    print(f"Classes: {y.nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\nTraining Random Forest...")

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\nRunning 5-fold cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"CV Mean: {cv_scores.mean():.4f}")
    print(f"CV Std: {cv_scores.std():.4f}")

    model_path = MODEL_DIR / "rf_optuna.joblib"

    joblib.dump(
        model,
        model_path
    )

    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "n_samples": int(len(X)),
        "n_classes": int(y.nunique()),
        "model": "Random Forest"
    }

    metrics_path = RESULTS_DIR / "metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nModel saved:")
    print(model_path)

    print("\nMetrics saved:")
    print(metrics_path)

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":
    main()
