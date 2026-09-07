# 🌾 Enhanced Crop Prediction System

An explainable machine-learning crop recommendation system based on soil and environmental conditions.

## Project overview

The system predicts suitable crops from:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
- Engineered Season feature

It produces a **Top-3 ranked crop recommendation** with model probability values.

## ML pipeline

1. Data loading
2. Season feature engineering
3. Missing-value preprocessing
4. Random Forest training
5. Optuna hyperparameter optimization framework
6. XGBoost / LightGBM comparison framework
7. 5-Fold Stratified Cross-Validation
8. Top-3 recommendation using `predict_proba()`
9. SHAP-based explainability

The project report describes five enhancements over a baseline approach: Optuna tuning, Season engineering, Top-3 output, multi-model comparison, and 5-Fold CV/SHAP explainability.

## Dataset note

`data/crop_data.csv` in this repository is a **synthetic demonstration dataset** created to match the feature ranges and 22-class structure described in the project report. It is not presented as the original ICAR/Kaggle dataset.

The 22 classes are:

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Beans,
Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon,
Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.

## Run locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

This creates:

```text
models/rf_optuna.joblib
results/metrics.json
```

### 4. Start the web app

```bash
streamlit run app.py
```

## Important note about reported accuracy

The project report reports a 97.05% test accuracy for its Optuna-tuned Random Forest experiment. That number is **report-derived and is not claimed here as a reproduced benchmark**. Run `src/train.py` on the included demonstration dataset to obtain the reproducible metrics for this repository.

## Future improvements

- Tune RF, XGBoost and LightGBM with the same Optuna objective.
- Add real verified agricultural datasets.
- Add SHAP visualizations to the Streamlit interface.
- Expose predictions through FastAPI.
- Deploy with Docker.
- Add yield prediction as a regression task.

## Disclaimer

This is an academic machine-learning project. Model probabilities are not guarantees of crop yield or agricultural success. Real-world recommendations should consider local agronomic conditions and expert advice.
