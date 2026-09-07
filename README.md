# 🌾 Enhanced Crop Prediction System

[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://enhanced-crop-prediction.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![GitHub Actions](https://github.com/shyamkumar9933/enhanced-crop-prediction/actions/workflows/train.yml/badge.svg)](https://github.com/shyamkumar9933/enhanced-crop-prediction/actions)

An explainable machine-learning based crop recommendation system that predicts the most suitable crops from soil and environmental conditions.

## 🚀 Live Demo

👉 **Try the application:**  
https://enhanced-crop-prediction.streamlit.app

The application accepts soil and environmental conditions and returns the **Top-3 recommended crops** with model probability scores.

---

## 📌 Project Overview

Selecting a suitable crop depends on multiple soil and environmental conditions.

This project develops a machine-learning based recommendation system using:

- Soil nutrient values
- Temperature
- Humidity
- Soil pH
- Rainfall
- An engineered seasonal feature

The system predicts the most suitable crop and provides the **Top-3 recommendations** instead of returning only a single prediction.

---

## ✨ Key Features

- 🌱 Crop recommendation using machine learning
- 🥇 Top-3 crop predictions
- 📊 Probability-based confidence scores
- 🧪 Soil and environmental feature processing
- 🌦️ Engineered season feature
- 🤖 Random Forest classification
- 🔄 5-fold stratified cross-validation
- 🔍 SHAP explainability support
- 💾 Saved trained model using Joblib
- ⚙️ Automated model training using GitHub Actions
- 🌐 Interactive Streamlit web application

---

## 🧾 Input Features

The model uses the following input parameters:

| Feature | Description |
|---|---|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| Temperature | Environmental temperature in °C |
| Humidity | Relative humidity (%) |
| pH | Soil acidity/alkalinity |
| Rainfall | Rainfall in mm |
| Season | Engineered from temperature |

---

## 🧠 Machine Learning Pipeline

```text
User Input
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Season Feature
    ↓
Missing Value Handling
    ↓
Random Forest Classifier
    ↓
Prediction Probabilities
    ↓
Rank Predictions
    ↓
Top-3 Crop Recommendations
