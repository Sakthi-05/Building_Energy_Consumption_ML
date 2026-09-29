Building Energy Consumption Prediction

An AI/ML regression project that predicts building energy consumption using weather conditions, building characteristics, and time-based features from the ASHRAE Great Energy Predictor III dataset.

Overview

The project combines:

- "train.csv" — historical energy consumption
- "weather_train.csv" — weather data
- "building_metadata.csv" — building information

Target: "meter_reading"

The data is cleaned, merged, transformed, and used to train and compare multiple regression models.

Models Compared

Model| Purpose
Linear Regression| Baseline model
XGBoost| Gradient boosting
LightGBM| Gradient boosting

XGBoost and LightGBM are tuned using 3-fold cross-validation and compared with the baseline.

The final model is selected based on the highest test R² score, rather than assuming a particular algorithm will perform best.

Feature Engineering

Key features include:

- Hour
- Day
- Month
- Weekday
- Weekend indicator
- Time period
- Weather variables
- Building characteristics

Additional preprocessing includes missing-value imputation, categorical encoding, outlier handling, and "log1p" transformation of the target.

Evaluation

Models are evaluated using:

- MAE
- MSE
- RMSE
- R² Score

The project also includes visualizations for model comparison, predicted vs. actual values, residuals, correlations, and feature importance.

Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn

Project Structure

Building_Energy_Consumption_ML/
├── model.py
├── requirements.txt
├── README.md
├── train.csv
├── weather_train.csv
└── building_metadata.csv

Key Learning

This project demonstrates practical experience with data preprocessing, feature engineering, regression, cross-validation, hyperparameter tuning, gradient boosting, model evaluation, and model comparison.

Author

Sakthi Mageswari V.
Computer Science Engineering | AI/ML
