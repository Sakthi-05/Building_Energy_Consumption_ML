# Building Energy Consumption Prediction

A machine learning project for predicting building energy consumption using historical meter readings, weather information, building metadata, and time-based features.

The project compares three regression models:

1. **Linear Regression** — baseline model
2. **XGBoost** — gradient boosting model
3. **LightGBM** — gradient boosting model

The models are evaluated using **MAE, MSE, RMSE, and R² Score**, with the model achieving the highest R² score selected as the best-performing model.

---

## Project Overview

Building energy consumption depends on several factors such as building characteristics, weather conditions, time of day, seasonality, and building usage.

This project combines these different sources of information and applies machine learning techniques to predict the `meter_reading` value.

The workflow includes:

* Loading the datasets
* Sampling the training data for performance
* Merging building and weather information
* Data cleaning
* Timestamp feature engineering
* Categorical feature encoding
* Memory optimization
* Log transformation of the target variable
* Train/test splitting
* Missing-value imputation
* Model training and comparison
* Hyperparameter tuning using cross-validation
* Model evaluation
* Visualization of results
* Feature importance analysis

---

## Models Used

### 1. Linear Regression

Linear Regression is used as the baseline model.

It assumes a linear relationship between the input features and the target variable.

A `StandardScaler` is applied before training because Linear Regression is sensitive to differences in feature scales.

### 2. XGBoost

XGBoost is a gradient boosting algorithm based on decision trees.

The project evaluates different combinations of:

* `max_depth`: 6, 8, 10
* `n_estimators`: 100, 200
* `learning_rate`: 0.05

Three-fold cross-validation is used to select the best configuration.

### 3. LightGBM

LightGBM is another gradient boosting framework designed for efficient training on tabular datasets.

The project evaluates:

* `max_depth`: 8, 12, 16
* `n_estimators`: 100, 200
* `learning_rate`: 0.05

Three-fold cross-validation is used to select the best configuration.

---

## Dataset

The project uses three CSV files:

```text
train.csv
weather_train.csv
building_metadata.csv
```

### `train.csv`

Contains the energy meter readings used as the prediction target.

The target variable is:

```text
meter_reading
```

### `weather_train.csv`

Contains weather-related information associated with the corresponding site and timestamp.

### `building_metadata.csv`

Contains metadata describing the buildings.

The datasets are merged using:

```text
building_id
```

and:

```text
site_id + timestamp
```

---

## Project Structure

```text
Building-Energy-Prediction/
│
├── energy_prediction.py
├── requirements.txt
├── README.md
│
├── train.csv
├── weather_train.csv
└── building_metadata.csv
```

---

## Data Processing Pipeline

### 1. Load Data

The three datasets are loaded using Pandas:

```python
train = pd.read_csv("train.csv")
weather = pd.read_csv("weather_train.csv")
building = pd.read_csv("building_metadata.csv")
```

### 2. Sampling

A sample of 50,000 records is selected from the training dataset to improve execution performance:

```python
train = train.sample(n=50000, random_state=42)
```

### 3. Merge Datasets

The training data is merged with building metadata and weather data.

```text
train
  │
  ├── building_id ──→ building_metadata
  │
  └── site_id + timestamp ──→ weather_train
```

### 4. Data Cleaning

The project removes:

* Zero meter readings
* Negative meter readings
* The top 1% of extreme meter-reading values

This reduces the effect of faulty readings and extreme outliers.

### 5. Feature Engineering

The timestamp is converted into several useful features:

```text
hour
day
month
weekday
is_weekend
time_period
```

`time_period` divides the day into:

```text
0 → Night
1 → Morning
2 → Afternoon
3 → Evening
```

### 6. Categorical Encoding

The `primary_use` feature is converted into numerical values using `LabelEncoder`.

### 7. Missing Value Handling

Missing values are handled using median imputation.

The imputer is fitted only on the training data and then applied to the test data to avoid data leakage.

### 8. Target Transformation

Because energy consumption can be highly skewed, the target is transformed using:

```python
y = np.log1p(df["meter_reading"])
```

Predictions are converted back to the original scale using:

```python
np.expm1()
```

---

## Model Evaluation

The following metrics are calculated for every model:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

**Lower is better.**

### MSE — Mean Squared Error

Measures the average squared prediction error.

**Lower is better.**

### RMSE — Root Mean Squared Error

The square root of MSE.

It gives greater importance to larger errors.

**Lower is better.**

### R² Score

Measures the proportion of variance explained by the model.

A value closer to `1.0` indicates stronger explanatory performance.

**Higher is better.**

---

## Model Selection

After evaluating all three models, the project selects the model with the highest R² score:

```python
best_model_name = result_df["R2 Score"].idxmax()
```

The final model comparison is displayed as:

```text
================ MODEL COMPARISON ================

                    MAE       MSE      RMSE    R2 Score
Linear Regression     ...
XGBoost               ...
LightGBM              ...
```

---

## Visualizations

The project generates several visualizations to analyze model performance.

### Correlation Matrix

Shows the correlation between the training features and the transformed target.

### RMSE Comparison

Compares the RMSE values of the three models.

### R² Comparison

Compares the R² scores of the three models.

### Predicted vs Actual

Compares the predicted energy consumption against actual energy consumption for the best-performing model.

### Residual Plot

Shows the difference between actual and predicted values.

### Feature Importance

For tree-based models such as XGBoost and LightGBM, the project displays the top 10 most important features.

---

## Installation

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

and activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Requirements

The project requires:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

---

## Running the Project

Make sure the following files are present in the project directory:

```text
train.csv
weather_train.csv
building_metadata.csv
energy_prediction.py
```

Activate the virtual environment and run:

```powershell
python energy_prediction.py
```

The program will:

1. Load the datasets
2. Process and clean the data
3. Generate features
4. Train Linear Regression
5. Tune and train XGBoost
6. Tune and train LightGBM
7. Compare model performance
8. Select the model with the highest R² score
9. Display evaluation visualizations
10. Display feature importance when applicable

---

## Expected Output

The terminal displays information such as:

```text
Original Train Shape: ...

Merged Shape: ...

===== meter_reading Diagnostics =====
Zero readings    : ...
Negative readings: ...
After cleaning   : ...

Train Size: ...
Test Size : ...

Training Linear Regression...
  R²: ...
  RMSE: ...

Tuning XGBoost...
  Best params : ...
  CV R²       : ...
  Test R²: ...
  RMSE: ...

Tuning LightGBM...
  Best params : ...
  CV R²       : ...
  Test R²: ...
  RMSE: ...

================ MODEL COMPARISON ================

...

Best Model: ...
```

The exact values depend on the dataset and execution environment.

---

## Technologies Used

| Technology   | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| Python       | Programming language                                       |
| Pandas       | Data loading and manipulation                              |
| NumPy        | Numerical operations                                       |
| Scikit-learn | Preprocessing, splitting, evaluation and Linear Regression |
| XGBoost      | Gradient boosting regression                               |
| LightGBM     | Gradient boosting regression                               |
| Matplotlib   | Data visualization                                         |
| Seaborn      | Statistical visualization                                  |

---

## Machine Learning Workflow

```text
                ┌─────────────────────┐
                │      Raw Data       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Data Integration   │
                │ Train + Weather +   │
                │ Building Metadata   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Data Cleaning     │
                │ Zero/Negative/      │
                │ Outlier Removal     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering │
                │ Time + Categorical  │
                │ Features            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Train/Test Split    │
                └──────────┬──────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │       Model Training      │
              ├─────────────┬─────────────┤
              │             │             │
              ▼             ▼             ▼
        Linear Reg.      XGBoost       LightGBM
              │             │             │
              └─────────────┼─────────────┘
                            │
                            ▼
                ┌─────────────────────┐
                │ Model Evaluation    │
                │ MAE / MSE / RMSE /  │
                │ R²                  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Best Model Selection│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Visual Analysis &   │
                │ Feature Importance  │
                └─────────────────────┘
```

---

## Notes

* The current implementation samples **50,000 training records** before merging to improve execution performance.
* The target variable is log-transformed before model training.
* Cross-validation is used for XGBoost and LightGBM hyperparameter selection.
* The best model is selected using the highest test-set R² score.
* Graph windows may need to be closed for execution to continue to the next visualization.

---

## Author

**Sakthi Mageswari V.**

Computer Science Engineering
Bengaluru, Karnataka

---

## License

This project is intended for academic and educational purposes.
