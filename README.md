# Building Energy Consumption Prediction

A machine learning project for predicting building energy consumption using **weather conditions, building metadata, and time-based features**. The project compares Linear Regression, XGBoost, and LightGBM models and evaluates their performance using multiple regression metrics.

---

## Results

Three regression models were trained and evaluated on the processed dataset.

| Model             |        MAE |       RMSE |         R² |
| ----------------- | ---------: | ---------: | ---------: |
| Linear Regression |     318.81 |    1074.41 |    -1.5398 |
| **XGBoost**       | **115.62** | **325.51** | **0.7669** |
| LightGBM          |     165.39 |     452.80 |     0.5489 |

### Best Performing Model

**XGBoost**

* **Test R²:** 0.7669
* **Test RMSE:** 325.51
* **Test MAE:** 115.62
* **Cross-validation R²:** 0.8265
* **Best configuration:** `max_depth=10`, `n_estimators=200`

XGBoost achieved the strongest test-set performance among the three evaluated models.

---

## Project Overview

Building energy consumption is influenced by several factors, including building characteristics, weather conditions, time of day, seasonality, and usage patterns.

This project combines these sources of information and builds regression models to predict the `meter_reading` value.

The workflow covers:

* Data integration and preprocessing
* Sampling for computational efficiency
* Data cleaning and outlier handling
* Timestamp-based feature engineering
* Categorical feature encoding
* Missing-value imputation
* Memory optimization
* Target transformation
* Model training
* Hyperparameter tuning with cross-validation
* Model comparison
* Performance evaluation
* Visualization
* Feature importance analysis

---

## Dataset

The project uses data from the **ASHRAE Great Energy Predictor III** dataset.

Three datasets are required:

```text
train.csv
weather_train.csv
building_metadata.csv
```

### `train.csv`

Contains energy meter readings used as the prediction target.

**Target variable:**

```text
meter_reading
```

### `weather_train.csv`

Contains weather information associated with sites and timestamps.

### `building_metadata.csv`

Contains information describing individual buildings.

### Data Integration

The datasets are connected using:

```text
building_id
```

and:

```text
site_id + timestamp
```

> **Note:** The original training dataset contains approximately **20.2 million records**. To make local execution computationally practical, this implementation samples **50,000 records** before performing the merge and subsequent processing.

---

## Machine Learning Pipeline

```text
Raw Data
   │
   ▼
Data Sampling
   │
   ▼
Data Integration
   │
   ├── Train Data
   ├── Weather Data
   └── Building Metadata
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Categorical Encoding
   │
   ▼
Missing Value Imputation
   │
   ▼
Log Transformation
   │
   ▼
Train / Test Split
   │
   ├──────────────┬──────────────┐
   ▼              ▼              ▼
Linear         XGBoost        LightGBM
Regression
   │              │              │
   └──────────────┼──────────────┘
                  ▼
           Model Evaluation
                  │
                  ▼
       Performance Comparison
                  │
                  ▼
       Feature & Error Analysis
```

---

## Data Processing

### 1. Data Sampling

The original training dataset contains approximately 20 million records. A reproducible sample of 50,000 records is selected to reduce local computation and memory requirements.

```python
train = train.sample(n=50000, random_state=42)
```

### 2. Data Cleaning

The preprocessing pipeline removes:

* Zero meter readings
* Negative meter readings
* The top 1% of extreme meter-reading values

The dataset contained **4,631 zero readings and no negative readings** before cleaning. After cleaning, 44,915 records remained.

### 3. Feature Engineering

Timestamp information is converted into useful temporal features:

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

### 4. Categorical Encoding

The `primary_use` feature is converted into numerical values using `LabelEncoder`.

### 5. Missing Values

Missing values are handled using median imputation.

The imputer is fitted on the training data and subsequently applied to the test data to avoid data leakage.

### 6. Target Transformation

Because energy consumption can be highly skewed, the target is transformed using a logarithmic transformation:

```python
y = np.log1p(df["meter_reading"])
```

Predictions are converted back to the original scale using:

```python
np.expm1()
```

---

## Models

### 1. Linear Regression

Linear Regression is used as the baseline model.

A `StandardScaler` is applied before training to normalize the numerical feature scales.

### 2. XGBoost

XGBoost is used as a gradient-boosting regression model.

The following hyperparameters are explored:

```text
max_depth      → 6, 8, 10
n_estimators   → 100, 200
learning_rate  → 0.05
```

Three-fold cross-validation is used for hyperparameter selection.

**Best configuration:**

```text
max_depth = 10
n_estimators = 200
learning_rate = 0.05
```

**Cross-validation R²:** `0.8265`

**Test R²:** `0.7669`

### 3. LightGBM

LightGBM is used as a second gradient-boosting approach designed for efficient tabular machine learning.

The following configurations are explored:

```text
max_depth      → 8, 12, 16
n_estimators   → 100, 200
learning_rate  → 0.05
```

Three-fold cross-validation is used for hyperparameter selection.

**Best configuration:**

```text
max_depth = 16
n_estimators = 200
learning_rate = 0.05
```

**Cross-validation R²:** `0.7294`

**Test R²:** `0.5489`

---

## Model Evaluation

The models are evaluated using four regression metrics.

| Metric   | Description                                   | Better Value |
| -------- | --------------------------------------------- | ------------ |
| **MAE**  | Average absolute prediction error             | Lower        |
| **MSE**  | Average squared prediction error              | Lower        |
| **RMSE** | Square root of MSE; emphasizes larger errors  | Lower        |
| **R²**   | Proportion of variance explained by the model | Higher       |

### Evaluation Results

```text
================ MODEL COMPARISON ================

                         MAE          MSE       RMSE       R²
Linear Regression      318.81    1.15e+06    1074.41   -1.5398
XGBoost                115.62    1.06e+05     325.51    0.7669
LightGBM               165.39    2.05e+05     452.80    0.5489
```

---

## Visual Analysis

The project generates visualizations to further analyze model behavior and performance:

### Correlation Matrix

Examines relationships between input features and the transformed target.

### Model Performance Comparison

Compares RMSE and R² across the three models.

### Predicted vs Actual

Visualizes predicted energy consumption against actual values for the selected model.

### Residual Analysis

Examines the difference between actual and predicted values.

### Feature Importance

Displays the most influential features identified by tree-based models such as XGBoost and LightGBM.

---

## Project Structure

```text
Building_Energy_Consumption_ML/
│
├── energy_prediction.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
└── data/
    ├── train.csv
    ├── weather_train.csv
    └── building_metadata.csv
```

> The raw dataset is not included in the repository if it exceeds GitHub's practical repository size limits. Download the required ASHRAE files separately and place them in the expected data location.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sakthi-05/Building_Energy_Consumption_ML.git
cd Building_Energy_Consumption_ML
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
```

### 3. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Requirements

The project uses:

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

Make sure the required dataset files are available:

```text
train.csv
weather_train.csv
building_metadata.csv
```

Then run:

```powershell
python energy_prediction.py
```

The program will:

1. Load the datasets
2. Sample the training data
3. Merge weather and building information
4. Clean the data
5. Generate temporal and categorical features
6. Perform train/test splitting
7. Train Linear Regression
8. Tune XGBoost using cross-validation
9. Tune LightGBM using cross-validation
10. Compare model performance
11. Display evaluation results
12. Generate visualizations
13. Analyze feature importance

---

## Expected Output

A typical execution produces output similar to:

```text
Original Train Shape: (20216100, 4)
Merged Shape: (50000, 16)

===== meter_reading Diagnostics =====
Zero readings    : 4631
Negative readings: 0
After cleaning   : (44915, 16)

Train Size: (35932, 20)
Test Size : (8983, 20)

Training Linear Regression...
R²: -1.5398
RMSE: 1074.41

Tuning XGBoost...
Best params: max_depth=10, n_estimators=200
CV R²: 0.8265
Test R²: 0.7669
RMSE: 325.51

Tuning LightGBM...
Best params: max_depth=16, n_estimators=200
CV R²: 0.7294
Test R²: 0.5489
RMSE: 452.80

================ MODEL COMPARISON ================

Best Model: XGBoost
```

Exact values may vary depending on the dataset, environment, and implementation.

---

## Technologies Used

| Technology       | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| **Python**       | Core programming language                       |
| **Pandas**       | Data loading and manipulation                   |
| **NumPy**        | Numerical computation                           |
| **Scikit-learn** | Preprocessing, evaluation and Linear Regression |
| **XGBoost**      | Gradient-boosting regression                    |
| **LightGBM**     | Gradient-boosting regression                    |
| **Matplotlib**   | Data visualization                              |
| **Seaborn**      | Statistical visualization                       |

---

## Key Takeaways

* Tree-based boosting models substantially outperformed the Linear Regression baseline.
* **XGBoost achieved the highest test R² (0.7669)** among the evaluated models.
* Feature engineering from timestamp data provides additional information for predicting energy consumption.
* Cross-validation was used during hyperparameter tuning for XGBoost and LightGBM.
* The preprocessing pipeline includes explicit handling of missing values, outliers, categorical variables, and target skew.

---

## Future Improvements

Potential extensions include:

* Training on a larger portion of the original dataset
* More extensive hyperparameter optimization
* Additional temporal and weather-derived features
* Comparing additional regression algorithms
* Improving model validation methodology
* Building a prediction API
* Deploying the trained model as a web application
* Experimenting with ensemble methods

---

## Author

**Sakthi Mageswari V.**

Computer Science Engineering Student
Bengaluru, India

[GitHub](https://github.com/Sakthi-05)

---

## License

This project is licensed under the MIT License.

Built for academic learning and experimentation in machine learning and energy prediction.
