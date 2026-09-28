# ============================================================
# BUILDING ENERGY CONSUMPTION PREDICTION
# FINAL VERSION — 3 MODEL COMPARISON
#
# Models:
#   1. Linear Regression  (baseline)
#   2. XGBoost            (gradient boosting)
#   3. LightGBM           (best performer)
#
# Dataset:
#   train.csv, weather_train.csv, building_metadata.csv
#
# Target: meter_reading (regression)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

train    = pd.read_csv("train.csv")
weather  = pd.read_csv("weather_train.csv")
building = pd.read_csv("building_metadata.csv")

print("Original Train Shape:", train.shape)

# ============================================================
# STEP 2: SAMPLE BEFORE MERGE  (performance fix)
# ============================================================

train = train.sample(n=50000, random_state=42)

# ============================================================
# STEP 3: MERGE DATASETS
# ============================================================

df = train.merge(building, on="building_id", how="left")
df = df.merge(weather, on=["site_id", "timestamp"], how="left")

print("Merged Shape:", df.shape)

# ============================================================
# STEP 4: DATA QUALITY — REMOVE BAD READINGS
# ============================================================

print("\n===== meter_reading Diagnostics =====")
print("Zero readings    :", (df["meter_reading"] == 0).sum())
print("Negative readings:", (df["meter_reading"] < 0).sum())

# Remove zeros and negatives (faulty meters)
df = df[df["meter_reading"] > 0].copy()

# Remove top 1% extreme outliers
upper = df["meter_reading"].quantile(0.99)
df = df[df["meter_reading"] <= upper].copy()

print("After cleaning   :", df.shape)

# ============================================================
# STEP 5: TIMESTAMP FEATURE ENGINEERING
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

df["hour"]       = df["timestamp"].dt.hour
df["day"]        = df["timestamp"].dt.day
df["month"]      = df["timestamp"].dt.month
df["weekday"]    = df["timestamp"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

# Time of day: 0=Night, 1=Morning, 2=Afternoon, 3=Evening
df["time_period"] = pd.cut(
    df["hour"],
    bins=[-1, 5, 11, 17, 23],
    labels=[0, 1, 2, 3]
).astype(int)

df.drop("timestamp", axis=1, inplace=True)

# ============================================================
# STEP 6: ENCODE CATEGORICAL FEATURES
# ============================================================

le = LabelEncoder()
df["primary_use"] = le.fit_transform(df["primary_use"].astype(str))

# ============================================================
# STEP 7: MEMORY OPTIMIZATION (downcast dtypes)
# ============================================================

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")

# ============================================================
# STEP 8: DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop("meter_reading", axis=1)
y = np.log1p(df["meter_reading"])   # log-transform for skewed target

# ============================================================
# STEP 9: TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTrain Size:", X_train.shape)
print("Test Size :", X_test.shape)

# ============================================================
# STEP 10: IMPUTATION — fit on train only (no leakage)
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)

# ============================================================
# STEP 10.5: CORRELATION MATRIX
# Placed here so it reflects the exact features the model sees
# ============================================================

plt.figure(figsize=(12, 8))
corr_df = X_train.copy()
corr_df["meter_reading"] = y_train.values
sns.heatmap(corr_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix (Training Features)")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 11: COMPUTE y_actual ONCE
# ============================================================

y_actual = np.expm1(y_test)

results          = {}
predictions_dict = {}

# ============================================================
# STEP 12: MODEL 1 — LINEAR REGRESSION (with StandardScaler)
#
# WHY: Simple baseline. Assumes linear relationships.
#      Needs scaling because it is sensitive to feature magnitudes.
# ============================================================

print("\nTraining Linear Regression...")

scaler      = StandardScaler()
X_train_lr  = scaler.fit_transform(X_train)
X_test_lr   = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_lr, y_train)

y_pred = np.expm1(lr.predict(X_test_lr))
predictions_dict["Linear Regression"] = y_pred

mae  = mean_absolute_error(y_actual, y_pred)
mse  = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_actual, y_pred)

results["Linear Regression"] = [mae, mse, rmse, r2]
print(f"  R²: {r2:.4f}   RMSE: {rmse:.2f}")

# ============================================================
# STEP 13: MODEL 2 — XGBOOST (with grid search via CV)
#
# WHY: Gradient boosting on decision trees.
#      Handles nonlinearity, robust to outliers,
#      industry standard for tabular regression.
# ============================================================

print("\nTuning XGBoost...")

xgb_best_score = -999
xgb_best_model = None

xgb_depth_grid = [6, 8, 10]
xgb_tree_grid  = [100, 200]

for depth, trees in itertools.product(xgb_depth_grid, xgb_tree_grid):

    model = XGBRegressor(
        max_depth=depth,
        n_estimators=trees,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    score = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="r2"
    ).mean()

    if score > xgb_best_score:
        xgb_best_score = score
        xgb_best_model = model
        xgb_best_params = f"XGBoost(d={depth}, trees={trees})"

print(f"  Best params : {xgb_best_params}")
print(f"  CV R²       : {xgb_best_score:.4f}")

xgb_best_model.fit(X_train, y_train)

y_pred = np.expm1(xgb_best_model.predict(X_test))
predictions_dict["XGBoost"] = y_pred

mae  = mean_absolute_error(y_actual, y_pred)
mse  = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_actual, y_pred)

results["XGBoost"] = [mae, mse, rmse, r2]
print(f"  Test R²: {r2:.4f}   RMSE: {rmse:.2f}")

# ============================================================
# STEP 14: MODEL 3 — LIGHTGBM (with grid search via CV)
#
# WHY: Faster and often more accurate than XGBoost.
#      Handles large datasets efficiently with leaf-wise growth.
#      Strong performer on energy/tabular Kaggle problems.
# ============================================================

print("\nTuning LightGBM...")

lgb_best_score = -999
lgb_best_model = None

lgb_depth_grid = [8, 12, 16]
lgb_tree_grid  = [100, 200]

for depth, trees in itertools.product(lgb_depth_grid, lgb_tree_grid):

    model = LGBMRegressor(
        max_depth=depth,
        n_estimators=trees,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    score = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="r2"
    ).mean()

    if score > lgb_best_score:
        lgb_best_score = score
        lgb_best_model = model
        lgb_best_params = f"LightGBM(d={depth}, trees={trees})"

print(f"  Best params : {lgb_best_params}")
print(f"  CV R²       : {lgb_best_score:.4f}")

lgb_best_model.fit(X_train, y_train)

y_pred = np.expm1(lgb_best_model.predict(X_test))
predictions_dict["LightGBM"] = y_pred

mae  = mean_absolute_error(y_actual, y_pred)
mse  = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_actual, y_pred)

results["LightGBM"] = [mae, mse, rmse, r2]
print(f"  Test R²: {r2:.4f}   RMSE: {rmse:.2f}")

# ============================================================
# STEP 15: RESULTS TABLE
# ============================================================

result_df = pd.DataFrame(
    results,
    index=["MAE", "MSE", "RMSE", "R2 Score"]
).T

print("\n================ MODEL COMPARISON ================\n")
print(result_df.to_string())

# ============================================================
# STEP 16: BEST MODEL SELECTION (highest R²)
# ============================================================

best_model_name = result_df["R2 Score"].idxmax()
best_pred       = predictions_dict[best_model_name]

print(f"\nBest Model: {best_model_name}")

# ============================================================
# STEP 17: RMSE BAR CHART
# ============================================================

plt.figure(figsize=(8, 5))
sns.barplot(x=result_df.index, y=result_df["RMSE"])
plt.title("RMSE Comparison (lower is better)")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 18: R² BAR CHART
# ============================================================

plt.figure(figsize=(8, 5))
sns.barplot(x=result_df.index, y=result_df["R2 Score"])
plt.title("R² Score Comparison (higher is better)")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 19: PREDICTED vs ACTUAL  (best model)
# ============================================================

plt.figure(figsize=(7, 6))
plt.scatter(y_actual[:500], best_pred[:500], alpha=0.6)
plt.xlabel("Actual Energy (kWh)")
plt.ylabel("Predicted Energy (kWh)")
plt.title(f"Predicted vs Actual — {best_model_name}")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 20: RESIDUAL PLOT  (best model)
# ============================================================

residuals = y_actual - best_pred

plt.figure(figsize=(7, 5))
plt.scatter(best_pred[:500], residuals[:500], alpha=0.6)
plt.axhline(y=0, linestyle="--", color="red")
plt.xlabel("Predicted Energy (kWh)")
plt.ylabel("Residual Error")
plt.title(f"Residual Plot — {best_model_name}")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 21: FEATURE IMPORTANCE  (best model, if tree-based)
# ============================================================

best_model_obj = {
    "XGBoost" : xgb_best_model,
    "LightGBM": lgb_best_model
}.get(best_model_name)

if best_model_obj is not None and hasattr(best_model_obj, "feature_importances_"):

    importance = pd.Series(
        best_model_obj.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title(f"Top 10 Important Features — {best_model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# ============================================================
# FINAL METRIC GUIDE
# ------------------------------------------------------------
# Lower MAE  = better average error
# Lower RMSE = better, penalises large errors more
# Higher R²  = more variance explained (1.0 = perfect)
# ============================================================
