import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo

print("Downloading UCI Energy Dataset...")

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
dataset = fetch_ucirepo(id=374)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ---------------------------------------------------------
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

# ---------------------------------------------------------
# 3. FIX DATE FORMAT
# ---------------------------------------------------------
df["date"] = df["date"].astype(str).str.replace(
    r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})",
    r"\1 \2",
    regex=True
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# 4. TIME FEATURES
# ---------------------------------------------------------
df["hour"] = df["date"].dt.hour + df["date"].dt.minute / 60
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["month"] = df["date"].dt.month

# Cyclic time features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# ---------------------------------------------------------
# 5. LAG FEATURES
# ---------------------------------------------------------
# Previous energy consumption
df["lag_1"] = df["Appliances"].shift(1)
df["lag_2"] = df["Appliances"].shift(2)
df["lag_3"] = df["Appliances"].shift(3)

# Consumption approximately one hour earlier
# Dataset interval is 10 minutes
df["lag_6"] = df["Appliances"].shift(6)

# Consumption approximately one day earlier
df["lag_144"] = df["Appliances"].shift(144)

# ---------------------------------------------------------
# 6. ROLLING FEATURES
# ---------------------------------------------------------
df["rolling_mean_3"] = (
    df["Appliances"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

df["rolling_mean_6"] = (
    df["Appliances"]
    .shift(1)
    .rolling(window=6)
    .mean()
)

df["rolling_mean_12"] = (
    df["Appliances"]
    .shift(1)
    .rolling(window=12)
    .mean()
)

# ---------------------------------------------------------
# 7. REMOVE ROWS CREATED BY LAG FEATURES
# ---------------------------------------------------------
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# 8. FEATURES
# ---------------------------------------------------------
features = [
    # Time
    "hour",
    "day_of_week",
    "is_weekend",
    "month",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",

    # Lighting
    "lights",

    # Temperature
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",

    # Humidity
    "RH_1",
    "RH_2",
    "RH_3",
    "RH_4",
    "RH_5",
    "RH_6",
    "RH_7",
    "RH_8",
    "RH_9",

    # Weather
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",

    # Previous consumption
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_144",

    # Recent consumption trends
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12"
]

target = "Appliances"

# ---------------------------------------------------------
# 9. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X = df[features]
y = df[target]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining model...")

# ---------------------------------------------------------
# 10. MODEL
# ---------------------------------------------------------
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 11. PREDICTION
# ---------------------------------------------------------
predictions = model.predict(X_test)

# ---------------------------------------------------------
# 12. MODEL EVALUATION
# ---------------------------------------------------------
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.2f} Wh")
print(f"RMSE : {rmse:.2f} Wh")
print(f"R²   : {r2:.4f}")

# ---------------------------------------------------------
# 13. ANOMALY DETECTION
# ---------------------------------------------------------
print("\nTraining anomaly detection model...")

anomaly_data = df.copy()

anomaly_data["hour_num"] = (
    anomaly_data["date"].dt.hour
    + anomaly_data["date"].dt.minute / 60
)

anomaly_features = [
    "Appliances",
    "lights",
    "hour_num",
    "day_of_week",
    "T_out",
    "RH_out"
]

anomaly_model = IsolationForest(
    n_estimators=150,
    contamination=0.02,
    random_state=42
)

anomaly_model.fit(
    anomaly_data[anomaly_features]
)

# ---------------------------------------------------------
# 14. SAVE MODELS
# ---------------------------------------------------------
joblib.dump(model, "energy_model.pkl")
joblib.dump(anomaly_model, "anomaly_model.pkl")

# Save feature list so the dashboard uses exactly
# the same features as the training model.
joblib.dump(features, "model_features.pkl")

print("\nModels saved successfully!")

print("\nFiles created:")
print("✓ energy_model.pkl")
print("✓ anomaly_model.pkl")
print("✓ model_features.pkl")

print("\nGreenPulse training completed.")