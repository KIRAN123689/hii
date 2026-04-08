import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# LOAD DATA
# ===============================
DATA_FILE = "data/processed/batting_with_context.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_FILE)

# ===============================
# FEATURES & TARGET
# ===============================
features = [
    "avg_runs_last5",
    "avg_sr_last5",
    "avg_runs_venue_last5",
    "avg_runs_opponent_last5",
    "inning"
]

X = df[features]
y = df["runs"]

# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# LINEAR REGRESSION
# ===============================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, lr_preds)
rmse_lr = mean_squared_error(y_test, lr_preds) ** 0.5

print("\nLinear Regression:")
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)

# ===============================
# RANDOM FOREST
# ===============================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, rf_preds)
rmse_rf = mean_squared_error(y_test, rf_preds) ** 0.5

print("\nRandom Forest:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

# ===============================
# SAVE FINAL MODEL
# ===============================
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf, os.path.join(MODEL_DIR, "batting_rf_context_model.pkl"))

print("\n✅ Context-aware Random Forest trained and saved")
