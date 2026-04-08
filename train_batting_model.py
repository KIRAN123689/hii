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
DATA_FILE = "data/processed/batting_recent_form.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_FILE)

# ===============================
# FEATURES & TARGET
# ===============================
X = df[["avg_runs_last5", "avg_sr_last5"]]
y = df["runs"]

# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ===============================
# BASELINE MODEL — LINEAR REGRESSION
# ===============================
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, lr_preds)
rmse_lr = mean_squared_error(y_test, lr_preds) ** 0.5

print("\nLinear Regression Results:")
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)

# ===============================
# FINAL MODEL — RANDOM FOREST
# ===============================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, rf_preds)
rmse_rf = mean_squared_error(y_test, rf_preds) ** 0.5

print("\nRandom Forest Results:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

# ===============================
# SAVE MODELS
# ===============================
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(lr, os.path.join(MODEL_DIR, "batting_lr_model.pkl"))
joblib.dump(rf, os.path.join(MODEL_DIR, "batting_rf_model.pkl"))

print("\n✅ Models trained and saved successfully")
