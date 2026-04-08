import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_with_context.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
print("Loading bowling context data...")
df = pd.read_csv(DATA_PATH)

# =====================================================
# FEATURES & TARGETS
# =====================================================
FEATURES = [
    "avg_wickets_last5",
    "avg_economy_last5",
    "season"
]

X = df[FEATURES]

y_wickets = df["wickets"]
y_economy = df["economy"]

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, yw_train, yw_test = train_test_split(
    X, y_wickets, test_size=0.2, random_state=42
)

_, _, ye_train, ye_test = train_test_split(
    X, y_economy, test_size=0.2, random_state=42
)

# =====================================================
# WICKET MODEL
# =====================================================
print("Training wickets model...")
wicket_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

wicket_model.fit(X_train, yw_train)
wicket_preds = wicket_model.predict(X_test)

print("Wickets Model MAE:", mean_absolute_error(yw_test, wicket_preds))
print("Wickets Model RMSE:", np.sqrt(mean_squared_error(yw_test, wicket_preds)))

joblib.dump(
    wicket_model,
    os.path.join(MODEL_DIR, "bowling_wicket_model.pkl")
)

# =====================================================
# ECONOMY MODEL
# =====================================================
print("Training economy model...")
econ_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

econ_model.fit(X_train, ye_train)
econ_preds = econ_model.predict(X_test)

print("Economy Model MAE:", mean_absolute_error(ye_test, econ_preds))
print("Economy Model RMSE:", np.sqrt(mean_squared_error(ye_test, econ_preds)))

joblib.dump(
    econ_model,
    os.path.join(MODEL_DIR, "bowling_economy_model.pkl")
)

print("✅ Bowling models trained and saved successfully")
