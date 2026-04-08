import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("Loading bowling context data...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_with_context.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

print("Raw shape:", df.shape)

# ----------------------------
# FEATURES & TARGETS
# ----------------------------
FEATURES = [
    "avg_wickets_last5",
    "avg_economy_last5",
    "avg_overs_last5",
    "avg_wickets_venue_last5",
    "avg_economy_venue_last5"
]

X = df[FEATURES]

y_wickets = df["wickets"]
y_economy = df["economy"]

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, yw_train, yw_test = train_test_split(
    X, y_wickets, test_size=0.2, random_state=42
)

_, _, ye_train, ye_test = train_test_split(
    X, y_economy, test_size=0.2, random_state=42
)

# ----------------------------
# TRAIN WICKETS MODEL
# ----------------------------
print("Training wickets model...")

wicket_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

wicket_model.fit(X_train, yw_train)
wicket_preds = wicket_model.predict(X_test)

print("Wickets MAE:", mean_absolute_error(yw_test, wicket_preds))
print("Wickets RMSE:", np.sqrt(mean_squared_error(yw_test, wicket_preds)))

# ----------------------------
# TRAIN ECONOMY MODEL
# ----------------------------
print("Training economy model...")

economy_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

economy_model.fit(X_train, ye_train)
economy_preds = economy_model.predict(X_test)

print("Economy MAE:", mean_absolute_error(ye_test, economy_preds))
print("Economy RMSE:", np.sqrt(mean_squared_error(ye_test, economy_preds)))

# ----------------------------
# SAVE MODELS
# ----------------------------
joblib.dump(
    wicket_model,
    os.path.join(MODEL_DIR, "bowling_wicket_model.pkl")
)

joblib.dump(
    economy_model,
    os.path.join(MODEL_DIR, "bowling_economy_model.pkl")
)

print("✅ Bowling models trained and saved successfully")
