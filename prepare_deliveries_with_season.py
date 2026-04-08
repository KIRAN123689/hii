import pandas as pd
import os

# ===============================
# PATHS
# ===============================
DELIVERIES_PATH = "data/processed/deliveries_2016_2025.csv"
MATCHES_PATH = "data/processed/matches_2016_2025.csv"
OUT_PATH = "data/processed/deliveries_with_season.csv"

# ===============================
# LOAD DATA
# ===============================
deliveries = pd.read_csv(DELIVERIES_PATH)
matches = pd.read_csv(MATCHES_PATH)

# ===============================
# MERGE SEASON INTO DELIVERIES
# ===============================
deliveries = deliveries.merge(
    matches[["match_id", "season"]],
    on="match_id",
    how="left"
)

# ===============================
# SAVE
# ===============================
os.makedirs("data/processed", exist_ok=True)
deliveries.to_csv(OUT_PATH, index=False)

print("✅ deliveries_with_season.csv created")
print("Shape:", deliveries.shape)
