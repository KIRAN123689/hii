import pandas as pd
import os

# ===============================
# FILE PATHS
# ===============================
RAW_MATCHES = "data/raw/matches.csv"
RAW_DELIVERIES = "data/raw/deliveries.csv"
OUT_DIR = "data/processed"

# ===============================
# LOAD DATA
# ===============================
matches = pd.read_csv(RAW_MATCHES)
deliveries = pd.read_csv(RAW_DELIVERIES)

print("Raw matches shape:", matches.shape)
print("Raw deliveries shape:", deliveries.shape)

# ===============================
# STANDARDIZE MATCH ID
# BOTH FILES USE 'matchId'
# ===============================
matches.rename(columns={"matchId": "match_id"}, inplace=True)
deliveries.rename(columns={"matchId": "match_id"}, inplace=True)

# ===============================
# FIX SEASON COLUMN
# Handles values like '2007/08'
# ===============================
matches["season"] = (
    matches["season"]
    .astype(str)
    .str.slice(0, 4)
    .astype(int)
)

# ===============================
# FILTER SEASONS: 2016–2025
# ===============================
matches_2016_2025 = matches[
    (matches["season"] >= 2016) &
    (matches["season"] <= 2025)
]

print(
    "Filtered matches seasons:",
    sorted(matches_2016_2025["season"].unique())
)

# ===============================
# FILTER DELIVERIES USING MATCH IDs
# ===============================
valid_match_ids = matches_2016_2025["match_id"]

deliveries_2016_2025 = deliveries[
    deliveries["match_id"].isin(valid_match_ids)
]

# ===============================
# REMOVE NOISE
# ===============================
# Remove no-result matches
if "result" in matches_2016_2025.columns:
    matches_2016_2025 = matches_2016_2025[
        matches_2016_2025["result"] != "no result"
    ]

# Remove super overs (if column exists)
if "is_super_over" in deliveries_2016_2025.columns:
    deliveries_2016_2025 = deliveries_2016_2025[
        deliveries_2016_2025["is_super_over"] == 0
    ]

# ===============================
# SAVE CLEAN DATA
# ===============================
os.makedirs(OUT_DIR, exist_ok=True)

matches_2016_2025.to_csv(
    os.path.join(OUT_DIR, "matches_2016_2025.csv"),
    index=False
)

deliveries_2016_2025.to_csv(
    os.path.join(OUT_DIR, "deliveries_2016_2025.csv"),
    index=False
)

print("\n✅ Cleaned 2016–2025 data created successfully")
print("Final matches shape:", matches_2016_2025.shape)
print("Final deliveries shape:", deliveries_2016_2025.shape)
