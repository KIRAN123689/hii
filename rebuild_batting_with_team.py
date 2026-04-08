import pandas as pd
import os

# ===============================
# LOAD DATA
# ===============================
DELIVERIES_FILE = "data/processed/deliveries_2016_2025.csv"
MATCHES_FILE = "data/processed/matches_2016_2025.csv"
OUT_FILE = "data/processed/batting_with_context_fixed.csv"

deliveries = pd.read_csv(DELIVERIES_FILE)
matches = pd.read_csv(MATCHES_FILE)

# ===============================
# BATSMAN MATCH-LEVEL AGGREGATION
# ===============================
batting = deliveries.groupby(
    ["match_id", "batsman", "batting_team"]
).agg(
    runs=("batsman_runs", "sum"),
    balls=("ball", "count")
).reset_index()

batting["strike_rate"] = (batting["runs"] / batting["balls"]) * 100

# ===============================
# MERGE MATCH CONTEXT
# ===============================
batting = batting.merge(
    matches[["match_id", "season", "venue", "team1", "team2"]],
    on="match_id",
    how="left"
)

# ===============================
# CORRECT OPPONENT LOGIC (FINAL)
# ===============================
batting["opponent"] = batting.apply(
    lambda x: x["team2"] if x["batting_team"] == x["team1"] else x["team1"],
    axis=1
)

# ===============================
# SAVE OUTPUT
# ===============================
os.makedirs("data/processed", exist_ok=True)
batting.to_csv(OUT_FILE, index=False)

print("✅ batting_with_context_fixed.csv created")
print("Final shape:", batting.shape)
