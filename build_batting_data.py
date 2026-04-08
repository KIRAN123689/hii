import pandas as pd
import os

# ===============================
# LOAD CLEAN DATA
# ===============================
MATCHES_FILE = "data/processed/matches_2016_2025.csv"
DELIVERIES_FILE = "data/processed/deliveries_2016_2025.csv"
OUT_FILE = "data/processed/batting_match_level.csv"

matches = pd.read_csv(MATCHES_FILE)
deliveries = pd.read_csv(DELIVERIES_FILE)

print("Matches loaded:", matches.shape)
print("Deliveries loaded:", deliveries.shape)

# ===============================
# BATSMAN MATCH-LEVEL AGGREGATION
# ===============================
batting = deliveries.groupby(
    ["match_id", "batsman"]
).agg(
    runs=("batsman_runs", "sum"),
    balls=("ball", "count"),
    fours=("batsman_runs", lambda x: (x == 4).sum()),
    sixes=("batsman_runs", lambda x: (x == 6).sum())
).reset_index()

# strike rate
batting["strike_rate"] = (batting["runs"] / batting["balls"]) * 100

# ===============================
# ADD MATCH CONTEXT
# ===============================
match_info = matches[
    ["match_id", "season", "venue", "team1", "team2"]
]

batting = batting.merge(match_info, on="match_id", how="left")

# ===============================
# SAVE OUTPUT
# ===============================
os.makedirs("data/processed", exist_ok=True)
batting.to_csv(OUT_FILE, index=False)

print("\n✅ Batting match-level data created successfully")
print("Final shape:", batting.shape)
print(batting.head())
