import pandas as pd
import os

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "deliveries_with_season.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_match_level.csv"
)

print("Loading deliveries data...")
deliveries = pd.read_csv(INPUT_PATH)

print("Raw shape:", deliveries.shape)

# =====================================================
# RUNS CONCEDED
# =====================================================
deliveries["runs_conceded"] = (
    deliveries["batsman_runs"]
    + deliveries.get("extras", 0)
    - deliveries.get("Byes", 0)
    - deliveries.get("LegByes", 0)
)

# =====================================================
# WICKET FLAG (exclude run outs)
# =====================================================
deliveries["is_wicket"] = deliveries["dismissal_kind"].notna()
deliveries.loc[
    deliveries["dismissal_kind"].isin(["run out", "retired hurt"]),
    "is_wicket"
] = False

# =====================================================
# AGGREGATE PER BOWLER PER MATCH
# =====================================================
bowling = deliveries.groupby(
    ["match_id", "season", "bowler", "bowling_team", "batting_team"]
).agg(
    runs_conceded=("runs_conceded", "sum"),
    balls_bowled=("ball", "count"),
    wickets=("is_wicket", "sum"),
    fours_conceded=("batsman_runs", lambda x: (x == 4).sum()),
    sixes_conceded=("batsman_runs", lambda x: (x == 6).sum())
).reset_index()

# =====================================================
# ECONOMY
# =====================================================
bowling["overs"] = bowling["balls_bowled"] / 6
bowling["economy"] = bowling["runs_conceded"] / bowling["overs"]

# Rename batting_team to opponent
bowling = bowling.rename(columns={
    "batting_team": "opponent"
})

print("Processed shape:", bowling.shape)

# =====================================================
# SAVE OUTPUT
# =====================================================
bowling.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)
