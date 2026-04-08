import pandas as pd
import os

print("Loading deliveries and matches data...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DELIVERIES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "deliveries_with_season.csv"
)
MATCHES_PATH = os.path.join(
    BASE_DIR, "data", "raw", "matches.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_match_level.csv"
)

# ----------------------------
# LOAD DATA
# ----------------------------
deliveries = pd.read_csv(DELIVERIES_PATH)
matches = pd.read_csv(MATCHES_PATH)

print("Deliveries shape:", deliveries.shape)
print("Matches shape:", matches.shape)

# ----------------------------
# BASIC CLEANING
# ----------------------------
deliveries["is_wicket"] = deliveries["dismissal_kind"].notna().astype(int)
# ----------------------------
# RUNS CONCEDED (DATASET-SAFE)
# ----------------------------
deliveries["runs_conceded"] = (
    deliveries["batsman_runs"] + deliveries["extras"]
)


# ----------------------------
# BUILD MATCH-LEVEL BOWLING STATS
# ----------------------------
print("Building match-level bowling statistics...")

bowling_match = deliveries.groupby(
    ["match_id", "season", "bowler", "bowling_team"]
).agg(
    balls=("ball", "count"),
    runs_conceded=("runs_conceded", "sum"),
    wickets=("is_wicket", "sum")
).reset_index()

bowling_match["overs"] = bowling_match["balls"] / 6
bowling_match["economy"] = bowling_match["runs_conceded"] / bowling_match["overs"]

# ----------------------------
# MERGE VENUE FROM MATCHES
# ----------------------------
print("Merging venue information...")

matches_subset = matches[["matchId", "venue", "team1", "team2"]].rename(
    columns={"matchId": "match_id"}
)

bowling_match = bowling_match.merge(
    matches_subset, on="match_id", how="left"
)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
bowling_match.to_csv(OUTPUT_PATH, index=False)

print("✅ Bowling match-level data with venue created successfully")
print("Final shape:", bowling_match.shape)
print("Saved to:", OUTPUT_PATH)
