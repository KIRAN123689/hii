import pandas as pd
import os

print("Loading bowling match-level data...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_match_level.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_with_context.csv"
)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(INPUT_PATH)

print("Raw shape:", df.shape)

# Ensure correct sorting
df = df.sort_values(["bowler", "season", "match_id"])

# ----------------------------
# ROLLING FEATURES (LAST 5 MATCHES)
# ----------------------------
print("Creating rolling features...")

df["avg_wickets_last5"] = (
    df.groupby("bowler")["wickets"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

df["avg_economy_last5"] = (
    df.groupby("bowler")["economy"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

df["avg_overs_last5"] = (
    df.groupby("bowler")["overs"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ----------------------------
# VENUE-SPECIFIC ROLLING FEATURES
# ----------------------------
df["avg_wickets_venue_last5"] = (
    df.groupby(["bowler", "venue"])["wickets"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

df["avg_economy_venue_last5"] = (
    df.groupby(["bowler", "venue"])["economy"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ----------------------------
# CLEAN NaNs (FIRST MATCHES)
# ----------------------------
df.fillna({
    "avg_wickets_last5": 0,
    "avg_economy_last5": df["economy"].mean(),
    "avg_overs_last5": 0,
    "avg_wickets_venue_last5": 0,
    "avg_economy_venue_last5": df["economy"].mean()
}, inplace=True)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Bowling context data created successfully")
print("Final shape:", df.shape)
print("Saved to:", OUTPUT_PATH)
