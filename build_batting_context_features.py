import pandas as pd
import os

# ===============================
# LOAD DATA
# ===============================
INPUT_FILE = "data/processed/batting_recent_form.csv"
OUT_FILE = "data/processed/batting_with_context.csv"

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

# ===============================
# INNINGS FEATURE
# ===============================
# If innings column exists, keep it; else default to 1
if "inning" not in df.columns:
    df["inning"] = 1

# ===============================
# OPPONENT FEATURE (FIXED LOGIC)
# ===============================
# opponent = the other team
df["opponent"] = df.apply(
    lambda x: x["team2"] if x["team1"] == x["team1"] else x["team1"],
    axis=1
)

# ===============================
# VENUE-WISE RECENT FORM
# ===============================
df = df.sort_values(["batsman", "venue", "season", "match_id"])

df["avg_runs_venue_last5"] = (
    df.groupby(["batsman", "venue"])["runs"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

# ===============================
# OPPONENT-WISE RECENT FORM
# ===============================
df = df.sort_values(["batsman", "opponent", "season", "match_id"])

df["avg_runs_opponent_last5"] = (
    df.groupby(["batsman", "opponent"])["runs"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

# ===============================
# FILL MISSING VALUES
# ===============================
df.fillna(0, inplace=True)

# ===============================
# SAVE OUTPUT
# ===============================
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print("✅ batting_with_context.csv created")
print("Final shape:", df.shape)
