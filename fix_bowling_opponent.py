import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_with_context.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_with_context.csv"
)

print("Loading bowling context data...")
df = pd.read_csv(INPUT_PATH)

print("Columns before:", df.columns.tolist())

# ----------------------------
# DERIVE OPPONENT
# ----------------------------
def get_opponent(row):
    if row["bowling_team"] == row["team1"]:
        return row["team2"]
    elif row["bowling_team"] == row["team2"]:
        return row["team1"]
    else:
        return None

df["opponent"] = df.apply(get_opponent, axis=1)

# ----------------------------
# SANITY CHECK
# ----------------------------
missing = df["opponent"].isna().sum()
print("Missing opponent rows:", missing)

# ----------------------------
# SAVE BACK (OVERWRITE SAFELY)
# ----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Opponent column added successfully")
print("Columns after:", df.columns.tolist())
