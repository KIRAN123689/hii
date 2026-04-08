import pandas as pd
import os

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_match_level.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bowling_with_context.csv"
)

print("Loading bowling match-level data...")
df = pd.read_csv(INPUT_PATH)

print("Input shape:", df.shape)

# =====================================================
# SORT PROPERLY (CRITICAL)
# =====================================================
df = df.sort_values(
    ["bowler", "season", "match_id"]
)

# =====================================================
# RECENT FORM FEATURES (LAST 5 MATCHES)
# =====================================================
df["avg_wickets_last5"] = (
    df.groupby("bowler")["wickets"]
      .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

df["avg_economy_last5"] = (
    df.groupby("bowler")["economy"]
      .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

print("Recent form features added.")

# =====================================================
# SAVE OUTPUT
# =====================================================
df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)
