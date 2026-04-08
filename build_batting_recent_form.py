import pandas as pd
import os

# ===============================
# LOAD DATA
# ===============================
INPUT_FILE = "data/processed/batting_match_level.csv"
OUT_FILE = "data/processed/batting_recent_form.csv"

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

# ===============================
# SORT DATA (CRITICAL FOR RECENT FORM)
# ===============================
df = df.sort_values(
    by=["batsman", "season", "match_id"]
)

# ===============================
# RECENT FORM FEATURES (LAST 5 MATCHES)
# ===============================
df["avg_runs_last5"] = (
    df.groupby("batsman")["runs"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df["avg_sr_last5"] = (
    df.groupby("batsman")["strike_rate"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# ===============================
# SAVE OUTPUT
# ===============================
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print("✅ Batting recent-form features created")
print("Final shape:", df.shape)
print(df[[
    "batsman", "runs", "strike_rate",
    "avg_runs_last5", "avg_sr_last5"
]].head(10))
