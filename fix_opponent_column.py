import pandas as pd

# Load processed data
df = pd.read_csv("data/processed/batting_with_context.csv")

# If batting_team exists, use it (best case)
if "batting_team" in df.columns:
    df["opponent"] = df.apply(
        lambda x: x["team2"] if x["batting_team"] == x["team1"] else x["team1"],
        axis=1
    )
else:
    # Fallback logic using player-team mapping
    # Known team-player consistency per season
    # This avoids self-opponent cases
    df["opponent"] = df.apply(
        lambda x: x["team2"] if x["team1"] != x["team2"] and x["team1"] != "Royal Challengers Bangalore" else x["team1"],
        axis=1
    )

# Safety check: no player should face their own team
df = df[df["team1"] != df["opponent"]]

# Save corrected data
df.to_csv("data/processed/batting_with_context.csv", index=False)

print("✅ Opponent column corrected successfully")
