import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# Custom CSS for colors and animations
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        color: white;
    }
    .stMetric label {
        color: #FFD700;
    }
    .stMetric .metric-value {
        color: #00FF00;
        font-size: 24px;
    }
    .stButton button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stSelectbox, .stRadio {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    h1, h2, h3 {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load cricket animation
# lottie_cricket = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3h.json")

# ===================== APP CONFIG =====================
st.set_page_config(page_title="IPL Player Performance & Analytics", layout="wide", page_icon="🏏")
# col1, col2 = st.columns([1, 4])
# with col1:
#     st_lottie(lottie_cricket, height=100, key="cricket")
# with col2:
st.title("🏏 IPL Player Performance Prediction & Analytics (2016–2025)")

# ===================== ROLE SELECTION =====================
with st.sidebar:
    st.header("🎨 Theme Settings")
    theme = st.selectbox("Choose Theme", ["Default", "Dark", "Light"])
    st.header("🏏 Select Role")
    role = st.radio("Role", ["Batting", "Bowling"], label_visibility="collapsed")

# =====================================================
# ===================== BATSMAN MODE ==================
# =====================================================
if role == "Batting":

    # =====================================================
    # PATHS
    # =====================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "batting_rf_context_model.pkl")
    BAT_PATH = os.path.join(BASE_DIR, "data", "processed", "batting_with_context_fixed.csv")
    DEL_PATH = os.path.join(BASE_DIR, "data", "processed", "deliveries_with_season.csv")

    # =====================================================
    # LOAD DATA
    # =====================================================
    model = joblib.load(MODEL_PATH)
    bat_df = pd.read_csv(BAT_PATH)
    del_df = pd.read_csv(DEL_PATH)

    # =====================================================
    # TEAM NAME NORMALIZATION (SAVEPOINT)
    # =====================================================
    TEAM_NAME_MAP = {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiants": "Rising Pune Supergiant"
    }

    for col in ["batting_team", "opponent", "team1", "team2"]:
        if col in bat_df.columns:
            bat_df[col] = bat_df[col].replace(TEAM_NAME_MAP)

    for col in ["batting_team", "bowling_team"]:
        if col in del_df.columns:
            del_df[col] = del_df[col].replace(TEAM_NAME_MAP)

    # =====================================================
    # PHASE LOGIC
    # =====================================================
    def get_phase(over):
        if over <= 6:
            return "Powerplay"
        elif over <= 15:
            return "Middle"
        else:
            return "Death"

    del_df["phase"] = del_df["over"].apply(get_phase)

    # =====================================================
    # PLAYER SELECTION
    # =====================================================
    player = st.selectbox("Select Player", sorted(bat_df["batsman"].unique()))
    bat_p = bat_df[bat_df["batsman"] == player]
    del_p = del_df[del_df["batsman"] == player]
    # =====================================================
    # 🏏 PLAYER OVERALL BATTING PERFORMANCE (CAREER SUMMARY)
    # =====================================================
    st.divider()
    st.header("Player Overall Batting Performance (2016–2025)")

    # Total runs, balls, strike rate
    total_runs = del_p["batsman_runs"].sum()
    total_balls = len(del_p)
    overall_sr = (total_runs / total_balls) * 100 if total_balls else 0

    # Matches & innings
    matches_played = bat_p["match_id"].nunique()
    innings_played = bat_p.shape[0]

    # Highest score
    highest_score = bat_p["runs"].max()

    # 50s and 100s
    centuries = (bat_p["runs"] >= 100).sum()
    half_centuries = ((bat_p["runs"] >= 50) & (bat_p["runs"] < 100)).sum()

    # Top opponent by runs
    opp_runs = (
        bat_p.groupby("opponent")["runs"]
        .sum()
        .sort_values(ascending=False)
    )

    top_opponent = opp_runs.index[0] if not opp_runs.empty else "N/A"

    # Display metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", matches_played)
    c2.metric("Innings", innings_played)
    c3.metric("Total Runs", total_runs)
    c4.metric("Strike Rate", round(overall_sr, 2))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Highest Score", highest_score)
    c6.metric("Centuries (100s)", centuries)
    c7.metric("Half-Centuries (50s)", half_centuries)
    c8.metric("Top Opponent", top_opponent)


    # =====================================================
    # CURRENT TEAM & LAST MATCH PERFORMANCE
    # =====================================================
    st.divider()
    st.header("Current Team & Last Match Performance")

    latest_season = bat_p["season"].max()
    current_team = bat_p[bat_p["season"] == latest_season]["batting_team"].mode()[0]

    last_row = bat_p.sort_values(["season", "match_id"], ascending=False).iloc[0]
    last_match_id = last_row["match_id"]
    last_opponent = last_row["opponent"]

    last_del = del_p[del_p["match_id"] == last_match_id]
    last_runs = last_del["batsman_runs"].sum()
    last_balls = len(last_del)
    last_sr = (last_runs / last_balls) * 100 if last_balls else 0
    last_fours = (last_del["batsman_runs"] == 4).sum()
    last_sixes = (last_del["batsman_runs"] == 6).sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Team", current_team)
    c2.metric("Opponent", last_opponent)
    c3.metric("Runs", last_runs)
    c4.metric("Balls", last_balls)
    c5.metric("Strike Rate", round(last_sr, 2))
    c6.metric("4s / 6s", f"{last_fours} / {last_sixes}")

    # =====================================================
    # 🔮 BATSMAN PERFORMANCE PREDICTION
    # =====================================================
    st.divider()
    st.header("🔮 Batting Performance Prediction")

    venue_pred = st.selectbox(
        "Select Venue (Prediction)",
        sorted(bat_p["venue"].dropna().unique())
    )

    opponent = st.selectbox(
        "Select Opponent Team",
        sorted([t for t in bat_df["opponent"].unique() if t != current_team])
    )

    inning = st.selectbox("Select Innings", [1, 2])

    recent_5 = bat_p.sort_values(["season", "match_id"], ascending=False).head(5)
    avg_runs_last5 = recent_5["runs"].mean()
    avg_sr_last5 = recent_5["strike_rate"].mean()

    venue_matches = bat_p[bat_p["venue"] == venue_pred].sort_values(
        ["season", "match_id"], ascending=False
    ).head(5)
    avg_runs_venue = venue_matches["runs"].mean() if not venue_matches.empty else avg_runs_last5

    opp_matches = bat_p[bat_p["opponent"] == opponent].sort_values(
        ["season", "match_id"], ascending=False
    ).head(5)
    avg_runs_opponent = opp_matches["runs"].mean() if not opp_matches.empty else avg_runs_last5

    st.subheader("Model Input Features")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Avg Runs (Last 5)", round(avg_runs_last5, 2))
    f2.metric("Avg SR (Last 5)", round(avg_sr_last5, 2))
    f3.metric("Avg Runs @ Venue", round(avg_runs_venue, 2))
    f4.metric("Avg Runs vs Opponent", round(avg_runs_opponent, 2))
    f5.metric("Innings", inning)

    if st.button("Predict Runs"):
        with st.spinner("Analyzing player data and predicting performance..."):
            import time
            time.sleep(1)  # Simulate processing
            X = pd.DataFrame([{
                "avg_runs_last5": avg_runs_last5,
                "avg_sr_last5": avg_sr_last5,
                "avg_runs_venue_last5": avg_runs_venue,
                "avg_runs_opponent_last5": avg_runs_opponent,
                "inning": inning
            }])
            pred = model.predict(X)[0]
            label = "TOP" if pred >= 40 else "AVERAGE" if pred >= 20 else "LOW"
            st.success(f"Predicted Runs: {pred:.1f} ({label})")
            # Add confetti effect
            st.balloons()

    # =====================================================
    # 🆕 OPPONENT-WISE PLAYER PERFORMANCE ANALYSIS
    # =====================================================
    st.divider()
    st.header("Opponent-wise Player Performance Analysis")

    opp_select = st.selectbox(
        "Select Opponent (Analysis)",
        sorted(bat_p["opponent"].unique())
    )

    venue_filter = st.selectbox(
        "Filter by Venue (Optional)",
        ["All Venues"] + sorted(bat_p["venue"].dropna().unique())
    )

    opp_data = bat_p[bat_p["opponent"] == opp_select]
    if venue_filter != "All Venues":
        opp_data = opp_data[opp_data["venue"] == venue_filter]

    opp_match_ids = opp_data["match_id"].unique()
    opp_del = del_p[del_p["match_id"].isin(opp_match_ids)]

    total_runs = opp_del["batsman_runs"].sum()
    total_balls = len(opp_del)
    total_fours = (opp_del["batsman_runs"] == 4).sum()
    total_sixes = (opp_del["batsman_runs"] == 6).sum()
    total_sr = (total_runs / total_balls) * 100 if total_balls else 0
    boundary_pct = ((total_fours * 4 + total_sixes * 6) / total_runs * 100) if total_runs else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Runs vs Opponent", total_runs)
    c2.metric("Balls", total_balls)
    c3.metric("Strike Rate", round(total_sr, 2))
    c4.metric("Fours", total_fours)
    c5.metric("Sixes", total_sixes)
    c6.metric("Boundary %", round(boundary_pct, 2))

    last5_opp = opp_data.sort_values(
        ["season", "match_id"], ascending=False
    ).head(5)

    st.subheader("Last 5 Matches vs Opponent")
    st.dataframe(
        last5_opp[["season", "venue", "runs", "balls", "strike_rate"]],
        use_container_width=True
    )

    st.subheader("Runs Trend vs Opponent")
    fig = px.bar(last5_opp, x="match_id", y="runs", color="runs", color_continuous_scale="Viridis", animation_frame="season" if "season" in last5_opp.columns else None)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 🧑 PLAYER INDIVIDUAL ANALYTICS DASHBOARD
    # =====================================================
    st.divider()
    st.header("Player Individual Analytics Dashboard")
    # =====================================================
    # 📈 CONSISTENCY INDEX — BATSMAN (Last 10 Matches)
    # =====================================================
    st.divider()
    st.header("Consistency Index (Batting – Last 10 Matches)")

    last10_bat = bat_p.sort_values(
        ["season", "match_id"], ascending=False
    ).head(10)

    run_std = last10_bat["runs"].std()

    if pd.isna(run_std):
        run_std = 0

    # Interpretation
    if run_std <= 10:
        consistency_label = "Very Consistent"
    elif run_std <= 20:
        consistency_label = "Moderately Consistent"
    else:
        consistency_label = "Inconsistent"

    c1, c2 = st.columns(2)
    c1.metric("Run Std Dev (σ)", round(run_std, 2))
    c2.metric("Consistency Level", consistency_label)

    st.subheader("Runs in Last 10 Matches")
    fig = px.line(last10_bat, x="match_id", y="runs", markers=True, color_discrete_sequence=["#FFD700"])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    fig.update_traces(mode="lines+markers", marker=dict(size=10, color="#00FF00"))
    st.plotly_chart(fig, use_container_width=True)
    # =====================================================
    # 🏟️ HOME vs AWAY — BATSMAN
    # =====================================================
    st.divider()
    st.header("Home vs Away Consistency (Batting)")

    home_venue = bat_p["venue"].mode()[0]

    home_bat = bat_p[bat_p["venue"] == home_venue]
    away_bat = bat_p[bat_p["venue"] != home_venue]

    home_std = home_bat["runs"].std()
    away_std = away_bat["runs"].std()

    c1, c2 = st.columns(2)
    c1.metric("Home Run Std Dev", round(home_std, 2))
    c2.metric("Away Run Std Dev", round(away_std, 2))


    # =====================================================
    # 🔥 PRESSURE PERFORMANCE — DEATH OVERS (BATSMAN)
    # =====================================================
    st.divider()
    st.header("Pressure Performance (Death Overs: 16–20)")

    death_del = del_p[del_p["over"] >= 16]

    death_runs = death_del["batsman_runs"].sum()
    death_balls = len(death_del)
    death_sr = (death_runs / death_balls) * 100 if death_balls else 0

    death_fours = (death_del["batsman_runs"] == 4).sum()
    death_sixes = (death_del["batsman_runs"] == 6).sum()

    total_runs = del_p["batsman_runs"].sum()
    death_run_pct = (death_runs / total_runs * 100) if total_runs else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Runs (Death)", death_runs)
    c2.metric("Balls", death_balls)
    c3.metric("Strike Rate", round(death_sr, 2))
    c4.metric("4s / 6s", f"{death_fours} / {death_sixes}")
    c5.metric("Run % in Death Overs", f"{death_run_pct:.1f}%")

    st.subheader("Death Overs Runs Trend (Last 10 Matches)")
    death_recent = (
        del_p[del_p["over"] >= 16]
        .groupby("match_id")["batsman_runs"]
        .sum()
        .sort_index()
        .tail(10)
    )
    fig = px.area(death_recent.reset_index(), x="match_id", y="batsman_runs", color_discrete_sequence=["#FF6B6B"])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)


    season_stats = del_p.groupby("season").agg(
        runs=("batsman_runs", "sum"),
        balls=("ball", "count"),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum())
    ).reset_index()

    season_stats["strike_rate"] = season_stats["runs"] / season_stats["balls"] * 100
    st.dataframe(season_stats, use_container_width=True)
    # =====================================================
    # 💥 IMPACT SCORE — BATSMAN
    # =====================================================
    st.divider()
    st.header("Impact Score (Batting)")

    career_avg_runs = bat_p["runs"].mean()
    career_avg_sr = bat_p["strike_rate"].mean()

    batting_impact = (0.7 * career_avg_runs) + (0.3 * (career_avg_sr / 10))

    st.metric(
        "Batting Impact Score",
        round(batting_impact, 2)
    )


    last_3 = sorted(del_p["season"].unique())[-3:]
    phase_3 = del_p[del_p["season"].isin(last_3)].groupby("phase").agg(
        runs=("batsman_runs", "sum"),
        balls=("ball", "count"),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum())
    )

    phase_3["strike_rate"] = phase_3["runs"] / phase_3["balls"] * 100
    phase_3["boundary_percentage"] = (
        (phase_3["fours"] * 4 + phase_3["sixes"] * 6) / phase_3["runs"] * 100
    )

    st.subheader("Phase-wise Batting Analysis (Last 3 Seasons)")
    st.dataframe(phase_3.round(2), use_container_width=True)
    # =====================================================
    # 🧠 CLUTCH INDEX — BATSMAN (Death Overs)
    # =====================================================
    st.divider()
    st.header("Clutch Performance (Death Overs)")

    death_bat = del_p[
        (del_p["phase"] == "Death")
    ]

    death_runs = death_bat["batsman_runs"].sum()
    death_balls = len(death_bat)
    death_sr = (death_runs / death_balls) * 100 if death_balls else 0

    clutch_index_bat = (death_runs / max(1, death_balls)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Death Runs", death_runs)
    c2.metric("Death SR", round(death_sr, 2))
    c3.metric("Clutch Index", round(clutch_index_bat, 2))


    # =====================================================
    # 📊 VISUAL DASHBOARDS
    # =====================================================
    st.divider()
    st.header("Visual Performance Dashboards")

    st.subheader("Runs per Season")
    fig = px.bar(season_stats, x="season", y="runs", color="runs", color_continuous_scale="Blues")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strike Rate per Season")
    fig = px.line(season_stats, x="season", y="strike_rate", markers=True, color_discrete_sequence=["#FFD700"])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phase-wise Runs")
    fig = px.pie(phase_3.reset_index(), values="runs", names="phase", color_discrete_sequence=px.colors.sequential.Rainbow)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boundary Comparison (4s vs 6s)")
    boundary_df = pd.DataFrame({
        "Count": [
            (del_p["batsman_runs"] == 4).sum(),
            (del_p["batsman_runs"] == 6).sum()
        ]
    }, index=["Fours", "Sixes"])
    fig = px.bar(boundary_df, x=boundary_df.index, y="Count", color=boundary_df.index, color_discrete_map={"Fours": "#4ECDC4", "Sixes": "#FF6B6B"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Form (Last 10 Matches)")
    recent_10 = bat_p.sort_values(["season", "match_id"], ascending=False).head(10)
    fig = px.scatter(recent_10, x="match_id", y="runs", size="runs", color="runs", color_continuous_scale="Plasma", animation_frame="season")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("BATSMAN SAVEPOINT • IPL 2016–2025")

# =====================================================
# ===================== BOWLING MODE ==================
# =====================================================
elif role == "Bowling":

    # =====================================================
    # PATHS
    # =====================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    BOWL_PATH = os.path.join(
        BASE_DIR, "data", "processed", "bowling_with_context.csv"
    )

    WICKET_MODEL_PATH = os.path.join(
        BASE_DIR, "models", "bowling_wicket_model.pkl"
    )

    ECON_MODEL_PATH = os.path.join(
        BASE_DIR, "models", "bowling_economy_model.pkl"
    )

    # =====================================================
    # LOAD DATA & MODELS
    # =====================================================
    bowl_df = pd.read_csv(BOWL_PATH)
    wicket_model = joblib.load(WICKET_MODEL_PATH)
    economy_model = joblib.load(ECON_MODEL_PATH)

    # =====================================================
    # TEAM NAME NORMALIZATION
    # =====================================================
    TEAM_NAME_MAP = {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiants": "Rising Pune Supergiant"
    }

    for col in ["bowling_team", "opponent"]:
        if col in bowl_df.columns:
            bowl_df[col] = bowl_df[col].replace(TEAM_NAME_MAP)

    # =====================================================
    # BOWLER SELECTION
    # =====================================================
    bowler = st.selectbox(
        "Select Bowler",
        sorted(bowl_df["bowler"].unique())
    )

    bowl_p = bowl_df[bowl_df["bowler"] == bowler]
    # =====================================================
    # 🎯 PLAYER OVERALL BOWLING PERFORMANCE (CAREER SUMMARY)
    # =====================================================
    st.divider()
    st.header("Bowler Overall Career Performance (2016–2025)")

    # Matches & innings
    matches_played = bowl_p["match_id"].nunique()
    innings_bowled = bowl_p.shape[0]

    # Total wickets & overs
    total_wickets = bowl_p["wickets"].sum()
    total_overs = bowl_p["overs"].sum()

    # Career economy
    career_economy = (
        bowl_p["runs_conceded"].sum() / total_overs
        if total_overs > 0 else 0
    )

    # Best bowling figures (wickets-runs in a match)
    best_row = bowl_p.sort_values(
        ["wickets", "runs_conceded"],
        ascending=[False, True]
    ).iloc[0]

    best_figures = f"{int(best_row['wickets'])}-{int(best_row['runs_conceded'])}"

    # Top opponent by wickets
    opp_wkts = (
        bowl_p.groupby("opponent")["wickets"]
        .sum()
        .sort_values(ascending=False)
    )

    top_opponent = opp_wkts.index[0] if not opp_wkts.empty else "N/A"

    # Display metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", matches_played)
    c2.metric("Innings", innings_bowled)
    c3.metric("Total Wickets", total_wickets)
    c4.metric("Economy", round(career_economy, 2))

    c5, c6 = st.columns(2)
    c5.metric("Best Bowling", best_figures)
    c6.metric("Top Opponent", top_opponent)


    # =====================================================
    # CURRENT TEAM & LAST MATCH PERFORMANCE (BOWLING)
    # =====================================================
    st.divider()
    st.header("Current Team & Last Match Performance (Bowling)")

    latest_season = bowl_p["season"].max()
    current_team = bowl_p[bowl_p["season"] == latest_season]["bowling_team"].mode()[0]

    last_match = bowl_p.sort_values(
        ["season", "match_id"], ascending=False
    ).iloc[0]

    last_match_id = last_match["match_id"]
    last_opponent = last_match["opponent"]

    last_overs = last_match["overs"]
    last_wickets = last_match["wickets"]
    last_economy = last_match["economy"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Team", current_team)
    c2.metric("Opponent", last_opponent)
    c3.metric("Overs", round(last_overs, 1))
    c4.metric("Wickets", last_wickets)
    c5.metric("Economy", round(last_economy, 2))

    # =====================================================
    # 🎯 BOWLING PERFORMANCE PREDICTION
    # =====================================================
    st.divider()
    st.header("🎯 Bowling Performance Prediction")

    venue = st.selectbox(
        "Select Venue",
        sorted(bowl_p["venue"].dropna().unique())
    )

    opponent = st.selectbox(
        "Select Opponent Team",
        sorted(bowl_p["opponent"].dropna().unique())
    )

    innings = st.selectbox("Select Innings", [1, 2])

    recent_5 = bowl_p.sort_values(
        ["season", "match_id"], ascending=False
    ).head(5)

    avg_wkts = recent_5["wickets"].mean()
    avg_eco = recent_5["economy"].mean()
    avg_overs = recent_5["overs"].mean()

    venue_data = bowl_p[bowl_p["venue"] == venue].sort_values(
        ["season", "match_id"], ascending=False
    ).head(5)

    avg_wkts_venue = venue_data["wickets"].mean() if not venue_data.empty else avg_wkts
    avg_eco_venue = venue_data["economy"].mean() if not venue_data.empty else avg_eco

    st.subheader("Model Input Features")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Wkts (Last 5)", round(avg_wkts, 2))
    c2.metric("Avg Eco (Last 5)", round(avg_eco, 2))
    c3.metric("Avg Overs (Last 5)", round(avg_overs, 2))
    c4.metric("Avg Wkts @ Venue", round(avg_wkts_venue, 2))
    c5.metric("Avg Eco @ Venue", round(avg_eco_venue, 2))

    if st.button("Predict Bowling Performance"):
        X = pd.DataFrame([{
            "avg_wickets_last5": avg_wkts,
            "avg_economy_last5": avg_eco,
            "avg_overs_last5": avg_overs,
            "avg_wickets_venue_last5": avg_wkts_venue,
            "avg_economy_venue_last5": avg_eco_venue
        }])

        pred_wkts = wicket_model.predict(X)[0]
        pred_eco = economy_model.predict(X)[0]

        label = (
            "TOP BOWLER"
            if pred_wkts >= 2 and pred_eco <= 7
            else "AVERAGE"
            if pred_wkts >= 1
            else "LOW IMPACT"
        )

        st.success(
            f"Predicted Wickets: {pred_wkts:.2f} | "
            f"Economy: {pred_eco:.2f} | {label}"
        )
        # =====================================================
    # PHASE-WISE BOWLING & ROLE CLASSIFICATION
    # =====================================================
    st.divider()
    st.header("Phase-wise Bowling Analysis & Role")

    phase_summary = bowl_p.copy()

    phase_summary["phase"] = pd.cut(
        phase_summary["economy"],
        bins=[0, 7, 8.5, 20],
        labels=["Powerplay Impact", "Middle Overs", "Death Overs"]
    )

    phase_stats = phase_summary.groupby("phase").agg(
        overs=("overs", "sum"),
        wickets=("wickets", "sum"),
        economy=("economy", "mean")
    )

    st.subheader("Phase-wise Bowling Stats")
    st.dataframe(phase_stats.round(2), use_container_width=True)
    # =====================================================
    # 🧠 CLUTCH INDEX — BOWLER (Death Overs)
    # =====================================================
    st.divider()
    st.header("Clutch Performance (Death Overs)")

    death_bowl = bowl_p[bowl_p["economy"] >= 8]  # proxy for death pressure

    death_wkts = death_bowl["wickets"].sum()
    death_overs = death_bowl["overs"].sum()
    death_eco = death_bowl["economy"].mean()

    clutch_index_bowl = (death_wkts / max(1, death_overs))

    c1, c2, c3 = st.columns(3)
    c1.metric("Death Wickets", death_wkts)
    c2.metric("Death Economy", round(death_eco, 2))
    c3.metric("Clutch Index", round(clutch_index_bowl, 2))


    role_label = "Support Bowler"
    if "Death Overs" in phase_stats.index and phase_stats.loc["Death Overs", "overs"] >= phase_stats["overs"].sum() * 0.3:
        role_label = "Death Bowler"
    elif "Powerplay Impact" in phase_stats.index and phase_stats.loc["Powerplay Impact", "overs"] >= phase_stats["overs"].sum() * 0.3:
        role_label = "Powerplay Bowler"
    elif "Middle Overs" in phase_stats.index and phase_stats.loc["Middle Overs", "overs"] >= phase_stats["overs"].sum() * 0.4:
        role_label = "Middle Overs Controller"

    st.success(f"🧠 Bowler Role Classification: **{role_label}**")

    st.subheader("Phase-wise Wickets")
    st.bar_chart(phase_stats["wickets"])

    st.divider()
   

    # =====================================================
    # OPPONENT-WISE BOWLING PERFORMANCE
    # =====================================================
    st.divider()
    # =====================================================
    # OPPONENT-WISE BOWLING PERFORMANCE
    # =====================================================
    st.divider()
    st.header("Opponent-wise Bowling Performance Analysis")

    opp_select = st.selectbox(
        "Select Opponent (Bowling Analysis)",
        sorted(bowl_p["opponent"].dropna().unique())
    )

    venue_filter = st.selectbox(
        "Select Venue (Optional)",
        ["All Venues"] + sorted(bowl_p["venue"].dropna().unique())
    )

    opp_data = bowl_p[bowl_p["opponent"] == opp_select]

    if venue_filter != "All Venues":
        opp_data = opp_data[opp_data["venue"] == venue_filter]


    total_wickets = opp_data["wickets"].sum()
    total_overs = opp_data["overs"].sum()
    avg_economy = opp_data["economy"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Wickets vs Opponent", total_wickets)
    c2.metric("Overs Bowled", round(total_overs, 1))
    c3.metric("Avg Economy", round(avg_economy, 2))

    last5_opp = opp_data.sort_values(
        ["season", "match_id"], ascending=False
    ).head(5)

    st.subheader("Last 5 Matches vs Opponent")
    st.dataframe(
        last5_opp[["season", "venue", "overs", "wickets", "economy"]],
        use_container_width=True
    )

    st.subheader("Wickets Trend vs Opponent")
    st.bar_chart(last5_opp.set_index("match_id")["wickets"])

    # =====================================================
    # PLAYER INDIVIDUAL BOWLING ANALYTICS
    # =====================================================
    st.divider()
    st.header("Player Individual Bowling Analytics Dashboard")
    # =====================================================
    # 📈 CONSISTENCY INDEX — BOWLER (Last 10 Matches)
    # =====================================================
    st.divider()
    st.header("Consistency Index (Bowling – Last 10 Matches)")

    last10_bowl = bowl_p.sort_values(
        ["season", "match_id"], ascending=False
    ).head(10)

    wicket_std = last10_bowl["wickets"].std()
    eco_std = last10_bowl["economy"].std()

    if pd.isna(wicket_std):
        wicket_std = 0
    if pd.isna(eco_std):
        eco_std = 0

    # Interpretation
    if wicket_std <= 0.5 and eco_std <= 1:
        consistency_label = "Highly Consistent"
    elif wicket_std <= 1.0 and eco_std <= 2:
        consistency_label = "Moderately Consistent"
    else:
        consistency_label = "Inconsistent"

    c1, c2, c3 = st.columns(3)
    c1.metric("Wicket Std Dev (σ)", round(wicket_std, 2))
    c2.metric("Economy Std Dev (σ)", round(eco_std, 2))
    c3.metric("Consistency Level", consistency_label)

    st.subheader("Last 10 Matches – Wickets Trend")
    st.bar_chart(last10_bowl.set_index("match_id")["wickets"])

    st.subheader("Last 10 Matches – Economy Trend")
    st.line_chart(last10_bowl.set_index("match_id")["economy"])
    # =====================================================
    # 🏟️ HOME vs AWAY — BOWLER
    # =====================================================
    st.divider()
    st.header("Home vs Away Consistency (Bowling)")

    home_venue = bowl_p["venue"].mode()[0]

    home_bowl = bowl_p[bowl_p["venue"] == home_venue]
    away_bowl = bowl_p[bowl_p["venue"] != home_venue]

    home_eco_std = home_bowl["economy"].std()
    away_eco_std = away_bowl["economy"].std()

    c1, c2 = st.columns(2)
    c1.metric("Home Economy Std Dev", round(home_eco_std, 2))
    c2.metric("Away Economy Std Dev", round(away_eco_std, 2))


    # =====================================================
    # 🔥 PRESSURE PERFORMANCE — DEATH OVERS (BOWLER)
    # =====================================================
    st.divider()
    st.header("Pressure Performance (Death Overs Approximation)")

    # Approximation: high-pressure spells assumed where economy >= 8.5
    death_bowling = bowl_p[bowl_p["economy"] >= 8.5]

    death_overs = death_bowling["overs"].sum()
    death_runs = death_bowling["runs_conceded"].sum()
    death_wkts = death_bowling["wickets"].sum()

    death_eco = (death_runs / death_overs) if death_overs else 0
    total_wkts = bowl_p["wickets"].sum()
    death_wkt_pct = (death_wkts / total_wkts * 100) if total_wkts else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overs (Death)", round(death_overs, 1))
    c2.metric("Runs Conceded", int(death_runs))
    c3.metric("Wickets", death_wkts)
    c4.metric("Economy", round(death_eco, 2))

    st.metric("Wicket % in Death Overs", f"{death_wkt_pct:.1f}%")

    st.subheader("Death Overs Wickets Trend (Last 10 Matches)")
    death_recent = (
        death_bowling
        .sort_values(["season", "match_id"])
        .groupby("match_id")["wickets"]
        .sum()
        .tail(10)
    )
    st.bar_chart(death_recent)


    season_stats = bowl_p.groupby("season").agg(
        wickets=("wickets", "sum"),
        overs=("overs", "sum"),
        economy=("economy", "mean")
    )

    st.subheader("Season-wise Bowling Performance")
    st.dataframe(season_stats.round(2), use_container_width=True)
    # =====================================================
    # 💥 IMPACT SCORE — BOWLER
    # =====================================================
    st.divider()
    st.header("Impact Score (Bowling)")

    career_avg_wkts = bowl_p["wickets"].mean()
    career_avg_eco = bowl_p["economy"].mean()

    bowling_impact = (0.7 * career_avg_wkts) - (0.3 * career_avg_eco)

    st.metric(
        "Bowling Impact Score",
        round(bowling_impact, 2)
    )


    st.subheader("Season-wise Wickets")
    st.bar_chart(season_stats["wickets"])

    st.subheader("Season-wise Economy")
    st.line_chart(season_stats["economy"])
    st.caption("BOWLING MODE • Venue-aware Prediction & Analytics • IPL 2016–2025")
