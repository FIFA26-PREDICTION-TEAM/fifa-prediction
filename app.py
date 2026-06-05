"""
FIFA Match Predictor — Streamlit app.
Entry point: streamlit run app.py
"""

import os
import json
import streamlit as st

st.set_page_config(
    page_title="FIFA Match Predictor",
    page_icon="⚽",
    layout="centered",
)

from data.ingest import load_all, get_all_teams, count_loaded
from model.features import FEATURE_COLUMNS

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "artifacts")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")


# ── Model load-or-train (cached resource — runs once per session) ────────────

@st.cache_resource(show_spinner=False)
def get_model_and_meta():
    """Load model artifacts, training inline if not present."""
    from model.train import train_and_save

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        st.info("No trained model found. Training now — this takes about 30 seconds...")
        meta = train_and_save(verbose=False)
    else:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
        else:
            meta = {"model_name": "Unknown", "accuracy": 0.0, "training_rows": 0, "features": len(FEATURE_COLUMNS)}

    import joblib
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    return model, scaler, meta


# ── Data loading ──────────────────────────────────────────────────────────────

with st.spinner("Loading match data..."):
    data = load_all()

matches_df = data["matches"]
goalscorers_df = data["goalscorers"]
shootouts_df = data["shootouts"]
rankings_df           = data["rankings"]
substitutions_df      = data.get("substitutions")
player_appearances_df = data.get("player_appearances")
player_goals_df       = data.get("player_goals")
award_winners_df      = data.get("award_winners")
copa_data             = data.get("copa_america")

loaded_count = count_loaded(data)

if matches_df is None:
    st.error(
        "**matches.csv is required to run this app.**\n\n"
        "Drop it (and optionally goalscorers.csv, shootouts.csv, rankings.csv) "
        "into the project root directory, then refresh the page."
    )
    st.stop()

# ── Model ─────────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    model_obj, scaler_obj, meta = get_model_and_meta()

accuracy = meta.get("accuracy", 0.0)
training_rows = meta.get("training_rows", 0)
model_name = meta.get("model_name", "Unknown")

# ── Title + status ────────────────────────────────────────────────────────────

n_matches = len(matches_df)
st.title("⚽ FIFA Match Predictor")
st.caption(
    f"Powered by {n_matches:,} historical international matches · "
    f"Model: {model_name} · Accuracy: {accuracy:.1%} · "
    f"Trained on {training_rows:,} rows · {loaded_count}/4 data sources loaded"
)

if accuracy < 0.55 and accuracy > 0:
    st.sidebar.warning(
        "Model accuracy is below 55%. Consider retraining with a narrower date range."
    )

# ── Input section ─────────────────────────────────────────────────────────────

st.divider()

all_teams = get_all_teams(matches_df)

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
with col2:
    team_b = st.selectbox("Team B", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)

predict_btn = st.button("Predict", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:
    if team_a == team_b:
        st.error("Please select two different teams.")
        st.stop()

    with st.spinner(f"Analysing {n_matches:,} historical matches..."):
        from model.predict import predict_match
        result = predict_match(
            team_a=team_a,
            team_b=team_b,
            matches_df=matches_df,
            goalscorers_df=goalscorers_df,
            shootouts_df=shootouts_df,
            rankings_df=rankings_df,
            substitutions_df=substitutions_df,
            player_appearances_df=player_appearances_df,
            player_goals_df=player_goals_df,
            award_winners_df=award_winners_df,
            copa_data=copa_data,
        )

    win_prob = result["win_prob"]
    draw_prob = result["draw_prob"]
    loss_prob = result["loss_prob"]
    conf_score = result["confidence_score"]
    conf_label = result["confidence_label"]

    st.divider()

    # ── Match header ──────────────────────────────────────────────────────────
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
        "⚽ FIFA World Cup</p>",
        unsafe_allow_html=True,
    )
    hcol1, hcol2, hcol3 = st.columns([2, 1, 2])
    with hcol1:
        rank_a = result["team_a_rank"]
        st.markdown(
            f"<h3 style='text-align:center'>{team_a}</h3>"
            f"<p style='text-align:center; color:#888'>Rank #{rank_a if rank_a else 'N/A'}</p>",
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown("<h3 style='text-align:center; color:#aaa'>vs</h3>", unsafe_allow_html=True)
    with hcol3:
        rank_b = result["team_b_rank"]
        st.markdown(
            f"<h3 style='text-align:center'>{team_b}</h3>"
            f"<p style='text-align:center; color:#888'>Rank #{rank_b if rank_b else 'N/A'}</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Probabilities ─────────────────────────────────────────────────────────
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.metric(label=f"{team_a} Win", value=f"{win_prob:.1%}")
        st.progress(win_prob)
    with pc2:
        st.metric(label="Draw", value=f"{draw_prob:.1%}")
        st.progress(draw_prob)
    with pc3:
        st.metric(label=f"{team_b} Win", value=f"{loss_prob:.1%}")
        st.progress(loss_prob)

    st.divider()

    # ── Confidence ────────────────────────────────────────────────────────────
    n_used = result["n_features_used"]
    n_total = result["n_features_total"]
    st.markdown(
        f"**Confidence: {conf_score:.0%} — {conf_label}** &nbsp;&nbsp; "
        f"<span style='color:#888; font-size:0.85rem'>{n_used} of {n_total} parameters used</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Key factors ───────────────────────────────────────────────────────────
    st.subheader("Key deciding factors")
    for factor in result["key_factors"]:
        st.markdown(f"- {factor}")

    st.divider()

    # ── Historical stats ──────────────────────────────────────────────────────
    st.subheader("Historical statistics")
    hs_col1, hs_col2 = st.columns(2)
    with hs_col1:
        st.metric(f"{team_a} Goals/game (WC)", f"{result['team_a_avg_goals_wc']:.2f}")
        st.metric(f"{team_a} WC Titles", result["team_a_wc_titles"])
        st.metric(
            f"{team_a} Penalty shootouts",
            f"{result['team_a_penalty_shootout_wins']}W / {result['team_a_penalty_shootout_losses']}L",
        )
    with hs_col2:
        st.metric(f"{team_b} Goals/game (WC)", f"{result['team_b_avg_goals_wc']:.2f}")
        st.metric(f"{team_b} WC Titles", result["team_b_wc_titles"])
        st.metric(
            f"{team_b} Penalty shootouts",
            f"{result['team_b_penalty_shootout_wins']}W / {result['team_b_penalty_shootout_losses']}L",
        )
    cs_col1, cs_col2 = st.columns(2)
    with cs_col1:
        st.metric(f"{team_a} Clean sheet rate", f"{result['team_a_clean_sheet_rate']:.1%}")
    with cs_col2:
        st.metric(f"{team_b} Clean sheet rate", f"{result['team_b_clean_sheet_rate']:.1%}")

    st.divider()

    # ── Super-sub alerts ──────────────────────────────────────────────────────
    ss_a = result.get("supersub_a")
    ss_b = result.get("supersub_b")
    if ss_a or ss_b:
        for ss in [ss_a, ss_b]:
            if ss:
                st.warning(
                    f"⚡ **Super-sub alert — {ss['name']}, {ss['team']}**\n\n"
                    f"{ss['goals_as_sub']} goals in {ss['sub_appearances']} substitute appearances "
                    f"({ss['goal_rate']:.2f} goals/app)"
                )

    # ── Model verdict ─────────────────────────────────────────────────────────
    st.subheader("Model verdict")
    st.info(result["verdict"])

    st.divider()

    # ── Key players ───────────────────────────────────────────────────────────
    kp_a = result.get("key_players_a", [])
    kp_b = result.get("key_players_b", [])
    if kp_a or kp_b:
        st.subheader("Key players")
        kp_col1, kp_col2 = st.columns(2)

        AWARD_BADGE = {
            "Golden Ball":  "🏆 Golden Ball",
            "Golden Boot":  "👟 Golden Boot",
            "Golden Glove": "🧤 Golden Glove",
        }

        def _render_players(players: list, team_name: str, col):
            with col:
                st.markdown(f"**{team_name}**")
                if not players:
                    st.caption("No player data available.")
                    return
                for p in players:
                    badges = " ".join(AWARD_BADGE.get(a, a) for a in p["awards"])
                    pos_str = f" · {p['position']}" if p["position"] else ""
                    goals_str = (
                        f"{p['goals']}G / {p['appearances']} apps"
                        f" ({p['goal_rate']:.2f} g/app)"
                    )
                    st.markdown(
                        f"**{p['name']}**{pos_str}  \n"
                        f"<span style='color:#888; font-size:0.85rem'>{goals_str}</span>"
                        + (f"  \n{badges}" if badges else ""),
                        unsafe_allow_html=True,
                    )

        _render_players(kp_a, team_a, kp_col1)
        _render_players(kp_b, team_b, kp_col2)
