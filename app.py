"""
app.py
------
Streamlit UI — four tabs:
  1. Tonight's Games
  2. Player Projections
  3. Custom Matchup (simulation)
  4. Ask Claude

Run with: python -m streamlit run app.py
"""

import os
import sys
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

PRED_DIR     = os.path.join(os.path.dirname(__file__), "data", "predictions")
CLAUDE_API   = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-5"

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 NBA Predictor")
    st.caption(f"Updated: {datetime.now().strftime('%b %d, %Y')}")
    page = st.radio("Navigate", [
        "🏟️  Tonight's Games",
        "📊  Player Projections",
        "🧪  Custom Matchup",
        "💬  Ask Claude",
    ])
    st.divider()
    claude_api_key = st.text_input(
        "Anthropic API Key", type="password",
        help="Only needed for Ask Claude. Get one free at console.anthropic.com"
    )
    st.divider()
    st.caption("Model: XGBoost + walk-forward CV")
    st.caption("⚠️ For informational use only.")


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_predictions():
    path = os.path.join(PRED_DIR, "latest_predictions.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_player_projections():
    path = os.path.join(PRED_DIR, "latest_player_projections.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Player table renderer ─────────────────────────────────────────────────────
def render_player_table(df):
    if df.empty:
        st.info("No player projection data available.")
        return
    display_cols = ["player_name", "team", "status", "play_prob",
                    "proj_min", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]
    display_cols = [c for c in display_cols if c in df.columns]
    rename = {
        "player_name": "Player", "team": "Team", "status": "Status",
        "play_prob": "Play%", "proj_min": "Min", "PTS": "Pts",
        "REB": "Reb", "AST": "Ast", "FG3M": "3PM",
        "STL": "Stl", "BLK": "Blk", "TOV": "TO",
    }
    display = df[display_cols].rename(columns=rename).copy()
    if "Play%" in display.columns:
        display["Play%"] = (display["Play%"] * 100).round(0).astype(str) + "%"
    st.dataframe(display, use_container_width=True, hide_index=True)


# ── Page 1: Tonight's Games ───────────────────────────────────────────────────
def render_games_page():
    st.title("🏟️ Tonight's Games")
    predictions = load_predictions()

    if not predictions:
        st.warning("No predictions found. Run `python run.py --daily` first.")
        return

    cols = st.columns(min(len(predictions), 3))
    for i, game in enumerate(predictions):
        home, away = game.get("home_team", ""), game.get("away_team", "")
        h_pts, a_pts = game.get("home_score", "—"), game.get("away_score", "—")
        total = game.get("total_pts", "—")
        spread = game.get("spread", 0)
        spread_str = f"{home} -{abs(spread):.1f}" if spread > 0 else f"{away} -{abs(spread):.1f}"

        with cols[i % 3]:
            with st.container(border=True):
                st.subheader(f"{away} @ {home}")
                st.metric("Projected Score", f"{away} {a_pts}  —  {home} {h_pts}")
                c1, c2 = st.columns(2)
                c1.metric("Spread", spread_str)
                c2.metric("O/U", f"{total}")
                if st.button("View players →", key=f"game_{i}"):
                    st.session_state["selected_game"] = f"{away} @ {home}"
                    st.rerun()

    if "selected_game" in st.session_state:
        st.divider()
        gf = st.session_state["selected_game"]
        st.subheader(f"Player Projections — {gf}")
        player_df = load_player_projections()
        if not player_df.empty and "game" in player_df.columns:
            render_player_table(player_df[player_df["game"] == gf])


# ── Page 2: Player Projections ────────────────────────────────────────────────
def render_players_page():
    st.title("📊 Player Projections")
    player_df = load_player_projections()

    if player_df.empty:
        st.warning("No player projections found. Run `python run.py --daily` first.")
        return

    c1, c2, c3 = st.columns(3)
    games = ["All"] + sorted(player_df["game"].unique().tolist()) if "game" in player_df.columns else ["All"]
    teams = ["All"] + sorted(player_df["team"].unique().tolist()) if "team" in player_df.columns else ["All"]

    selected_game = c1.selectbox("Game", games)
    selected_team = c2.selectbox("Team", teams)
    search        = c3.text_input("Search player", placeholder="e.g. LeBron")

    filtered = player_df.copy()
    if selected_game != "All":
        filtered = filtered[filtered["game"] == selected_game]
    if selected_team != "All":
        filtered = filtered[filtered["team"] == selected_team]
    if search:
        filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(filtered)} players")
    render_player_table(filtered)


# ── Page 3: Custom Matchup (simulation) ───────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_teams():
    from nba_api.stats.static import teams as nba_teams
    return sorted(nba_teams.get_teams(), key=lambda t: t["full_name"])


@st.cache_data(ttl=300)
def cached_lineup(team_abbr, out_players):
    from lineups.resolver import resolve_lineup
    overrides = {name: "Out" for name in out_players} if out_players else None
    return resolve_lineup(team_abbr, manual_overrides=overrides)


def render_matchup_page():
    st.title("🧪 Custom Matchup Simulator")
    st.caption(
        "Simulate any two teams with the trained models and each team's recent form. "
        "This is a hypothetical simulation, not a prediction of a real scheduled game."
    )

    try:
        all_teams = load_all_teams()
    except Exception as e:
        st.error(f"Couldn't load team list: {e}")
        return

    team_names = {t["abbreviation"]: t["full_name"] for t in all_teams}
    abbrs = list(team_names.keys())

    c1, c2 = st.columns(2)
    home_abbr = c1.selectbox("Home team", abbrs, format_func=lambda a: team_names[a], index=0)
    away_abbr = c2.selectbox("Away team", abbrs, format_func=lambda a: team_names[a], index=1)

    if home_abbr == away_abbr:
        st.warning("Pick two different teams.")
        return

    st.divider()
    st.subheader("Optional: adjust lineups")
    st.caption("Mark a player OUT to simulate them missing the game — their minutes are "
               "redistributed across the rest of the roster.")

    lc1, lc2 = st.columns(2)
    home_out, away_out = [], []

    with lc1:
        st.markdown(f"**{team_names[home_abbr]}**")
        try:
            home_baseline = cached_lineup(home_abbr, ())
            home_out = st.multiselect(
                "Mark OUT", home_baseline["PLAYER_NAME"].tolist(), key=f"home_out_{home_abbr}"
            )
        except Exception:
            st.info("No roster data yet — run `python run.py --setup` first.")

    with lc2:
        st.markdown(f"**{team_names[away_abbr]}**")
        try:
            away_baseline = cached_lineup(away_abbr, ())
            away_out = st.multiselect(
                "Mark OUT", away_baseline["PLAYER_NAME"].tolist(), key=f"away_out_{away_abbr}"
            )
        except Exception:
            st.info("No roster data yet — run `python run.py --setup` first.")

    st.divider()

    if st.button("Run Simulation", type="primary"):
        from pipeline import predict_game_full
        with st.spinner("Simulating matchup..."):
            try:
                home_lineup = cached_lineup(home_abbr, tuple(sorted(home_out)))
                away_lineup = cached_lineup(away_abbr, tuple(sorted(away_out)))
                st.session_state["matchup_result"] = predict_game_full(
                    home_abbr, away_abbr, home_lineup, away_lineup
                )
            except Exception as e:
                st.error(f"Simulation failed: {e}. Have you run `python run.py --setup`?")
                return

    result = st.session_state.get("matchup_result")
    if not result:
        return

    if result.get("error"):
        st.error(result["error"])
        return

    st.info("🧪 Simulated matchup — based on current models and recent team form, "
            "not tonight's actual schedule or confirmed lineups.")

    home, away = result["home_team"], result["away_team"]
    st.subheader(f"{away} @ {home}  (simulated)")
    st.metric("Projected Score", f"{away} {result['away_score']}  —  {home} {result['home_score']}")

    spread = result.get("spread", 0)
    spread_str = f"{home} -{abs(spread):.1f}" if spread > 0 else f"{away} -{abs(spread):.1f}"
    sc1, sc2 = st.columns(2)
    sc1.metric("Spread", spread_str)
    sc2.metric("O/U", f"{result.get('total_pts', '—')}")

    st.divider()
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**{home} Player Projections**")
        render_player_table(pd.DataFrame(result.get("home_players", [])))
    with pc2:
        st.markdown(f"**{away} Player Projections**")
        render_player_table(pd.DataFrame(result.get("away_players", [])))


# ── Page 4: Ask Claude ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an NBA analyst assistant with access to tonight's model-generated
game and player projections. Ground every answer in the provided context.
Acknowledge uncertainty — these are statistical projections, not guarantees.
Keep responses concise and analytical. Never give definitive financial advice."""


def build_context(player_df, predictions):
    parts = ["=== TONIGHT'S GAMES ===\n"]
    for game in predictions:
        home, away = game.get("home_team"), game.get("away_team")
        spread = game.get("spread", 0)
        spread_str = f"{home} by {abs(spread):.1f}" if spread > 0 else f"{away} by {abs(spread):.1f}"
        parts.append(
            f"{away} @ {home}: {away} {game.get('away_score')} — {home} {game.get('home_score')} "
            f"| Total: {game.get('total_pts')} | Spread: {spread_str}\n"
        )
    if not player_df.empty:
        parts.append("\n=== PLAYER PROJECTIONS ===\n")
        cols = [c for c in ["player_name", "team", "status", "proj_min",
                             "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"] if c in player_df.columns]
        parts.append(player_df[cols].sort_values("PTS", ascending=False).head(60).to_string(index=False))
    return "\n".join(parts)


def call_claude(user_message, context, api_key, history):
    if not api_key:
        return "⚠️ Enter your Anthropic API key in the sidebar to use this feature."

    messages = []
    if not history:
        messages += [
            {"role": "user",      "content": f"Tonight's projections:\n\n{context}"},
            {"role": "assistant", "content": "Got it — I have tonight's projections loaded. What would you like to know?"},
        ]
    messages += history
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(CLAUDE_API, headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }, json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"Error: {e}"


def render_claude_page():
    st.title("💬 Ask Claude")
    st.caption("Ask anything about tonight's games and projections.")

    player_df   = load_player_projections()
    predictions = load_predictions()

    if not predictions:
        st.warning("No predictions loaded. Run `python run.py --daily` first.")
        return

    context = build_context(player_df, predictions)

    examples = [
        "Who are the top scorers projected tonight?",
        "Which game has the closest spread?",
        "Best rebounders on the slate tonight?",
    ]
    c1, c2, c3 = st.columns(3)
    for col, ex in zip([c1, c2, c3], examples):
        if col.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "claude_messages" not in st.session_state:
        st.session_state.claude_messages = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prefill = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask about tonight...") or (prefill or None)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_claude(user_input, context, claude_api_key,
                                       st.session_state.claude_messages)
            st.markdown(response)

        st.session_state.chat_history += [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ]
        st.session_state.claude_messages += [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ]

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.claude_messages = []
            st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
if "Tonight" in page:
    render_games_page()
elif "Player" in page:
    render_players_page()
elif "Matchup" in page:
    render_matchup_page()
elif "Claude" in page:
    render_claude_page()
