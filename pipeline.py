"""
pipeline.py
-----------
Daily prediction runner. Called by run.py.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from data.fetch import run_daily_refresh
from lineups.resolver import resolve_all_tonight
from models.game_model import load_game_models, prepare_X, FEATURE_COLS, MODEL_DIR
from models.player_model import load_player_models, predict_player_game

PRED_DIR = os.path.join(os.path.dirname(__file__), "data", "predictions")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "data", "features")
RAW_DIR  = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(PRED_DIR, exist_ok=True)


def get_team_features(team_abbr):
    df = pd.read_csv(os.path.join(FEAT_DIR, "team_features.csv"), parse_dates=["GAME_DATE"])
    team_df = df[df["TEAM_ABBREVIATION"] == team_abbr].sort_values("GAME_DATE")
    if team_df.empty:
        return {}
    return team_df.iloc[-1].to_dict()


def get_player_features(player_id, projected_min, game_total, is_home):
    df = pd.read_csv(os.path.join(FEAT_DIR, "player_features.csv"), parse_dates=["GAME_DATE"])
    player_df = df[df["PLAYER_ID"] == player_id].sort_values("GAME_DATE")
    if player_df.empty:
        return {}
    latest = player_df.iloc[-1].to_dict()
    latest["PROJECTED_MIN"]        = projected_min
    latest["PROJECTED_GAME_TOTAL"] = game_total
    latest["IS_HOME"]              = is_home
    return latest


def predict_game_full(home_abbr, away_abbr, home_lineup, away_lineup):
    game_models = load_game_models()

    feat_path = os.path.join(MODEL_DIR, "game_model_features.json")
    if not os.path.exists(feat_path):
        return {"error": "Game model not trained yet — run python run.py --setup first"}

    with open(feat_path) as f:
        game_feat_cols = json.load(f)

    home_feats = get_team_features(home_abbr)
    away_feats = get_team_features(away_abbr)

    if not home_feats or not away_feats:
        return {"error": f"Missing features for {home_abbr} or {away_abbr}"}

    row = {}
    for k, v in home_feats.items():
        row[f"HOME_{k}"] = v
    for k, v in away_feats.items():
        row[f"AWAY_{k}"] = v

    X = pd.DataFrame([row])
    X = prepare_X(X, game_feat_cols)

    game_preds = {}
    for name, model in game_models.items():
        try:
            game_preds[name] = float(model.predict(X)[0])
        except Exception:
            game_preds[name] = None

    home_score = round(game_preds["home_pts"] if game_preds.get("home_pts") is not None else 110)
    away_score = round(game_preds["away_pts"] if game_preds.get("away_pts") is not None else 107)
    total_pts  = game_preds["total_pts"] if game_preds.get("total_pts") is not None else home_score + away_score
    spread     = game_preds["spread"] if game_preds.get("spread") is not None else home_score - away_score

    player_projections = {"home": [], "away": []}

    for side, lineup, is_home in [("home", home_lineup, 1), ("away", away_lineup, 0)]:
        if lineup is None or len(lineup) == 0:
            continue
        for _, player_row in lineup.iterrows():
            player_id   = player_row.get("PLAYER_ID")
            player_name = player_row.get("PLAYER_NAME", "Unknown")
            proj_min    = player_row.get("PROJECTED_MIN", 0)
            play_prob   = player_row.get("PLAY_PROB", 1.0)

            if proj_min < 1 or play_prob < 0.05:
                continue

            pf = get_player_features(player_id, proj_min, total_pts, is_home)
            if not pf:
                continue

            stats = predict_player_game(pf)
            stats = {k: round(v * play_prob, 1) for k, v in stats.items()}

            player_projections[side].append({
                "player_name": player_name,
                "team":        home_abbr if is_home else away_abbr,
                "status":      player_row.get("STATUS", "Available"),
                "play_prob":   round(play_prob, 2),
                "proj_min":    round(proj_min, 1),
                **stats,
            })

    return {
        "home_team":    home_abbr,
        "away_team":    away_abbr,
        "home_score":   home_score,
        "away_score":   away_score,
        "total_pts":    round(total_pts, 1),
        "spread":       round(spread, 1),
        "home_players": sorted(player_projections["home"], key=lambda x: -x.get("PTS", 0)),
        "away_players": sorted(player_projections["away"], key=lambda x: -x.get("PTS", 0)),
        "generated_at": datetime.now().isoformat(),
    }


def run_tonight(final_run=False):
    run_type = "FINAL (90-min lock)" if final_run else "MORNING"
    print("=" * 55)
    print(f"NBA Predictor — Tonight's Predictions [{run_type}]")
    print("=" * 55)

    print("\n[1] Refreshing data...")
    run_daily_refresh()

    print("\n[2] Resolving lineups...")
    lineups = resolve_all_tonight()

    games_path = os.path.join(RAW_DIR, "todays_games.csv")
    if not os.path.exists(games_path):
        print("No games file found.")
        return []

    games = pd.read_csv(games_path)
    if games.empty:
        print("No games scheduled tonight.")
        return []

    print(f"\n[3] Predicting {len(games)} games...")

    all_predictions = []
    all_player_rows = []

    for _, game in games.iterrows():
        home = game.get("HOME_TEAM_ABBREVIATION", "")
        away = game.get("VISITOR_TEAM_ABBREVIATION", "")
        if not home or not away:
            continue

        print(f"  {away} @ {home}...")
        try:
            result = predict_game_full(
                home, away,
                lineups.get(home, pd.DataFrame()),
                lineups.get(away, pd.DataFrame()),
            )
            all_predictions.append(result)
            for player in result.get("home_players", []) + result.get("away_players", []):
                all_player_rows.append({"game": f"{away} @ {home}", **player})
            print(f"    → {away} {result['away_score']} @ {home} {result['home_score']} "
                  f"| Total: {result['total_pts']} | Spread: {result['spread']:+.1f}")
        except Exception as e:
            print(f"    ✗ {e}")

    suffix = "_final" if final_run else "_morning"
    today  = datetime.now().strftime("%Y-%m-%d")

    json_path = os.path.join(PRED_DIR, f"predictions_{today}{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(all_predictions, f, indent=2)

    with open(os.path.join(PRED_DIR, "latest_predictions.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)

    if all_player_rows:
        csv_path = os.path.join(PRED_DIR, f"player_projections_{today}{suffix}.csv")
        pd.DataFrame(all_player_rows).to_csv(csv_path, index=False)
        pd.DataFrame(all_player_rows).to_csv(
            os.path.join(PRED_DIR, "latest_player_projections.csv"), index=False)

    print(f"\n✅ Done — {len(all_predictions)} games predicted.")
    return all_predictions
