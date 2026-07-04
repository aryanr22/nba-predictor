"""
features/engineer.py
--------------------
Builds all model-ready features from cleaned data.
All rolling windows use shift(1) — no same-game leakage ever.
"""

import os
import pandas as pd
import numpy as np

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
os.makedirs(FEAT_DIR, exist_ok=True)

WINDOWS = [5, 10, 20]

TEAM_STAT_COLS = [
    "PTS", "FGA", "FG_PCT", "FG3A", "FG3_PCT",
    "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]

PLAYER_STAT_COLS = [
    "MIN", "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
]


def _save(df, filename):
    path = os.path.join(FEAT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df):,} rows → {filename}")


def build_team_features(team_df):
    print("\n[1/4] Building team rolling features...")
    team_df = team_df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    rows = []

    for team_id, grp in team_df.groupby("TEAM_ID"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        feat = grp[["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
                     "IS_HOME", "OPP_ABBREVIATION", "REST_DAYS", "IS_B2B",
                     "WIN", "PTS", "SEASON", "GAME_TYPE"]].copy()

        for col in TEAM_STAT_COLS:
            if col not in grp.columns:
                continue
            for w in WINDOWS:
                feat[f"{col}_ROLL{w}"] = (
                    grp[col].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
                )

        for w in WINDOWS:
            feat[f"WIN_RATE_ROLL{w}"] = (
                grp["WIN"].shift(1).rolling(w, min_periods=1).mean()
            )

        grp["OFF_RTG_PROXY"] = grp["PTS"] / grp["FGA"].replace(0, np.nan)
        feat["OFF_RTG_PROXY_ROLL10"] = (
            grp["OFF_RTG_PROXY"].shift(1).rolling(10, min_periods=3).mean()
        )

        feat["SEASON_GAME_NUM"] = range(len(feat))
        feat["IS_PLAYOFF"] = (grp["GAME_TYPE"] == "Playoffs").astype(int)
        rows.append(feat)

    result = pd.concat(rows, ignore_index=True)
    _save(result, "team_features.csv")
    return result


def build_defensive_features(team_df):
    print("\n[2/4] Building defensive features...")
    games = pd.read_csv(os.path.join(PROC_DIR, "games_combined.csv"),
                        parse_dates=["GAME_DATE"])

    # FIX: column is SEASON not HOME_SEASON
    home_def = games[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_PTS", "SEASON"]].copy()
    home_def.columns = ["GAME_ID", "GAME_DATE", "TEAM_ID", "PTS_ALLOWED", "SEASON"]

    away_def = games[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ID", "HOME_PTS", "SEASON"]].copy()
    away_def.columns = ["GAME_ID", "GAME_DATE", "TEAM_ID", "PTS_ALLOWED", "SEASON"]

    def_df = pd.concat([home_def, away_def], ignore_index=True)
    def_df = def_df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    rows = []
    for team_id, grp in def_df.groupby("TEAM_ID"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        feat = grp[["GAME_ID", "TEAM_ID"]].copy()
        for w in WINDOWS:
            feat[f"PTS_ALLOWED_ROLL{w}"] = (
                grp["PTS_ALLOWED"].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
            )
        rows.append(feat)

    result = pd.concat(rows, ignore_index=True)
    _save(result, "defensive_features.csv")
    return result


def build_player_features(player_df):
    print("\n[3/4] Building player rolling features...")
    player_df = player_df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    rows = []

    for player_id, grp in player_df.groupby("PLAYER_ID"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        feat = grp[[
            "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
            "TEAM_ID", "TEAM_ABBREVIATION", "IS_HOME", "OPP_ABBREVIATION",
            "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
            "FG3M", "SEASON", "GAME_TYPE", "RECENCY_WEIGHT",
        ]].copy()

        feat["IS_PLAYOFF"] = (grp["GAME_TYPE"] == "Playoffs").astype(int)

        for col in PLAYER_STAT_COLS:
            if col not in grp.columns:
                continue
            for w in WINDOWS:
                feat[f"{col}_ROLL{w}"] = (
                    grp[col].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
                )

        if all(c in grp.columns for c in ["FGA", "FTA", "TOV", "MIN"]):
            usage_raw = (
                grp["FGA"] + 0.44 * grp["FTA"] + grp["TOV"]
            ) / grp["MIN"].replace(0, np.nan)
            for w in WINDOWS:
                feat[f"USAGE_ROLL{w}"] = (
                    usage_raw.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
                )

        for w in [10, 20]:
            feat[f"GAMES_PLAYED_LAST{w}"] = (
                grp["MIN"].shift(1).rolling(w, min_periods=1).count()
            )

        feat["PTS_SEASON_AVG"] = grp["PTS"].expanding().mean().shift(1)
        feat["MIN_SEASON_AVG"] = grp["MIN"].expanding().mean().shift(1)

        rows.append(feat)

    result = pd.concat(rows, ignore_index=True)
    _save(result, "player_features.csv")
    return result


def build_game_model_features():
    """
    Join game table + team rolling features + defensive features.
    FIX: only merges rolling columns to avoid duplicate raw stat conflicts.
    """
    print("\n[4/4] Assembling game model feature matrix...")

    games  = pd.read_csv(os.path.join(PROC_DIR, "games_combined.csv"), parse_dates=["GAME_DATE"])
    t_feat = pd.read_csv(os.path.join(FEAT_DIR, "team_features.csv"),  parse_dates=["GAME_DATE"])
    d_feat = pd.read_csv(os.path.join(FEAT_DIR, "defensive_features.csv"))

    # Only keep engineered rolling columns — avoids conflicts with raw stats in games_combined
    roll_cols = [c for c in t_feat.columns if any(
        c.endswith(s) for s in
        ["_ROLL5", "_ROLL10", "_ROLL20", "_PROXY_ROLL10",
         "WIN_RATE_ROLL10", "REST_DAYS", "IS_B2B", "IS_PLAYOFF", "SEASON_GAME_NUM"]
    )]
    id_cols = ["GAME_ID", "GAME_DATE", "TEAM_ID"]
    t_feat_slim = t_feat[[c for c in id_cols + roll_cols if c in t_feat.columns]].copy()

    # Tag each team feature row as home or away for that game
    home_ids = games.set_index("GAME_ID")["HOME_TEAM_ID"].to_dict()
    away_ids = games.set_index("GAME_ID")["AWAY_TEAM_ID"].to_dict()

    t_feat_slim["SIDE"] = t_feat_slim.apply(
        lambda r: "HOME" if home_ids.get(r["GAME_ID"]) == r["TEAM_ID"]
        else ("AWAY" if away_ids.get(r["GAME_ID"]) == r["TEAM_ID"] else None),
        axis=1
    )

    home_feat = (t_feat_slim[t_feat_slim["SIDE"] == "HOME"]
                 .drop(columns=["SIDE", "TEAM_ID"])
                 .add_prefix("HOME_")
                 .rename(columns={"HOME_GAME_ID": "GAME_ID", "HOME_GAME_DATE": "GAME_DATE"}))

    away_feat = (t_feat_slim[t_feat_slim["SIDE"] == "AWAY"]
                 .drop(columns=["SIDE", "TEAM_ID"])
                 .add_prefix("AWAY_")
                 .rename(columns={"AWAY_GAME_ID": "GAME_ID", "AWAY_GAME_DATE": "GAME_DATE"}))

    # Defensive features
    def_roll_cols = [c for c in d_feat.columns if "PTS_ALLOWED" in c]
    d_slim = d_feat[["GAME_ID", "TEAM_ID"] + def_roll_cols]

    home_def = (d_slim.merge(
        games[["GAME_ID", "HOME_TEAM_ID"]].rename(columns={"HOME_TEAM_ID": "TEAM_ID"}),
        on=["GAME_ID", "TEAM_ID"]
    ).drop(columns=["TEAM_ID"])
     .add_prefix("HOME_OPP_DEF_")
     .rename(columns={"HOME_OPP_DEF_GAME_ID": "GAME_ID"}))

    away_def = (d_slim.merge(
        games[["GAME_ID", "AWAY_TEAM_ID"]].rename(columns={"AWAY_TEAM_ID": "TEAM_ID"}),
        on=["GAME_ID", "TEAM_ID"]
    ).drop(columns=["TEAM_ID"])
     .add_prefix("AWAY_OPP_DEF_")
     .rename(columns={"AWAY_OPP_DEF_GAME_ID": "GAME_ID"}))

    matrix = (games
        .merge(home_feat, on=["GAME_ID", "GAME_DATE"], how="left")
        .merge(away_feat, on=["GAME_ID", "GAME_DATE"], how="left")
        .merge(home_def,  on="GAME_ID", how="left")
        .merge(away_def,  on="GAME_ID", how="left"))

    matrix["TARGET_HOME_PTS"]  = matrix["HOME_PTS"]
    matrix["TARGET_AWAY_PTS"]  = matrix["AWAY_PTS"]
    matrix["TARGET_TOTAL_PTS"] = matrix["TOTAL_PTS"]
    matrix["TARGET_SPREAD"]    = matrix["SPREAD"]

    roll_check = [c for c in ["HOME_PTS_ROLL10", "AWAY_PTS_ROLL10"] if c in matrix.columns]
    if roll_check:
        matrix = matrix.dropna(subset=roll_check)

    _save(matrix, "game_model_features.csv")
    print(f"  ✓ Shape: {matrix.shape}")
    return matrix


def run_feature_engineering():
    print("=" * 55)
    print("NBA Predictor — Feature Engineering")
    print("=" * 55)

    team_df   = pd.read_csv(os.path.join(PROC_DIR, "team_game_logs_clean.csv"),   parse_dates=["GAME_DATE"])
    player_df = pd.read_csv(os.path.join(PROC_DIR, "player_game_logs_clean.csv"), parse_dates=["GAME_DATE"])

    build_team_features(team_df)
    build_defensive_features(team_df)
    build_player_features(player_df)
    build_game_model_features()

    print("\n✅ Feature engineering complete.")


if __name__ == "__main__":
    run_feature_engineering()
