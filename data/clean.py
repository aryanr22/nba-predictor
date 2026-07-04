"""
data/clean.py
-------------
Cleans raw CSVs from fetch.py into analysis-ready DataFrames.
"""

import os
import pandas as pd
import numpy as np

RAW_DIR  = os.path.join(os.path.dirname(__file__), "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def _save(df, filename):
    path = os.path.join(PROC_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df):,} rows → {filename}")


def _load_raw(filename):
    return pd.read_csv(os.path.join(RAW_DIR, filename), low_memory=False)


def clean_team_logs():
    print("\n[1/3] Cleaning team game logs...")
    df = _load_raw("team_game_logs.csv")
    df.columns = df.columns.str.upper()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    df["IS_HOME"] = df["MATCHUP"].str.contains(r" vs\.", regex=True).astype(int)
    df["OPP_ABBREVIATION"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+(\w+)")
    df["WIN"] = (df["WL"] == "W").astype(int)

    df["REST_DAYS"] = (
        df.groupby("TEAM_ID")["GAME_DATE"]
        .diff().dt.days.fillna(3).clip(upper=10)
    )
    df["IS_B2B"] = (df["REST_DAYS"] == 1).astype(int)

    keep = [
        "SEASON", "GAME_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "IS_HOME", "OPP_ABBREVIATION", "WIN", "W", "L",
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
        "REST_DAYS", "IS_B2B",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    _save(df, "team_game_logs_clean.csv")
    return df


def clean_player_logs():
    print("\n[2/3] Cleaning player game logs...")
    df = _load_raw("player_game_logs.csv")
    df.columns = df.columns.str.upper()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    df["IS_HOME"] = df["MATCHUP"].str.contains(r" vs\.", regex=True).astype(int)
    df["OPP_ABBREVIATION"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+(\w+)")
    df["DNP"] = df["MIN"].isna() | (df["MIN"] == 0)

    def parse_minutes(m):
        if pd.isna(m):
            return 0.0
        m = str(m)
        if ":" in m:
            parts = m.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        try:
            return float(m)
        except ValueError:
            return 0.0

    df["MIN"] = df["MIN"].apply(parse_minutes)

    today = pd.Timestamp.today()
    df["DAYS_AGO"] = (today - df["GAME_DATE"]).dt.days
    df["RECENCY_WEIGHT"] = np.exp(-df["DAYS_AGO"] / 180)

    keep = [
        "SEASON", "GAME_TYPE", "GAME_ID", "GAME_DATE",
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "IS_HOME", "OPP_ABBREVIATION", "MIN", "DNP",
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
        "DAYS_AGO", "RECENCY_WEIGHT",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df = df.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])

    df_played = df[~df["DNP"]].copy()
    _save(df_played, "player_game_logs_clean.csv")
    _save(df, "player_game_logs_all.csv")
    return df_played


def build_game_table(team_df):
    print("\n[3/3] Building combined game table...")
    home = team_df[team_df["IS_HOME"] == 1].copy()
    away = team_df[team_df["IS_HOME"] == 0].copy()

    home = home.add_prefix("HOME_").rename(columns={
        "HOME_GAME_ID": "GAME_ID", "HOME_GAME_DATE": "GAME_DATE",
        "HOME_SEASON": "SEASON", "HOME_GAME_TYPE": "GAME_TYPE"
    })
    away = away.add_prefix("AWAY_").rename(columns={
        "AWAY_GAME_ID": "GAME_ID", "AWAY_GAME_DATE": "GAME_DATE",
        "AWAY_SEASON": "SEASON", "AWAY_GAME_TYPE": "GAME_TYPE"
    })

    games = home.merge(away, on=["GAME_ID", "GAME_DATE", "SEASON", "GAME_TYPE"], how="inner")
    games["TOTAL_PTS"] = games["HOME_PTS"] + games["AWAY_PTS"]
    games["SPREAD"]    = games["HOME_PTS"] - games["AWAY_PTS"]

    _save(games, "games_combined.csv")
    print(f"  ✓ {len(games):,} total games")
    return games


def run_cleaning():
    team_df   = clean_team_logs()
    player_df = clean_player_logs()
    game_df   = build_game_table(team_df)
    print("\n✅ Cleaning complete.")
    return team_df, player_df, game_df


if __name__ == "__main__":
    run_cleaning()
