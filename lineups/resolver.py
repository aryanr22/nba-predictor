"""
lineups/resolver.py
-------------------
Resolves expected lineups for tonight's games.
Handles injury designations, usage redistribution, and gametime overrides.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")

STATUS_OUT          = "Out"
STATUS_DOUBTFUL     = "Doubtful"
STATUS_QUESTIONABLE = "Questionable"
STATUS_PROBABLE     = "Probable"
STATUS_AVAILABLE    = "Available"

PLAY_PROBABILITY = {
    STATUS_OUT:          0.00,
    STATUS_DOUBTFUL:     0.15,
    STATUS_QUESTIONABLE: 0.50,
    STATUS_PROBABLE:     0.85,
    STATUS_AVAILABLE:    1.00,
}


def load_rosters():
    return pd.read_csv(os.path.join(RAW_DIR, "rosters.csv"), low_memory=False)


def load_player_logs():
    return pd.read_csv(
        os.path.join(PROC_DIR, "player_game_logs_all.csv"),
        parse_dates=["GAME_DATE"], low_memory=False
    )


def load_injury_report():
    path = os.path.join(RAW_DIR, "injury_report.csv")
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def build_depth_chart(team_abbr, player_logs, n_games=15):
    recent = (
        player_logs[
            (player_logs["TEAM_ABBREVIATION"] == team_abbr)
            & (player_logs["SEASON"] == "2025-26")
        ]
        .sort_values("GAME_DATE", ascending=False)
        .groupby("PLAYER_ID")
        .head(n_games)
    )
    depth = (
        recent.groupby(["PLAYER_ID", "PLAYER_NAME"])
        .agg(AVG_MIN=("MIN", "mean"), GAMES_PLAYED=("MIN", "count"), AVG_PTS=("PTS", "mean"))
        .reset_index()
        .sort_values("AVG_MIN", ascending=False)
    )
    depth["DEPTH_RANK"] = range(1, len(depth) + 1)
    return depth


def get_player_status(player_name, injury_df):
    if injury_df.empty or "PLAYER_NAME" not in injury_df.columns:
        return STATUS_AVAILABLE
    matches = injury_df[injury_df["PLAYER_NAME"].str.lower() == player_name.lower()]
    if matches.empty:
        return STATUS_AVAILABLE
    status_text = str(matches.iloc[0].get("STATUS", "")).lower()
    if "out" in status_text:
        return STATUS_OUT
    if "doubtful" in status_text:
        return STATUS_DOUBTFUL
    if "questionable" in status_text or "day-to-day" in status_text or "day to day" in status_text:
        return STATUS_QUESTIONABLE
    if "probable" in status_text:
        return STATUS_PROBABLE
    return STATUS_AVAILABLE


def redistribute_minutes(depth_chart, out_player_ids, injury_df):
    dc = depth_chart.copy()
    dc["STATUS"] = dc["PLAYER_NAME"].apply(lambda n: get_player_status(n, injury_df))
    dc.loc[dc["PLAYER_ID"].isin(out_player_ids), "STATUS"] = STATUS_OUT

    available  = dc[dc["STATUS"] != STATUS_OUT].copy()
    out_players = dc[dc["STATUS"] == STATUS_OUT].copy()

    if out_players.empty:
        dc["PROJECTED_MIN"] = dc["AVG_MIN"]
        dc["PLAY_PROB"] = dc["STATUS"].map(PLAY_PROBABILITY).fillna(1.0)
        return dc

    mins_to_redistribute = out_players["AVG_MIN"].sum()
    total_avail_min = available["AVG_MIN"].sum()

    if total_avail_min > 0:
        available["EXTRA_MIN"] = (
            available["AVG_MIN"] / total_avail_min * mins_to_redistribute
        )
        available["PROJECTED_MIN"] = (available["AVG_MIN"] + available["EXTRA_MIN"]).clip(upper=40)
    else:
        available["PROJECTED_MIN"] = available["AVG_MIN"]

    out_players["PROJECTED_MIN"] = 0.0
    result = pd.concat([available, out_players], ignore_index=True)
    result["PLAY_PROB"] = result["STATUS"].map(PLAY_PROBABILITY).fillna(1.0)
    result["PROJECTED_MIN"] = result["PROJECTED_MIN"] * result["PLAY_PROB"]

    return result.sort_values("PROJECTED_MIN", ascending=False).reset_index(drop=True)


def resolve_lineup(team_abbr, manual_overrides=None):
    player_logs = load_player_logs()
    injury_df   = load_injury_report()
    depth = build_depth_chart(team_abbr, player_logs)

    out_ids = []
    if manual_overrides:
        for player_name, status in manual_overrides.items():
            mask = depth["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
            if status == STATUS_OUT:
                out_ids.extend(depth.loc[mask, "PLAYER_ID"].tolist())

    result = redistribute_minutes(depth, out_ids, injury_df)
    return result[result["PROJECTED_MIN"] > 0].head(12).copy()


def resolve_all_tonight():
    path = os.path.join(RAW_DIR, "todays_games.csv")
    if not os.path.exists(path):
        print("No today's games file — run fetch.py first")
        return {}

    games = pd.read_csv(path)
    lineups = {}
    teams_tonight = set()

    for _, row in games.iterrows():
        for col in ["HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]:
            if col in row and pd.notna(row[col]):
                teams_tonight.add(row[col])

    print(f"Resolving lineups for {len(teams_tonight)} teams...")
    for team in sorted(teams_tonight):
        try:
            lineup = resolve_lineup(team)
            lineups[team] = lineup
            print(f"  ✓ {team}: {len(lineup)} players")
        except Exception as e:
            print(f"  ✗ {team}: {e}")

    return lineups


def compute_conditional_lineups(team_abbr, player_name):
    return {
        "with_player":    resolve_lineup(team_abbr),
        "without_player": resolve_lineup(team_abbr, manual_overrides={player_name: STATUS_OUT}),
        "player_name":    player_name,
    }
