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

STATUS_OUT           = "Out"
STATUS_DOUBTFUL      = "Doubtful"
STATUS_QUESTIONABLE  = "Questionable"
STATUS_PROBABLE      = "Probable"
STATUS_AVAILABLE     = "Available"
STATUS_LIKELY_INJURED = "Likely Injured"

PLAY_PROBABILITY = {
    STATUS_OUT:           0.00,
    STATUS_DOUBTFUL:      0.15,
    STATUS_QUESTIONABLE:  0.50,
    STATUS_PROBABLE:      0.85,
    STATUS_AVAILABLE:     1.00,
    STATUS_LIKELY_INJURED: 0.00,
}

INACTIVITY_DAYS = 30


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


def build_depth_chart(team_abbr, player_logs, n_games=15, recency_days=INACTIVITY_DAYS):
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
    )

    roster = load_rosters()
    team_roster = roster[roster["TEAM_ABBREVIATION"] == team_abbr]
    roster_ids = set(team_roster["PLAYER_ID"])
    roster_names = team_roster.set_index("PLAYER_ID")["PLAYER"].to_dict()

    # Current-roster players always belong on the depth chart, even if they
    # haven't logged a game for this team yet this season (new signing, rookie,
    # long-term injury) — add placeholder rows for anyone missing.
    missing_ids = roster_ids - set(depth["PLAYER_ID"])
    if missing_ids:
        filler = pd.DataFrame([{
            "PLAYER_ID": pid, "PLAYER_NAME": roster_names.get(pid, "Unknown"),
            "AVG_MIN": 0.0, "GAMES_PLAYED": 0, "AVG_PTS": 0.0,
        } for pid in missing_ids])
        depth = pd.concat([depth, filler], ignore_index=True)

    last_played = player_logs.groupby("PLAYER_ID")["GAME_DATE"].max()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=recency_days)
    depth["LAST_PLAYED"] = depth["PLAYER_ID"].map(last_played)
    depth["ON_ROSTER"] = depth["PLAYER_ID"].isin(roster_ids)
    is_stale = depth["LAST_PLAYED"].isna() | (depth["LAST_PLAYED"] < cutoff)

    # Off-roster + stale = traded away and inactive elsewhere: drop entirely.
    depth = depth[depth["ON_ROSTER"] | ~is_stale].reset_index(drop=True)

    # On-roster + stale = likely injured: keep visible and flagged, defaulting
    # to 0 projected minutes downstream unless the user overrides it.
    is_stale = depth["LAST_PLAYED"].isna() | (depth["LAST_PLAYED"] < cutoff)
    depth["STATUS_FLAG"] = None
    depth.loc[depth["ON_ROSTER"] & is_stale, "STATUS_FLAG"] = STATUS_LIKELY_INJURED

    depth = depth.drop(columns=["LAST_PLAYED", "ON_ROSTER"])
    depth = depth.sort_values("AVG_MIN", ascending=False).reset_index(drop=True)
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


def redistribute_minutes(depth_chart, injury_df, status_overrides=None):
    dc = depth_chart.copy()

    # Default to the likely-injured flag from build_depth_chart, then let a real
    # injury-report entry take precedence over it where one exists.
    if "STATUS_FLAG" in dc.columns:
        dc["STATUS"] = dc["STATUS_FLAG"].fillna(STATUS_AVAILABLE)
    else:
        dc["STATUS"] = STATUS_AVAILABLE

    report_status = dc["PLAYER_NAME"].apply(lambda n: get_player_status(n, injury_df))
    reported = report_status != STATUS_AVAILABLE
    dc.loc[reported, "STATUS"] = report_status[reported]

    if status_overrides:
        for player_id, status in status_overrides.items():
            dc.loc[dc["PLAYER_ID"] == player_id, "STATUS"] = status

    available  = dc[dc["STATUS"] != STATUS_OUT].copy()
    out_players = dc[dc["STATUS"] == STATUS_OUT].copy()

    if out_players.empty:
        dc["PLAY_PROB"] = dc["STATUS"].map(PLAY_PROBABILITY).fillna(1.0)
        dc["PROJECTED_MIN"] = dc["AVG_MIN"] * dc["PLAY_PROB"]
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


def resolve_lineup(team_abbr, manual_overrides=None, ignore_injury_report=False):
    player_logs = load_player_logs()
    injury_df   = pd.DataFrame() if ignore_injury_report else load_injury_report()
    depth = build_depth_chart(team_abbr, player_logs)

    status_overrides = {}
    if manual_overrides:
        for player_name, status in manual_overrides.items():
            mask = depth["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
            for player_id in depth.loc[mask, "PLAYER_ID"].tolist():
                status_overrides[player_id] = status

    result = redistribute_minutes(depth, injury_df, status_overrides)
    active = result[result["PROJECTED_MIN"] > 0].head(12)
    likely_injured = result[result["STATUS"] == STATUS_LIKELY_INJURED]
    combined = pd.concat([active, likely_injured], ignore_index=True)
    return combined.drop_duplicates(subset="PLAYER_ID", keep="first").copy()


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
