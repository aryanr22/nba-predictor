"""
data/fetch.py
-------------
Pulls game logs, player logs, rosters, today's games, and injury report from nba_api.
Run once historically, then daily.
"""

import os
import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import (
    LeagueGameLog, CommonTeamRoster, ScoreboardV2,
)
from nba_api.stats.static import teams

SEASONS = ["2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = "2025-26"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)
API_DELAY = 0.7


def _save(df, filename):
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  ✓ {len(df):,} rows → {filename}")


def _sleep():
    time.sleep(API_DELAY)


def fetch_team_game_logs():
    print("\n[1/4] Fetching team game logs...")
    frames = []
    for season in SEASONS:
        for game_type in ["Regular Season", "Playoffs"]:
            _sleep()
            try:
                log = LeagueGameLog(
                    season=season,
                    season_type_all_star=game_type,
                    player_or_team_abbreviation="T",
                )
                df = log.get_data_frames()[0]
                df["SEASON"] = season
                df["GAME_TYPE"] = game_type
                frames.append(df)
                print(f"  {season} {game_type}: {len(df)} games")
            except Exception as e:
                print(f"  ✗ {season} {game_type}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    _save(combined, "team_game_logs.csv")
    return combined


def fetch_player_game_logs():
    print("\n[2/4] Fetching player game logs...")
    frames = []
    for season in SEASONS:
        for game_type in ["Regular Season", "Playoffs"]:
            _sleep()
            try:
                log = LeagueGameLog(
                    season=season,
                    season_type_all_star=game_type,
                    player_or_team_abbreviation="P",
                )
                df = log.get_data_frames()[0]
                df["SEASON"] = season
                df["GAME_TYPE"] = game_type
                frames.append(df)
                print(f"  {season} {game_type}: {len(df)} rows")
            except Exception as e:
                print(f"  ✗ {season} {game_type}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    _save(combined, "player_game_logs.csv")
    return combined


def fetch_todays_games():
    print("\n[3/4] Fetching today's games...")
    _sleep()
    try:
        sb = ScoreboardV2()
        games = sb.get_data_frames()[0]
        line_score = sb.get_data_frames()[1]

        # GameHeader only has HOME_TEAM_ID / VISITOR_TEAM_ID (numeric) — the rest of
        # the pipeline keys off team abbreviations, so map IDs to abbreviations here.
        id_to_abbr = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
        games["HOME_TEAM_ABBREVIATION"]    = games["HOME_TEAM_ID"].map(id_to_abbr)
        games["VISITOR_TEAM_ABBREVIATION"] = games["VISITOR_TEAM_ID"].map(id_to_abbr)

        _save(games, "todays_games.csv")
        _save(line_score, "todays_line_score.csv")
        return games, line_score
    except Exception as e:
        print(f"  ✗ {e}")
        return pd.DataFrame(), pd.DataFrame()


def fetch_team_rosters():
    print("\n[4/4] Fetching rosters...")
    all_teams = teams.get_teams()
    frames = []
    for team in tqdm(all_teams, desc="  Rosters"):
        _sleep()
        try:
            roster = CommonTeamRoster(team_id=team["id"], season=CURRENT_SEASON)
            df = roster.get_data_frames()[0]
            df["TEAM_ABBREVIATION"] = team["abbreviation"]
            frames.append(df)
        except Exception as e:
            print(f"  ✗ {team['abbreviation']}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    _save(combined, "rosters.csv")
    return combined


def fetch_injury_report():
    print("\nFetching injury report...")
    import requests
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for team_entry in data.get("injuries", []):
            for injury in team_entry.get("injuries", []):
                athlete = injury.get("athlete", {}) or {}
                rows.append({
                    "PLAYER_NAME": athlete.get("displayName", ""),
                    "TEAM":        (athlete.get("team") or {}).get("abbreviation", ""),
                    "STATUS":      injury.get("status", ""),
                })
        df = pd.DataFrame(rows)
        _save(df, "injury_report.csv")
        return df
    except Exception as e:
        print(f"  ✗ Injury report failed: {e}")
        return pd.DataFrame()


def run_full_fetch():
    print("=" * 55)
    print("NBA Predictor — Full Data Fetch")
    print(f"Seasons: {SEASONS}")
    print("Estimated time: 10-15 minutes")
    print("=" * 55)
    fetch_team_game_logs()
    fetch_player_game_logs()
    fetch_todays_games()
    fetch_team_rosters()
    fetch_injury_report()
    print("\n✅ Full fetch complete.")


def run_daily_refresh():
    print("NBA Predictor — Daily Refresh")
    fetch_todays_games()
    fetch_injury_report()
    print("✅ Daily refresh complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "daily":
        run_daily_refresh()
    else:
        run_full_fetch()
