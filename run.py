"""
run.py — single entrypoint for everything.

  python run.py --setup       First time: pull data + train models (~15 min)
  python run.py --daily       Every morning: refresh + predict tonight
  python run.py --final       90 min before tip-off: confirmed lineups
  python run.py --retrain     Weekly: retrain models on latest data
  python run.py --backtest    Run backtesting only, no retraining
  python run.py --simulate    Skip the daily refresh, launch UI in simulation-only mode
"""

import argparse
import subprocess
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def run_setup():
    print("\n" + "=" * 55)
    print("  NBA Predictor — FIRST TIME SETUP")
    print("  Takes ~15 minutes. Do not close this window.")
    print("=" * 55 + "\n")

    print("STEP 1/5 — Fetching historical data...")
    from data.fetch import run_full_fetch
    run_full_fetch()

    print("\nSTEP 2/5 — Cleaning data...")
    from data.clean import run_cleaning
    run_cleaning()

    print("\nSTEP 3/5 — Engineering features...")
    from features.engineer import run_feature_engineering
    run_feature_engineering()

    print("\nSTEP 4/5 — Training game model + backtesting...")
    from models.game_model import run_game_model_pipeline
    run_game_model_pipeline()

    print("\nSTEP 5/5 — Training player models + backtesting...")
    from models.player_model import run_player_model_pipeline
    run_player_model_pipeline()

    print("\n" + "=" * 55)
    print("  ✅ Setup complete!")
    print("  python run.py --daily    → generate tonight's predictions")
    print("  python -m streamlit run app.py    → launch UI")
    print("=" * 55 + "\n")


def run_daily():
    print("\nNBA Predictor — Daily Run\n")
    from pipeline import run_tonight
    run_tonight(final_run=False)
    print("\n✅ Done. Launch UI: python -m streamlit run app.py\n")


def run_final():
    print("\nNBA Predictor — Final Run (90 min pre-tip)\n")
    from pipeline import run_tonight
    run_tonight(final_run=True)
    print("\n✅ Final predictions locked. Refresh the UI.\n")


def run_retrain():
    print("\nNBA Predictor — Weekly Retrain\n")

    from data.fetch import run_daily_refresh
    run_daily_refresh()

    from features.engineer import run_feature_engineering
    run_feature_engineering()

    from models.game_model import run_game_model_pipeline
    run_game_model_pipeline()

    from models.player_model import run_player_model_pipeline
    run_player_model_pipeline()

    print("\n✅ Retrain complete.\n")


def run_simulate():
    print("\nNBA Predictor — Simulation Mode\n")
    print("Skipping daily data refresh — launching UI in simulation-only mode.")
    print("Use the Custom Matchup Simulator tab to model any matchup.\n")
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def run_backtest():
    print("\nNBA Predictor — Backtesting Only\n")
    import pandas as pd
    from models.game_model import _load_features, walk_forward_backtest, print_backtest_summary, TARGETS
    from models.player_model import walk_forward_backtest_player, print_player_backtest_summary, STAT_TARGETS
    import os

    FEAT_DIR = os.path.join(os.path.dirname(__file__), "data", "features")

    print("[Game model]")
    df = _load_features()
    for name, target_col in TARGETS.items():
        if target_col in df.columns:
            results = walk_forward_backtest(df, target_col)
            print_backtest_summary(results, name)

    print("\n[Player models]")
    player_df = pd.read_csv(
        os.path.join(FEAT_DIR, "player_features.csv"),
        parse_dates=["GAME_DATE"], low_memory=False
    )
    for stat in STAT_TARGETS:
        if stat in player_df.columns:
            results = walk_forward_backtest_player(player_df, stat)
            print_player_backtest_summary(results, stat)

    print("\n✅ Backtesting complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Predictor")
    parser.add_argument("--setup",    action="store_true")
    parser.add_argument("--daily",    action="store_true")
    parser.add_argument("--final",    action="store_true")
    parser.add_argument("--retrain",  action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    if args.setup:
        run_setup()
    elif args.daily:
        run_daily()
    elif args.final:
        run_final()
    elif args.retrain:
        run_retrain()
    elif args.backtest:
        run_backtest()
    elif args.simulate:
        run_simulate()
    else:
        parser.print_help()
