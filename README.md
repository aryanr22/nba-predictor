# NBA Predictor

Predicts NBA game scores, spreads, totals, and per-player box-score stats (points, rebounds,
assists, 3PM, steals, blocks, turnovers, minutes) using historical game logs pulled from the
NBA's own stats API. Includes a lineup resolver that accounts for injuries and redistributes
minutes when players are out, a Streamlit dashboard for browsing predictions, and a
walk-forward backtesting harness that reports model accuracy against fixed MAE benchmarks
before every retrain.

> ⚠️ For informational and educational use only. Not betting advice.

## Tech stack

| Layer | Tool |
|---|---|
| Data source | [`nba_api`](https://github.com/swar/nba_api) (official NBA stats endpoints) + ESPN injury feed |
| Data wrangling | pandas / numpy |
| Modeling | XGBoost (`xgb.XGBRegressor`) |
| Validation | Custom walk-forward (time-respecting) backtester |
| UI | Streamlit |
| Optional chat | Anthropic API (Claude) for the "Ask Claude" tab |

## How it works, end to end

```
data/fetch.py        → pull raw team/player game logs, rosters, today's games, injuries
data/clean.py         → normalize columns, derive WIN/REST_DAYS/IS_HOME/IS_B2B, etc.
features/engineer.py  → build rolling-window features (5/10/20 games), all shift(1)'d
                         so no game ever leaks its own result into its own features
models/game_model.py   → walk-forward backtest, then train XGBoost regressors for
                         home score, away score, total, and spread
models/player_model.py → same, but one model per stat (PTS, REB, AST, FG3M, STL, BLK, TOV, MIN)
lineups/resolver.py    → builds each team's depth chart from recent games, applies
                         injury statuses, and redistributes an OUT player's minutes
                         across the rest of the roster weighted by their own recent minutes
pipeline.py            → ties it together for "tonight's games": resolve lineups,
                         run both models, write predictions to data/predictions/
app.py                 → Streamlit UI over the generated predictions
```

### Models

- **Game model** (`models/game_model.py`): four independent XGBoost regressors
  (`home_pts`, `away_pts`, `total_pts`, `spread`), trained on team-level rolling
  averages (scoring, shooting splits, assists/turnovers, an offensive-rating proxy,
  win rate, rest days, back-to-back flag, opponent points-allowed) for both teams
  in a matchup.
- **Player models** (`models/player_model.py`): one XGBoost regressor per stat,
  trained on that player's own rolling averages, season averages, a usage-rate
  proxy, and (at prediction time) the projected minutes and projected game total
  coming out of the lineup resolver and game model.
- Both use fixed hyperparameters (no tuning search) — `n_estimators`, `max_depth`,
  `learning_rate`, etc. are hardcoded at the top of each file.
- All rolling features are computed with `.shift(1)` before the rolling window, so
  a game's features never include that game's own result.

### Lineup resolution

`lineups/resolver.py` ranks each team's roster by average minutes over their last
15 games, looks up each player's status from the injury report, and — if any
player is marked `Out` — redistributes their average minutes across the rest of
the roster proportionally to each player's own current minutes (capped at 40),
scaled down further for `Doubtful`/`Questionable`/`Probable` players by a fixed
play-probability table.

## Install

Requires Python 3.10+.

```bash
git clone <this-repo>
cd nba_predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No API key is required for data fetching or model training — `nba_api` and the
ESPN injury endpoint are both unauthenticated. An Anthropic API key is only
needed if you want to use the "Ask Claude" tab in the Streamlit UI (entered
directly in the sidebar at runtime, not stored anywhere).

## Running it

```bash
python run.py --setup       # first time: fetch history + engineer features + train + backtest (~15 min)
python run.py --daily       # every morning: refresh today's games/injuries + predict
python run.py --final       # ~90 min before tip-off: re-resolve lineups off confirmed injury statuses
python run.py --retrain     # weekly: refetch data, rebuild features, retrain + backtest both models
python run.py --backtest    # backtest only, using whatever data/models already exist — no fetching or training
python run.py --simulate    # offseason / no games: skip the daily refresh, launch UI in simulation-only mode
```

During the offseason, when there are no live games scheduled, use `--simulate`
instead of `--daily`/`--final` — it skips the daily data refresh entirely and
launches straight into the Streamlit UI, defaulting to the Custom Matchup
Simulator so you can still model hypothetical matchups.

Then view results with:

```bash
python -m streamlit run app.py
```

The UI has four tabs: **Tonight's Games**, **Player Projections**, a **Custom Matchup**
simulator (pick any two teams and optionally mark players out), and **Ask Claude**
(chat over tonight's generated projections).

### Command reference

| Command | What it does | When to run it |
|---|---|---|
| `--setup` | Full historical fetch (`SEASONS` in `data/fetch.py`) → clean → engineer features → train + backtest both models | Once, on a fresh checkout |
| `--daily` | Refresh today's games + injury report, resolve lineups, generate predictions | Every morning |
| `--final` | Same as `--daily`, but intended to run ~90 min before tip-off once injury designations are confirmed | Late afternoon / pre-game |
| `--retrain` | Refetch all game logs, rebuild features, retrain + backtest both models from scratch | Weekly, to keep rolling stats and models current |
| `--backtest` | Re-runs the walk-forward backtest on existing feature files without touching data or models | Whenever you want fresh accuracy numbers without a full retrain |
| `--simulate` | Skips the daily data refresh and launches the Streamlit UI directly in simulation-only mode | Offseason, or any time there are no live games scheduled |

## Backtesting

Both model pipelines run a **walk-forward** backtest before every training run:
the data is sorted chronologically, trained on an initial window, evaluated on
the next block, then the window slides forward and repeats — so no test fold is
ever trained on future data.

- Game model: trains on an initial 500 games, tests in blocks of 50, and slides forward across the full history (`models/game_model.py:walk_forward_backtest`).
- Player models: trains on an initial 5,000 rows, tests in blocks of 500 (`models/player_model.py:walk_forward_backtest_player`).

Each run prints MAE, RMSE, and bias against a fixed benchmark per target, and
flags anything worse than benchmark:

**Game model** (points/spread, `MAE_BENCHMARKS` in `models/game_model.py`)

| Target | MAE benchmark |
|---|---|
| Home points | ≤ 8.0 |
| Away points | ≤ 8.0 |
| Total points | ≤ 8.0 |
| Spread | ≤ 9.0 |

**Player models** (`MAE_BENCHMARKS` in `models/player_model.py`)

| Stat | MAE benchmark |
|---|---|
| PTS | ≤ 5.5 |
| REB | ≤ 2.2 |
| AST | ≤ 2.0 |
| FG3M | ≤ 1.2 |
| STL | ≤ 0.8 |
| BLK | ≤ 0.7 |
| TOV | ≤ 1.2 |
| MIN | ≤ 6.0 |

Per-fold results are written to `models/saved/backtest_<target>.csv` and
`models/saved/backtest_player_<stat>.csv`, with summary tables in
`models/saved/game_model_backtest_summary.csv` and
`models/saved/player_model_backtest_summary.csv`. The player backtest also
prints each stat's 3 best- and worst-predicted players by mean absolute error.

## Data layout

```
data/raw/         # unmodified nba_api / ESPN pulls (game logs, rosters, today's games, injuries)
data/processed/   # cleaned, normalized game logs
data/features/    # rolling-window feature matrices used for training/inference
data/predictions/ # daily prediction JSON/CSV output, plus latest_predictions.json
models/saved/     # trained model .pkl files, feature column lists, backtest CSVs
```
