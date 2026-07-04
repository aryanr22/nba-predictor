"""
models/player_model.py
----------------------
Per-stat XGBoost regression models for player projections.
One model per stat: PTS, REB, AST, FG3M, STL, BLK, TOV, MIN.
Includes prepare_X() which fixes the XGBoost duplicate-column dtype error.
"""

import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

FEAT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

STAT_TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]

MAE_BENCHMARKS = {
    "PTS": 5.5, "REB": 2.2, "AST": 2.0, "FG3M": 1.2,
    "STL": 0.8, "BLK": 0.7, "TOV": 1.2, "MIN": 6.0,
}

XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 5, "learning_rate": 0.04,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
    "reg_alpha": 0.2, "reg_lambda": 1.5, "random_state": 42, "n_jobs": -1,
}


def get_player_feature_cols(stat, df):
    base = [
        f"{stat}_ROLL5", f"{stat}_ROLL10", f"{stat}_ROLL20",
        "MIN_ROLL5", "MIN_ROLL10", "MIN_SEASON_AVG",
        "USAGE_ROLL10", "IS_HOME", "IS_PLAYOFF",
        "GAMES_PLAYED_LAST10", "GAMES_PLAYED_LAST20",
    ]
    if stat == "PTS":
        base.append("PTS_SEASON_AVG")
    extra = ["PROJECTED_GAME_TOTAL", "PROJECTED_MIN"]
    return [c for c in base + extra if c in df.columns]


def prepare_X(df, feat_cols, fill_vals=None):
    """
    Safe feature matrix builder.
    - Deduplicates columns (fixes XGBoost 'DataFrame has no attribute dtype' error)
    - Forces numeric types
    - Fills NaNs consistently
    """
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].copy()
    # Drop duplicate column names — root cause of the XGBoost error
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    # Force every column to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # Fill NaNs
    if fill_vals is not None:
        for col in X.columns:
            X[col] = X[col].fillna(fill_vals.get(col, 0.0))
    else:
        medians = X.median(numeric_only=True)
        for col in X.columns:
            X[col] = X[col].fillna(medians.get(col, 0.0))
    return X.reset_index(drop=True)


def walk_forward_backtest_player(df, stat, n_train=5000, step=500):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    feat_cols = get_player_feature_cols(stat, df)
    if not feat_cols:
        return pd.DataFrame()

    results = []
    for i in range(n_train, len(df), step):
        train = df.iloc[:i].dropna(subset=[stat] + feat_cols[:2])
        test  = df.iloc[i:i + step].dropna(subset=[stat] + feat_cols[:2])
        if len(test) < 10:
            break

        fill_vals = train[[c for c in feat_cols if c in train.columns]].apply(
            pd.to_numeric, errors="coerce").median().to_dict()

        X_train = prepare_X(train, feat_cols, fill_vals)
        y_train = train[stat].reset_index(drop=True)
        X_test  = prepare_X(test, list(X_train.columns), fill_vals)
        y_test  = test[stat].reset_index(drop=True)

        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_test = X_test[X_train.columns]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test).clip(0)

        id_cols = [c for c in ["GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
                                "TEAM_ABBREVIATION"] if c in test.columns]
        fold = test[id_cols].copy().reset_index(drop=True)
        fold["PREDICTED"] = preds
        fold["ACTUAL"]    = y_test.values
        fold["ERROR"]     = preds - y_test.values
        fold["ABS_ERROR"] = np.abs(fold["ERROR"])
        results.append(fold)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def print_player_backtest_summary(results, stat):
    if len(results) == 0:
        return {}
    mae  = results["ABS_ERROR"].mean()
    rmse = np.sqrt((results["ERROR"] ** 2).mean())
    bias = results["ERROR"].mean()
    benchmark = MAE_BENCHMARKS.get(stat, 999)
    flag = "⚠️  WORSE THAN BENCHMARK" if mae > benchmark else "✓"
    print(f"\n  ── {stat} ──")
    print(f"    MAE  : {mae:.2f}  (benchmark ≤{benchmark})  {flag}")
    print(f"    RMSE : {rmse:.2f}  |  Bias: {bias:+.2f}")
    if "PLAYER_NAME" in results.columns:
        by_player = results.groupby("PLAYER_NAME")["ABS_ERROR"].mean().sort_values()
        print(f"    Best  3: {list(by_player.head(3).index)}")
        print(f"    Worst 3: {list(by_player.tail(3).index)}")
    return {"stat": stat, "mae": mae, "rmse": rmse, "bias": bias}


def train_player_models(df):
    models = {}
    for stat in STAT_TARGETS:
        if stat not in df.columns:
            continue
        feat_cols = get_player_feature_cols(stat, df)
        if not feat_cols:
            continue
        train_df = df.dropna(subset=[stat] + feat_cols[:2])
        X = prepare_X(train_df, feat_cols)
        y = train_df[stat].reset_index(drop=True)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X, y, verbose=False)
        final_cols = list(X.columns)
        models[stat] = {"model": model, "feature_cols": final_cols}
        path = os.path.join(MODEL_DIR, f"player_model_{stat}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": final_cols}, f)
        print(f"  ✓ Saved player_model_{stat}")
    return models


def load_player_models():
    models = {}
    for stat in STAT_TARGETS:
        path = os.path.join(MODEL_DIR, f"player_model_{stat}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[stat] = pickle.load(f)
    return models


def predict_player_game(player_features, game_context=None):
    models = load_player_models()
    row = dict(player_features)
    if game_context:
        row.update(game_context)
    predictions = {}
    for stat, m in models.items():
        X = pd.DataFrame([row])
        for col in m["feature_cols"]:
            if col not in X.columns:
                X[col] = 0.0
        X = prepare_X(X, m["feature_cols"])
        pred = float(m["model"].predict(X)[0])
        predictions[stat] = round(max(0.0, pred), 1)
    return predictions


def run_player_model_pipeline():
    print("=" * 55)
    print("NBA Predictor — Player Stat Models")
    print("=" * 55)

    player_df = pd.read_csv(
        os.path.join(FEAT_DIR, "player_features.csv"),
        parse_dates=["GAME_DATE"], low_memory=False
    )
    print(f"Loaded {len(player_df):,} rows, {player_df['PLAYER_ID'].nunique():,} players")

    print("\n[BACKTESTING]")
    all_metrics = []
    for stat in STAT_TARGETS:
        if stat not in player_df.columns:
            continue
        results = walk_forward_backtest_player(player_df, stat)
        metrics = print_player_backtest_summary(results, stat)
        if metrics:
            all_metrics.append(metrics)
            results.to_csv(os.path.join(MODEL_DIR, f"backtest_player_{stat}.csv"), index=False)

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(MODEL_DIR, "player_model_backtest_summary.csv"), index=False)

    print("\n[TRAINING final models]")
    train_player_models(player_df)
    print("\n✅ Player model pipeline complete.")


if __name__ == "__main__":
    run_player_model_pipeline()
