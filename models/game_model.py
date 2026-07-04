"""
models/game_model.py
--------------------
XGBoost model predicting HOME score, AWAY score, total, and spread.
Walk-forward backtesting runs automatically before training.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

FEAT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "HOME_PTS_ROLL5", "HOME_PTS_ROLL10", "HOME_PTS_ROLL20",
    "HOME_FG_PCT_ROLL10", "HOME_FG3_PCT_ROLL10",
    "HOME_AST_ROLL10", "HOME_TOV_ROLL10",
    "HOME_OFF_RTG_PROXY_ROLL10", "HOME_WIN_RATE_ROLL10",
    "HOME_REST_DAYS", "HOME_IS_B2B", "HOME_IS_PLAYOFF",
    "AWAY_PTS_ROLL5", "AWAY_PTS_ROLL10", "AWAY_PTS_ROLL20",
    "AWAY_FG_PCT_ROLL10", "AWAY_FG3_PCT_ROLL10",
    "AWAY_AST_ROLL10", "AWAY_TOV_ROLL10",
    "AWAY_OFF_RTG_PROXY_ROLL10", "AWAY_WIN_RATE_ROLL10",
    "AWAY_REST_DAYS", "AWAY_IS_B2B",
    "HOME_OPP_DEF_PTS_ALLOWED_ROLL10",
    "AWAY_OPP_DEF_PTS_ALLOWED_ROLL10",
]

TARGETS = {
    "home_pts":  "TARGET_HOME_PTS",
    "away_pts":  "TARGET_AWAY_PTS",
    "total_pts": "TARGET_TOTAL_PTS",
    "spread":    "TARGET_SPREAD",
}

XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1,
}

MAE_BENCHMARKS = {
    "home_pts": 8.0, "away_pts": 8.0,
    "total_pts": 8.0, "spread": 9.0,
}


def _load_features():
    return pd.read_csv(os.path.join(FEAT_DIR, "game_model_features.csv"),
                       parse_dates=["GAME_DATE"], low_memory=False)


def _get_feat_cols(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def prepare_X(df, feat_cols, fill_vals=None):
    """Deduplicate columns, force numeric, fill NaNs safely."""
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].copy()
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if fill_vals is not None:
        for col in X.columns:
            X[col] = X[col].fillna(fill_vals.get(col, 0.0))
    else:
        medians = X.median(numeric_only=True)
        for col in X.columns:
            X[col] = X[col].fillna(medians.get(col, 0.0))
    return X.reset_index(drop=True)


def walk_forward_backtest(df, target_col, n_train=500, step=50):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    feat_cols = _get_feat_cols(df)
    results = []

    for i in range(n_train, len(df), step):
        train = df.iloc[:i]
        test  = df.iloc[i:i + step]
        if len(test) == 0:
            break

        train_numeric = train[feat_cols].apply(pd.to_numeric, errors="coerce")
        fill_vals = train_numeric.median().to_dict()

        X_train = prepare_X(train, feat_cols, fill_vals)
        y_train = train[target_col].reset_index(drop=True)
        X_test  = prepare_X(test, list(X_train.columns), fill_vals)
        y_test  = test[target_col].reset_index(drop=True)

        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_test = X_test[X_train.columns]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)

        id_cols = ["GAME_ID", "GAME_DATE"]
        id_cols = [c for c in id_cols if c in test.columns]
        fold = test[id_cols].copy().reset_index(drop=True)
        fold["PREDICTED"] = preds
        fold["ACTUAL"]    = y_test.values
        fold["ERROR"]     = preds - y_test.values
        fold["ABS_ERROR"] = np.abs(fold["ERROR"])
        results.append(fold)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def print_backtest_summary(results, name):
    if results.empty:
        return {}
    mae  = results["ABS_ERROR"].mean()
    rmse = np.sqrt((results["ERROR"] ** 2).mean())
    bias = results["ERROR"].mean()
    benchmark = MAE_BENCHMARKS.get(name, 999)
    flag = "⚠️  WORSE THAN BENCHMARK" if mae > benchmark else "✓"
    print(f"\n  ── {name} ──")
    print(f"    MAE  : {mae:.2f}  (benchmark ≤{benchmark})  {flag}")
    print(f"    RMSE : {rmse:.2f}  |  Bias: {bias:+.2f}")
    return {"target": name, "mae": mae, "rmse": rmse, "bias": bias}


def train_game_models(df):
    feat_cols = _get_feat_cols(df)
    X = prepare_X(df, feat_cols)
    models = {}

    for name, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue
        y = df[target_col].reset_index(drop=True)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X, y, verbose=False)
        models[name] = model
        path = os.path.join(MODEL_DIR, f"game_model_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  ✓ Saved game_model_{name}")

    with open(os.path.join(MODEL_DIR, "game_model_features.json"), "w") as f:
        json.dump(list(X.columns), f)

    return models


def load_game_models():
    models = {}
    for name in TARGETS:
        path = os.path.join(MODEL_DIR, f"game_model_{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def run_game_model_pipeline():
    print("=" * 55)
    print("NBA Predictor — Game Score Model")
    print("=" * 55)

    df = _load_features()
    print(f"Loaded {len(df):,} games")

    print("\n[BACKTESTING]")
    all_metrics = []
    for name, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue
        results = walk_forward_backtest(df, target_col)
        metrics = print_backtest_summary(results, name)
        all_metrics.append(metrics)
        results.to_csv(os.path.join(MODEL_DIR, f"backtest_{name}.csv"), index=False)

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(MODEL_DIR, "game_model_backtest_summary.csv"), index=False)

    print("\n[TRAINING final models]")
    train_game_models(df)

    print("\n✅ Game model pipeline complete.")


if __name__ == "__main__":
    run_game_model_pipeline()
