"""LightGBM の学習とパイプライン実行。

Colab では:

    from src import train
    result = train.run_pipeline("/content/drive/MyDrive/keiba/race_result.csv")

だけで、前処理 → 特徴量 → 学習 → 回収率評価 → 期待値検証まで一気に走る。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import betting, config, evaluate, features, preprocess


@dataclass
class PipelineResult:
    model: object
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    pred: np.ndarray
    feature_cols: list[str]
    importance: pd.DataFrame
    topn_report: pd.DataFrame
    ev_report: pd.DataFrame
    high_odds_report: pd.DataFrame


def train_lgb(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str],
              params: dict | None = None, num_boost_round: int = 2000,
              early_stopping_rounds: int = 100):
    """LightGBM の二値分類モデルを学習する。"""
    import lightgbm as lgb

    params = {**config.LGB_PARAMS, **(params or {})}
    target = config.TARGET

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[target])
    dvalid = lgb.Dataset(valid_df[feature_cols], label=valid_df[target], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def importance_table(model, feature_cols: list[str]) -> pd.DataFrame:
    """gain 重要度を降順で返す。"""
    return pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)


def run_pipeline(csv_path: str | None = None, df: pd.DataFrame | None = None,
                 include_odds: bool = False, strict_daily_lag: bool = False,
                 start_year: int | None = None,
                 verbose: bool = True) -> PipelineResult:
    """CSV から評価まで一気通貫で実行する。

    Parameters
    ----------
    include_odds : True にすると単勝オッズ・人気も特徴量に入れる（比較用）。
                   本来の目的（市場より賢く予測する）では False のまま使う。
    start_year   : メモリ・時間がきつい場合に学習対象を年で絞る。
                   特徴量は絞る前の全期間から計算するので通算成績は壊れない。
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    if df is None:
        log("[1/6] CSV 読み込み中...")
        df = preprocess.load_race_result(csv_path)

    log("[2/6] 前処理...")
    df = preprocess.basic_clean(df)
    log(f"      {len(df):,} 行 / メモリ {preprocess.memory_usage_mb(df):.0f} MB")

    log("[3/6] 特徴量作成...")
    df = features.add_all_features(df, strict_daily_lag=strict_daily_lag)
    df = preprocess.downcast(df)
    df = preprocess.to_category(df)
    log(f"      {df.shape[1]} 列 / メモリ {preprocess.memory_usage_mb(df):.0f} MB")

    if start_year is not None:
        df = preprocess.filter_years(df, start_year=start_year)
        log(f"      {start_year} 年以降に絞り込み → {len(df):,} 行")

    log("[4/6] 時系列分割...")
    train_df, test_df, split_date = preprocess.time_series_split(df)
    log(f"      分割日 {split_date:%Y-%m-%d} / train {len(train_df):,} / test {len(test_df):,}")

    feature_cols = features.feature_columns(df, include_odds=include_odds)
    log(f"[5/6] 学習（特徴量 {len(feature_cols)} 個）...")
    model = train_lgb(train_df, test_df, feature_cols)
    pred = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)

    log("[6/6] 回収率で評価...")
    imp = importance_table(model, feature_cols)
    topn = evaluate.backtest_by_top_n(test_df, pred, ns=(1, 2, 3))
    ev = betting.ev_threshold_sweep(test_df, pred)
    high = betting.high_odds_report(test_df, pred, ev_threshold=1.0)

    if verbose:
        print("\n--- gain 重要度 上位15 ---")
        print(imp.head(15).to_string(index=False))
        print("\n--- 単勝: 予測上位N頭を買った場合 ---")
        print(topn.to_string(index=False))
        print("\n--- 参考: 1番人気を買い続けた場合 ---")
        print(evaluate.favorite_baseline(test_df).summary())
        print("\n--- 期待値フィルタ（閾値スイープ） ---")
        print(ev.to_string(index=False))
        print("\n--- 期待値1.0以上 × オッズ帯 ---")
        print(high.to_string(index=False))

    return PipelineResult(
        model=model, train_df=train_df, test_df=test_df, pred=pred,
        feature_cols=feature_cols, importance=imp,
        topn_report=topn, ev_report=ev, high_odds_report=high,
    )
