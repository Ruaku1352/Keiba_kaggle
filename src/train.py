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

from . import (betting, config, corner, evaluate, features, pace, payout,
               preprocess, trifecta)


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
    auc: float | None = None
    race_table: pd.DataFrame | None = None      # 1行1レースの検証テーブル
    strategy_all: pd.DataFrame | None = None    # 買い方の総当たり（全レース）
    strategy_graded: pd.DataFrame | None = None  # 同（重賞のみ）
    graded_vs_flat: pd.DataFrame | None = None  # 重賞 vs 平場
    gap_result: dict | None = None              # 乖離ベースの成績
    verdict: pd.DataFrame | None = None         # v2ベースラインとの比較


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
                 corner_path: str | None = None, lap_path: str | None = None,
                 odds_path: str | None = None,
                 corner_df: pd.DataFrame | None = None,
                 lap_df: pd.DataFrame | None = None,
                 payout_df: pd.DataFrame | None = None,
                 verbose: bool = True) -> PipelineResult:
    """CSV から評価まで一気通貫で実行する。

    Parameters
    ----------
    include_odds : True にすると単勝オッズ・人気も特徴量に入れる（比較用）。
                   本来の目的（市場より賢く予測する）では False のまま使う。
    start_year   : メモリ・時間がきつい場合に学習対象を年で絞る。
                   特徴量は絞る前の全期間から計算するので通算成績は壊れない。
    corner_path  : corner_passing_order.csv のパス。渡すと脚質特徴量（依頼A）が入る。
    lap_path     : laptime.csv のパス。渡すとペース特徴量（依頼B）が入る。
    odds_path    : odds.csv のパス。渡すと3連単の買い方検証（依頼C）まで実行する。
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

    # --- 補助データの読み込み ------------------------------------------------
    if corner_df is None and corner_path is not None:
        log("      コーナー通過順を読み込み中...")
        corner_df = corner.load_corner(corner_path)
        log(f"      {len(corner_df):,} 行（1行=1馬1コーナー）")
    if lap_df is None and lap_path is not None:
        log("      ラップタイムを読み込み中...")
        lap_df = pace.load_laptime(lap_path)
        log(f"      {len(lap_df):,} レース分")
    if payout_df is None and odds_path is not None:
        log("      3連単配当を読み込み中...")
        payout_df = payout.load_trifecta_payout(odds_path)
        log(f"      {len(payout_df):,} レース分")

    log("[3/6] 特徴量作成...")
    df = features.add_all_features(df, strict_daily_lag=strict_daily_lag,
                                   corner_df=corner_df, lap_df=lap_df)
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

    auc = None
    try:
        auc = float(model.best_score["valid"]["auc"])
    except (KeyError, TypeError):
        pass

    # --- 依頼C：3連単の買い方検証 -------------------------------------------
    race_table = strategy_all = strategy_graded = graded_vs_flat = None
    gap_result = verdict_table = None
    if payout_df is not None:
        log("      3連単の買い方を総当たり...")
        race_table = trifecta.build_race_table(test_df, pred, payout_df)
        strategy_all = trifecta.strategy_table(race_table)
        gap_result = trifecta.simulate_gap_box(test_df, pred, payout_df)
        verdict_table = trifecta.verdict(auc, strategy_all, gap_result)

        n_graded = int(race_table["重賞"].sum())
        if n_graded > 0:
            strategy_graded = trifecta.strategy_table(race_table, graded_only=True)
            graded_vs_flat = trifecta.compare_graded_vs_flat(race_table)
        log(f"      検証レース数 {len(race_table):,}（うち重賞 {n_graded:,}）")

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

        if strategy_all is not None:
            print("\n--- 3連単の買い方（全レース。達成率=熱い当たりが出た割合 %） ---")
            print(trifecta.format_table(strategy_all).to_string(index=False))
            print("\n--- 乖離ベース（最重要の判定対象） ---")
            print(pd.Series(gap_result).to_string())
        if strategy_graded is not None:
            print("\n--- 3連単の買い方（重賞のみ） ---")
            print(trifecta.format_table(strategy_graded).to_string(index=False))
            print("\n--- 重賞 vs 平場 ---")
            print(trifecta.format_table(graded_vs_flat).to_string(index=False))
        if verdict_table is not None:
            print("\n--- v2ベースラインとの比較（依頼書セクション6） ---")
            print(verdict_table.to_string(index=False))

    return PipelineResult(
        model=model, train_df=train_df, test_df=test_df, pred=pred,
        feature_cols=feature_cols, importance=imp,
        topn_report=topn, ev_report=ev, high_odds_report=high,
        auc=auc, race_table=race_table, strategy_all=strategy_all,
        strategy_graded=strategy_graded, graded_vs_flat=graded_vs_flat,
        gap_result=gap_result, verdict=verdict_table,
    )
