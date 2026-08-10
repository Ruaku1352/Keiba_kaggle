"""期待値ベースの買い目抽出（依頼3）。

考え方はシンプルで、

    期待値 EV = 予測確率 × 単勝オッズ

これが 1.0 を超える馬は「理論上は買えば増える」馬。
つまり **市場（オッズ）が過小評価している馬** を探すということ。

注意点が 2 つある。

1. 予測確率はレース内で合計 1 に正規化してから使う。
   LightGBM の binary 出力はレース単位で合計 1 にならないため、
   そのまま掛けると期待値が系統的にずれる。

2. 期待値 1.0 ちょうどを閾値にすると、モデルの過信でノイズを拾いやすい。
   実運用では 1.1〜1.3 くらいまで上げた方が安定することが多いので、
   `ev_threshold_sweep()` で閾値を振って確認できるようにしている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import BET_UNIT, BacktestResult, _prepare, _settle, normalize_prob_by_race


def compute_expected_value(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                           normalize: bool = True) -> pd.DataFrame:
    """各馬の期待値を計算した DataFrame を返す。

    戻り値の列: race_id, date, rank, odds, pred, prob（正規化後）, ev
    """
    base = _prepare(df, pred)
    base["prob"] = normalize_prob_by_race(base) if normalize else base["pred"]
    base["ev"] = base["prob"] * base["odds"]
    # 市場が見積もる勝率（控除率込みなので合計は 1 を超える）
    base["市場勝率"] = 1.0 / base["odds"]
    # モデルと市場の乖離。大きいほど「妙味あり」と判断している
    base["妙味"] = base["prob"] - base["市場勝率"]
    return base


def select_value_bets(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                      ev_threshold: float = 1.0,
                      min_odds: float | None = None,
                      max_odds: float | None = None,
                      min_prob: float | None = None,
                      max_bets_per_race: int | None = None) -> pd.DataFrame:
    """期待値が閾値を超えた馬（＝買い目）を抽出する。

    Parameters
    ----------
    ev_threshold : 期待値の下限（1.0 なら理論上トントン以上）
    min_odds     : 例) 10 を指定すると高配当だけに絞る
    max_odds     : 上限オッズ（穴を狙いすぎないための蓋）
    min_prob     : 予測確率の下限（当たらなすぎる買い目を除く）
    max_bets_per_race : 1レースあたりの最大点数（期待値の高い順に残す）
    """
    ev = compute_expected_value(df, pred)

    mask = ev["ev"] >= ev_threshold
    if min_odds is not None:
        mask &= ev["odds"] >= min_odds
    if max_odds is not None:
        mask &= ev["odds"] < max_odds
    if min_prob is not None:
        mask &= ev["prob"] >= min_prob
    picks = ev.loc[mask].copy()

    if max_bets_per_race is not None and not picks.empty:
        order = picks.groupby("race_id")["ev"].rank(ascending=False, method="first")
        picks = picks.loc[order <= max_bets_per_race]

    return picks.sort_values(["date", "race_id", "ev"], ascending=[True, True, False])


def backtest_value_bets(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                        **kwargs) -> BacktestResult:
    """期待値ベースの買い目で単勝100円を賭けたときの成績。

    kwargs は select_value_bets と同じ（ev_threshold, min_odds, ...）。
    """
    picks = select_value_bets(df, pred, **kwargs)
    n_races = _prepare(df, pred)["race_id"].nunique()
    if picks.empty:
        return BacktestResult(n_races=n_races, n_bets=0, n_hits=0,
                              invested=0, returned=0.0, bets=picks)
    return _settle(picks, n_races)


def ev_threshold_sweep(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                       thresholds: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0),
                       **kwargs) -> pd.DataFrame:
    """期待値の閾値を振って、回収率がどう変わるかを一覧にする。

    「閾値を上げる → 点数は減るが回収率は上がる」なら、モデルの期待値は
    ある程度信頼できているというサイン。逆にバラバラなら単なるノイズ。
    """
    rows = []
    for t in thresholds:
        r = backtest_value_bets(df, pred, ev_threshold=t, **kwargs)
        rows.append({
            "EV閾値": t, "購入点数": r.n_bets, "的中数": r.n_hits,
            "的中率": r.hit_rate, "回収率": r.roi, "収支": r.profit,
        })
    return pd.DataFrame(rows)


def high_odds_report(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                     ev_threshold: float = 1.0,
                     odds_floors: tuple[float, ...] = (1, 5, 10, 20, 50)) -> pd.DataFrame:
    """「オッズ X 倍以上」に絞ったときの回収率を並べる（高配当狙いの検証用）。"""
    rows = []
    for floor in odds_floors:
        r = backtest_value_bets(df, pred, ev_threshold=ev_threshold, min_odds=floor)
        rows.append({
            "最低オッズ": floor, "購入点数": r.n_bets, "的中数": r.n_hits,
            "的中率": r.hit_rate, "回収率": r.roi, "収支": r.profit,
        })
    return pd.DataFrame(rows)


def kelly_fraction(prob: pd.Series | np.ndarray, odds: pd.Series | np.ndarray,
                   cap: float = 0.05) -> np.ndarray:
    """ケリー基準による賭け金比率（資金の何割を賭けるか）。

    f = (p * b - (1 - p)) / b,  b = オッズ - 1
    予測確率のブレで一気に破産しないよう、既定で 5% を上限にしている。
    """
    p = np.asarray(prob, dtype="float64")
    b = np.asarray(odds, dtype="float64") - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p * b - (1.0 - p)) / b
    return np.clip(np.nan_to_num(f, nan=0.0), 0.0, cap)


def race_picks(df: pd.DataFrame, pred: np.ndarray | pd.Series, race_id,
               top: int = 8) -> pd.DataFrame:
    """特定の1レースについて、期待値順の買い目候補を表示する。

    本番（有馬記念など）で「この馬が妙味あり」を確認するための関数。
    """
    ev = compute_expected_value(df, pred)
    one = ev.loc[ev["race_id"] == race_id].copy()
    show = [c for c in ["post", "horse", "odds", "prob", "ev", "市場勝率", "妙味", "rank"]
            if c in one.columns]
    return one.sort_values("ev", ascending=False).head(top)[show]


def bet_amounts(picks: pd.DataFrame, bankroll: float, use_kelly: bool = False,
                cap: float = 0.05) -> pd.DataFrame:
    """買い目に賭け金を割り当てる（均等 or ケリー）。100円単位に丸める。"""
    out = picks.copy()
    if use_kelly:
        f = kelly_fraction(out["prob"], out["odds"], cap=cap)
        amount = bankroll * f
    else:
        amount = np.full(len(out), float(BET_UNIT))
    out["賭け金"] = (np.floor(amount / BET_UNIT) * BET_UNIT).astype(int)
    return out.loc[out["賭け金"] > 0]
