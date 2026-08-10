"""評価関数（依頼2）：正解率ではなく「回収率」で評価する。

単勝100円を賭けたときの収支をシミュレーションする。

  払戻金 = 単勝オッズ × 100円（1着のときのみ。外れたら 0 円）
  回収率 = 総払戻 / 総投資

回収率 100% が損益トントン。JRA の単勝の控除率は約 20% なので、
何も考えずに買うと回収率は 80% 前後に収束する。
つまり「80% を超えたか」が最初の関門になる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

BET_UNIT = 100  # 1点あたりの賭け金（円）


@dataclass
class BacktestResult:
    """シミュレーション結果のまとめ。"""

    n_races: int
    n_bets: int
    n_hits: int
    invested: int
    returned: float
    bets: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def hit_rate(self) -> float:
        """的中率（賭けた点数のうち当たった割合）。"""
        return self.n_hits / self.n_bets if self.n_bets else float("nan")

    @property
    def roi(self) -> float:
        """回収率（1.0 = 100%）。"""
        return self.returned / self.invested if self.invested else float("nan")

    @property
    def profit(self) -> float:
        """収支（円）。"""
        return self.returned - self.invested

    def summary(self) -> str:
        return (
            f"レース数     : {self.n_races:,}\n"
            f"購入点数     : {self.n_bets:,}\n"
            f"的中数       : {self.n_hits:,}\n"
            f"的中率       : {self.hit_rate:.2%}\n"
            f"投資額       : {self.invested:,} 円\n"
            f"払戻額       : {self.returned:,.0f} 円\n"
            f"収支         : {self.profit:+,.0f} 円\n"
            f"回収率       : {self.roi:.2%}"
        )

    def __str__(self) -> str:  # print(result) で見やすく
        return self.summary()


def _prepare(df: pd.DataFrame, pred: np.ndarray | pd.Series) -> pd.DataFrame:
    """評価に必要な列だけを取り出した軽い DataFrame を作る。"""
    cols = config.resolve_columns(df)
    config.require(cols, "race_id", "rank", "win_odds", "date")

    out = pd.DataFrame({
        "race_id": df[cols["race_id"]].to_numpy(),
        "date": df[cols["date"]].to_numpy(),
        "rank": df[cols["rank"]].to_numpy(),
        "odds": pd.to_numeric(df[cols["win_odds"]], errors="coerce").to_numpy(),
        "pred": np.asarray(pred, dtype="float64"),
    })
    if "horse" in cols:
        out["horse"] = df[cols["horse"]].to_numpy()
    if "post" in cols:
        out["post"] = df[cols["post"]].to_numpy()

    # オッズ欠損の馬は賭けようがないので除外する
    return out.loc[out["odds"].notna() & (out["odds"] > 0)].reset_index(drop=True)


def normalize_prob_by_race(base: pd.DataFrame) -> pd.Series:
    """レース内で予測確率を正規化して合計 1 にする。

    LightGBM の binary 出力は「レース内で合計 1」にはなっていない。
    期待値を計算するには確率のスケールが正しい必要があるので、
    レース単位で割り算して足して 1 になるよう揃える。
    """
    total = base.groupby("race_id")["pred"].transform("sum")
    return base["pred"] / total.replace(0, np.nan)


def _settle(bets: pd.DataFrame, n_races: int) -> BacktestResult:
    """賭けた行の集合から収支を計算する。"""
    bets = bets.copy()
    bets["hit"] = (bets["rank"] == 1).astype(int)
    bets["payout"] = np.where(bets["hit"] == 1, bets["odds"] * BET_UNIT, 0.0)
    bets["profit"] = bets["payout"] - BET_UNIT

    return BacktestResult(
        n_races=n_races,
        n_bets=len(bets),
        n_hits=int(bets["hit"].sum()),
        invested=int(len(bets) * BET_UNIT),
        returned=float(bets["payout"].sum()),
        bets=bets.sort_values("date").reset_index(drop=True),
    )


def backtest_top_n(df: pd.DataFrame, pred: np.ndarray | pd.Series, n: int = 1,
                   min_odds: float | None = None,
                   max_odds: float | None = None) -> BacktestResult:
    """予測確率の上位 n 頭に単勝100円ずつ賭けたときの成績。

    min_odds / max_odds を指定するとオッズ帯で絞り込める
    （例: min_odds=10 で「10倍以上の高配当だけ買う」）。
    """
    base = _prepare(df, pred)
    n_races = base["race_id"].nunique()

    # レース内で予測確率の高い順に順位をつけ、上位 n 頭を選ぶ
    order = base.groupby("race_id")["pred"].rank(ascending=False, method="first")
    bets = base.loc[order <= n]

    if min_odds is not None:
        bets = bets.loc[bets["odds"] >= min_odds]
    if max_odds is not None:
        bets = bets.loc[bets["odds"] < max_odds]

    return _settle(bets, n_races)


def backtest_by_top_n(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                      ns: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """上位1頭・2頭・3頭…を横並びで比較する表を返す。"""
    rows = []
    for n in ns:
        r = backtest_top_n(df, pred, n=n)
        rows.append({
            "上位N頭": n, "購入点数": r.n_bets, "的中率": r.hit_rate,
            "回収率": r.roi, "収支": r.profit,
        })
    return pd.DataFrame(rows)


def odds_band_report(df: pd.DataFrame, pred: np.ndarray | pd.Series, n: int = 1,
                     bands: tuple[float, ...] = (1, 3, 5, 10, 20, 50, 1000)
                     ) -> pd.DataFrame:
    """オッズ帯ごとの成績を出す。どのゾーンで勝てているかを見るため。"""
    r = backtest_top_n(df, pred, n=n)
    bets = r.bets
    if bets.empty:
        return pd.DataFrame()

    bets = bets.assign(帯=pd.cut(bets["odds"], bins=list(bands), right=False))
    grouped = bets.groupby("帯", observed=True).agg(
        点数=("hit", "size"), 的中=("hit", "sum"), 払戻=("payout", "sum")
    )
    grouped["的中率"] = grouped["的中"] / grouped["点数"]
    grouped["回収率"] = grouped["払戻"] / (grouped["点数"] * BET_UNIT)
    return grouped.reset_index()


def plot_profit_curve(result: BacktestResult, title: str = "収支推移", ax=None):
    """収支の推移をグラフ化する。Colab ではそのまま表示される。"""
    import matplotlib.pyplot as plt

    if result.bets.empty:
        print("賭けた馬券がないのでグラフを描けません")
        return None

    bets = result.bets.sort_values("date")
    cum = bets["profit"].cumsum()

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.plot(bets["date"].to_numpy(), cum.to_numpy(), linewidth=1)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"{title}（回収率 {result.roi:.1%}／的中率 {result.hit_rate:.1%}）")
    ax.set_xlabel("日付")
    ax.set_ylabel("累積収支（円）")
    ax.grid(alpha=0.3)
    return ax


def random_baseline(df: pd.DataFrame, seed: int = 0, n: int = 1) -> BacktestResult:
    """ランダムに1頭選んだ場合のベースライン。控除率の分だけ負けるはず。"""
    rng = np.random.default_rng(seed)
    return backtest_top_n(df, rng.random(len(df)), n=n)


def favorite_baseline(df: pd.DataFrame, n: int = 1) -> BacktestResult:
    """1番人気（＝オッズ最低）を買い続けた場合のベースライン。

    モデルはこれを超えられて初めて意味がある。
    """
    cols = config.resolve_columns(df)
    odds = pd.to_numeric(df[cols["win_odds"]], errors="coerce")
    # オッズが低いほど「予測確率が高い」とみなす
    return backtest_top_n(df, 1.0 / odds.fillna(1e9), n=n)
