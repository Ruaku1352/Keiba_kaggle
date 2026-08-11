"""依頼C：3連単の買い方シミュレーションと重賞限定の検証。

■ このプロジェクトの目的関数（v2で変更された点）
  回収率の最大化ではなく **「熱い当たりが出る確率」の最大化**。

      予算1,000円 → 10,000円
      予算3,000円 → 20,000円
      予算5,000円 → 30,000円
      予算10,000円 → 50,000円

  この基準を超える払戻が出たレースの割合を「達成率」と呼び、これを最大化する。
  回収率60%でも達成率が高ければ勝ちという設計。

■ シミュレーションの計算量について
  3連単のBOXは点数が多い（5頭BOXで60点）が、**当たる組み合わせは必ず1つだけ**
  （実際の1-2-3着の並び）なので、全点を展開する必要はない。

      当たり判定 = 実際の1着馬の予測順位 ≤ n1
                かつ 2着馬の予測順位 ≤ n2
                かつ 3着馬の予測順位 ≤ n3

  これだけで済むので、22,956レースでも一瞬で終わる。

■ 点数の数え方
  上位n1頭から1着、上位n2頭から2着、上位n3頭から3着を選ぶ（n1≤n2≤n3、入れ子）。
  1着に n1 通り、2着は1着で使った1頭を除いて n2-1 通り、
  3着はさらに2頭除いて n3-2 通り。よって

      点数 = n1 × (n2 - 1) × (n3 - 2)

  BOX(k) は n1=n2=n3=k なので k(k-1)(k-2)。3頭BOX=6点、4頭BOX=24点、5頭BOX=60点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, leakfree

BET_UNIT = 100  # 1点100円


# ---------------------------------------------------------------------------
# 買い方の定義
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Strategy:
    """3連単の買い方。上位 n1/n2/n3 頭から1着/2着/3着を選ぶ。"""

    name: str
    n1: int
    n2: int
    n3: int

    @property
    def points(self) -> int:
        """点数（出走頭数が足りている場合）。"""
        return max(self.n1 * (self.n2 - 1) * (self.n3 - 2), 0)


def box(k: int) -> Strategy:
    """上位k頭のBOX買い。"""
    return Strategy(f"上位{k}頭BOX", k, k, k)


def formation(n1: int, n2: int, n3: int) -> Strategy:
    """フォーメーション（1着候補n1頭・2着候補n2頭・3着候補n3頭）。"""
    return Strategy(f"F{n1}→{n2}→{n3}", n1, n2, n3)


# 標準の比較セット（進捗まとめ 3.3 の表に対応）
DEFAULT_STRATEGIES = [
    formation(1, 3, 4), box(3), formation(1, 4, 6), formation(1, 4, 8),
    box(4), formation(2, 5, 6), formation(2, 5, 8), box(5),
    formation(2, 6, 10), box(6),
]


# ---------------------------------------------------------------------------
# 熱さの基準
# ---------------------------------------------------------------------------
def hot_threshold(investment: float) -> int:
    """投資額に対する「熱い」基準額を返す。

    開発者の設定した4段階（1,000/3,000/5,000/10,000円）のうち、
    投資額に **最も近い** 段階の基準を使う。
    例）投資2,400円 → 3,000円の段が近い → 基準20,000円
    """
    tiers = np.array(sorted(config.HOT_CRITERIA))
    nearest = tiers[np.argmin(np.abs(tiers - investment))]
    return config.HOT_CRITERIA[int(nearest)]


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """達成率の信頼区間（Wilson score interval）。

    達成率は数%と非常に小さいので、教科書どおりの
    p ± 1.96*sqrt(p(1-p)/n) だと下限が負に飛び出して意味を失う。
    Wilson 区間は 0 件でも [0, 上限] という妥当な区間を返すので、
    重賞のように母数が小さい場合はこちらが適切。
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def normal_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """依頼書に書かれていた素朴な二項近似（比較用に残してある）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


# ---------------------------------------------------------------------------
# レース単位の表を作る
# ---------------------------------------------------------------------------
def build_race_table(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                     payout_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """1行1レースの検証用テーブルを作る。

    列: レースID, 日付, 出走頭数, 1着予測順位, 2着予測順位, 3着予測順位,
        3連単払戻, 重賞
    「1着予測順位」= 実際に1着だった馬が、モデルの予測で何番目に評価されていたか。
    """
    cols = config.resolve_columns(df)
    race_id, rank, date = cols["race_id"], cols["rank"], cols["date"]

    base = pd.DataFrame({
        # payout 側と型を揃えるため normalize_id を通す
        "レースID": leakfree.normalize_id(df[race_id]).to_numpy(),
        "日付": df[date].to_numpy(),
        "着順": df[rank].to_numpy(),
        "pred": np.asarray(pred, dtype="float64"),
    })
    if "grade" in cols:
        base["_grade"] = df[cols["grade"]].to_numpy()

    # 予測確率の高い順に 1,2,3... と順位を振る
    base["予測順位"] = base.groupby("レースID")["pred"].rank(
        ascending=False, method="first"
    ).astype("int16")

    race = base.groupby("レースID", sort=False).agg(
        日付=("日付", "first"), 出走頭数=("着順", "size")
    ).reset_index()

    # 実際の1〜3着馬の予測順位を横に並べる
    for k in (1, 2, 3):
        # 同着があると1レースに複数行出るため min で1つに潰す
        s = (base.loc[base["着順"] == k]
             .groupby("レースID")["予測順位"].min().rename(f"{k}着予測順位"))
        race = race.merge(s, on="レースID", how="left")

    if "_grade" in base.columns:
        grade = base.groupby("レースID")["_grade"].first().rename("格付け")
        race = race.merge(grade, on="レースID", how="left")
        race["重賞"] = race["格付け"].isin(config.GRADED_VALUES)
    else:
        race["重賞"] = False

    if payout_df is not None:
        pay = payout_df.rename(columns={"レースID": "_pid"})
        race = race.merge(pay, left_on="レースID", right_on="_pid", how="left")
        race = race.drop(columns=[c for c in ["_pid"] if c in race.columns])
    else:
        race["3連単払戻"] = np.nan

    # 3着まで揃っていないレース（出走3頭未満・同着で欠けるなど）は検証対象外
    race = race.dropna(subset=["1着予測順位", "2着予測順位", "3着予測順位"])
    return race.sort_values("日付").reset_index(drop=True)


# ---------------------------------------------------------------------------
# シミュレーション
# ---------------------------------------------------------------------------
def simulate(race: pd.DataFrame, strategy: Strategy,
             require_payout: bool = True) -> dict:
    """1つの買い方について成績を計算する。"""
    r = race
    if require_payout:
        r = r.loc[r["3連単払戻"].notna()]
    if r.empty:
        return {"買い方": strategy.name, "レース数": 0}

    m = r["出走頭数"].to_numpy()
    # 出走頭数が候補数に満たない場合は、実際に買える点数まで縮める
    n1 = np.minimum(strategy.n1, m)
    n2 = np.minimum(strategy.n2, m)
    n3 = np.minimum(strategy.n3, m)
    points = np.clip(n1 * (n2 - 1) * (n3 - 2), 0, None)

    hit = (
        (r["1着予測順位"].to_numpy() <= n1)
        & (r["2着予測順位"].to_numpy() <= n2)
        & (r["3着予測順位"].to_numpy() <= n3)
    )

    invest = points * BET_UNIT
    payout = np.where(hit, r["3連単払戻"].to_numpy(), 0.0)

    # 「熱い」基準はその買い方の1レースあたり投資額で決まる
    typical_invest = float(np.median(invest))
    threshold = hot_threshold(typical_invest)
    hot = payout >= threshold

    n_races = len(r)
    n_hits = int(hit.sum())
    n_hot = int(hot.sum())
    lo, hi = wilson_interval(n_hot, n_races)

    hit_payouts = payout[hit]
    return {
        "買い方": strategy.name,
        "点数": int(np.median(points)),
        "投資/R": int(typical_invest),
        "レース数": n_races,
        "的中数": n_hits,
        "的中率": n_hits / n_races,
        "配当中央値": float(np.median(hit_payouts)) if n_hits else np.nan,
        "熱い基準": threshold,
        "熱い当たり": n_hot,
        "達成率": n_hot / n_races,
        "達成率下限": lo,
        "達成率上限": hi,
        "回収率": float(payout.sum() / invest.sum()) if invest.sum() else np.nan,
        "収支": float(payout.sum() - invest.sum()),
    }


def strategy_table(race: pd.DataFrame, strategies: list[Strategy] | None = None,
                   graded_only: bool = False) -> pd.DataFrame:
    """買い方の総当たり表を作る（依頼Cのメイン出力）。

    graded_only=True で重賞（G1/G2/G3/G）だけに絞る。
    学習データは絞らず、**評価だけ**を重賞に絞るのがポイント。
    """
    strategies = strategies or DEFAULT_STRATEGIES
    target = race.loc[race["重賞"]] if graded_only else race
    rows = [simulate(target, s) for s in strategies]
    out = pd.DataFrame(rows)
    return out.sort_values("投資/R").reset_index(drop=True) if "投資/R" in out else out


def compare_graded_vs_flat(race: pd.DataFrame,
                           strategies: list[Strategy] | None = None) -> pd.DataFrame:
    """重賞と平場を並べて比較する。

    重賞は出走馬の実力が拮抗して人気が割れやすいので、
    平場と傾向が違う可能性がある。母数が小さいため信頼区間を必ず見ること。
    """
    strategies = strategies or DEFAULT_STRATEGIES
    graded = strategy_table(race.loc[race["重賞"]], strategies)
    flat = strategy_table(race.loc[~race["重賞"]], strategies)

    keep = ["買い方", "点数", "レース数", "的中率", "配当中央値", "達成率",
            "達成率下限", "達成率上限", "回収率"]
    merged = graded[keep].merge(
        flat[keep].drop(columns=["点数"]), on="買い方", suffixes=("_重賞", "_平場")
    )
    # 重賞の方が達成率が高ければプラス
    merged["達成率差"] = merged["達成率_重賞"] - merged["達成率_平場"]
    # 信頼区間が重ならない＝統計的に差があると言える
    merged["有意"] = (
        (merged["達成率下限_重賞"] > merged["達成率上限_平場"])
        | (merged["達成率上限_重賞"] < merged["達成率下限_平場"])
    )
    return merged


def simulate_gap_box(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                     payout_df: pd.DataFrame | None = None,
                     n_gap: int = 2, graded_only: bool = False) -> dict:
    """乖離ベースの買い方（v2 セクション3.4 の再現・改善判定用）。

    軸1頭 = 予測1位。残り n_gap 頭 = 乖離 gap の上位。この3頭BOX（6点）。

        gap = 予測確率 ÷ 市場勝率 = 予測確率 × 単勝オッズ

    「モデルは高く評価しているのに人気がない馬」を拾う狙い。
    v2時点では達成率0.065%（上位3頭BOXの1/15）で完敗している。
    **新特徴量でここが改善するかが最重要の判定基準。**
    """
    cols = config.resolve_columns(df)
    race_id, rank, odds = cols["race_id"], cols["rank"], cols["win_odds"]

    base = pd.DataFrame({
        "レースID": leakfree.normalize_id(df[race_id]).to_numpy(),
        "着順": df[rank].to_numpy(),
        "pred": np.asarray(pred, dtype="float64"),
        "odds": pd.to_numeric(df[odds], errors="coerce").to_numpy(),
    })
    if "grade" in cols:
        base["重賞"] = pd.Series(df[cols["grade"]].to_numpy()).isin(config.GRADED_VALUES).to_numpy()
    else:
        base["重賞"] = False
    if graded_only:
        base = base.loc[base["重賞"]]

    base = base.loc[base["odds"].notna() & (base["odds"] > 0)]
    if base.empty:
        return {"買い方": f"乖離BOX(1+{n_gap})", "レース数": 0}

    g = base.groupby("レースID", sort=False)
    base["予測順位"] = g["pred"].rank(ascending=False, method="first")
    # 乖離 = 予測確率 × オッズ（市場勝率の逆数を掛けるのと同じ）
    base["gap"] = base["pred"] * base["odds"]

    is_axis = base["予測順位"] == 1
    # 軸を除いた中での乖離順位
    base["gap順位"] = base.loc[~is_axis].groupby("レースID", sort=False)["gap"].rank(
        ascending=False, method="first"
    )
    selected = is_axis | (base["gap順位"] <= n_gap)
    base["選択"] = selected.astype(int)

    # 実際の1〜3着が、選んだ3頭とちょうど一致すれば的中（BOXなので順序は不問）
    top3 = base.loc[base["着順"] <= 3]
    matched = top3.groupby("レースID")["選択"].sum()
    n_top3 = top3.groupby("レースID")["選択"].size()
    hit = ((matched == 3) & (n_top3 == 3))

    points = 1 + n_gap
    points = points * (points - 1) * (points - 2)  # 3頭BOX = 6点
    invest_per_race = points * BET_UNIT

    race = pd.DataFrame({"レースID": hit.index, "hit": hit.to_numpy()})
    if payout_df is not None:
        race = race.merge(payout_df, on="レースID", how="left")
        race = race.dropna(subset=["3連単払戻"])
    else:
        race["3連単払戻"] = np.nan

    if race.empty:
        return {"買い方": f"乖離BOX(1+{n_gap})", "レース数": 0}

    payout = np.where(race["hit"].to_numpy(), race["3連単払戻"].to_numpy(), 0.0)
    threshold = hot_threshold(invest_per_race)
    n_races = len(race)
    n_hits = int(race["hit"].sum())
    n_hot = int((payout >= threshold).sum())
    lo, hi = wilson_interval(n_hot, n_races)

    return {
        "買い方": f"乖離BOX(1+{n_gap})",
        "点数": points,
        "投資/R": invest_per_race,
        "レース数": n_races,
        "的中数": n_hits,
        "的中率": n_hits / n_races,
        "配当中央値": float(np.median(payout[payout > 0])) if n_hits else np.nan,
        "熱い基準": threshold,
        "熱い当たり": n_hot,
        "達成率": n_hot / n_races,
        "達成率下限": lo,
        "達成率上限": hi,
        "回収率": float(payout.sum() / (invest_per_race * n_races)),
        "収支": float(payout.sum() - invest_per_race * n_races),
    }


# v2 時点のベースライン（改善したかを機械的に判定するため）
V2_BASELINE = {
    "valid AUC": 0.7787,
    "上位3頭BOX 達成率": 0.01015,
    "上位5頭BOX 達成率": 0.02919,
    "乖離BOX 達成率": 0.00065,
}


def verdict(auc: float | None, table: pd.DataFrame, gap_result: dict) -> pd.DataFrame:
    """v2 のベースラインと比べて改善したかを判定する（依頼書セクション6）。"""
    def achievement(name: str) -> float:
        row = table.loc[table["買い方"] == name]
        return float(row["達成率"].iloc[0]) if len(row) else float("nan")

    rows = [
        ("valid AUC", auc, V2_BASELINE["valid AUC"]),
        ("上位3頭BOX 達成率", achievement("上位3頭BOX"), V2_BASELINE["上位3頭BOX 達成率"]),
        ("上位5頭BOX 達成率", achievement("上位5頭BOX"), V2_BASELINE["上位5頭BOX 達成率"]),
        ("乖離BOX 達成率", gap_result.get("達成率", float("nan")),
         V2_BASELINE["乖離BOX 達成率"]),
    ]
    out = pd.DataFrame(rows, columns=["指標", "今回", "v2ベースライン"])
    out["改善"] = out["今回"] > out["v2ベースライン"]
    out["差"] = out["今回"] - out["v2ベースライン"]
    return out


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """パーセント表記に整えて見やすくする（表示用）。"""
    out = table.copy()
    for col in out.columns:
        if any(k in col for k in ("率", "下限", "上限")):
            out[col] = (out[col] * 100).round(3)
    return out
