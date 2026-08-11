"""v3 依頼A〜C：レース選別で達成率を上げる。

■ 狙い
  全重賞を機械的に買うのではなく、「熱い当たりが出やすいレース」だけを買う。
  点数を増やすと熱い基準額も上がってしまうので、点数以外の軸で上げるしかない。

■ レース難易度指標
  すべて **そのレースの予測確率だけ** から計算する。
  過去の集計も検証期間の統計量も使わないので、構造的にリークしようがない。

      エントロピー       -Σ p log p        高い＝混戦
      正規化エントロピー  上記 ÷ log(頭数)   頭数の違いを吸収した混戦度
      確率標準偏差       ばらつき           低い＝横一線
      ジニ係数           不平等度           低い＝横一線
      1位2位差           突出度             小さい＝拮抗
      上位3頭確率和      上位の固まり具合    低い＝広く割れている
      最大確率           1番手の強さ
      出走頭数

  「モデルから見て混戦のレースは荒れやすく、荒れれば配当が伸びる」という仮説を検証する。

■ 判定は必ず信頼区間の下限で行う
  絞り込むと母数が減って区間が広がる。点推定が上がっても下限が下がるなら、
  それは「たまたま当たりが濃く出た区間を拾っただけ」の可能性が高い。
  本モジュールの表は常に達成率下限（Wilson）を併記し、
  下限で並べ替えられるようにしてある。

■ 多重比較について
  条件を探せば探すほど「偶然よく見える条件」が見つかる。
  そのため探索系の関数は **試した条件の数** を返り値に持たせている
  （`table.attrs["n_conditions"]`）。報告時は必ずこの数を添えること。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config, leakfree, trifecta

# 難易度指標の一覧（値が大きいほど「混戦」になる向きに符号を揃えてある）
DIFFICULTY_METRICS = [
    "エントロピー", "正規化エントロピー", "確率標準偏差", "ジニ係数",
    "1位2位差", "上位3頭確率和", "最大確率", "出走頭数",
]

# 「混戦なら大きくなる」指標か、「混戦なら小さくなる」指標か
HIGHER_IS_MESSIER = {
    "エントロピー": True, "正規化エントロピー": True, "出走頭数": True,
    "確率標準偏差": False, "ジニ係数": False, "1位2位差": False,
    "上位3頭確率和": False, "最大確率": False,
}


# ---------------------------------------------------------------------------
# 依頼A-1：レース難易度指標
# ---------------------------------------------------------------------------
def race_difficulty(df: pd.DataFrame, pred: np.ndarray | pd.Series) -> pd.DataFrame:
    """レースごとの難易度指標を計算する（1行1レース）。

    予測確率はレース内で合計1に正規化してから使う。
    LightGBM の binary 出力はレース単位で合計1にならないため、
    そのままではエントロピーが頭数やモデルの自信度に引っ張られてしまう。
    """
    cols = config.resolve_columns(df)

    base = pd.DataFrame({
        "レースID": leakfree.normalize_id(df[cols["race_id"]]).to_numpy(),
        "pred": np.asarray(pred, dtype="float64"),
    })
    g = base.groupby("レースID", sort=False)
    base["p"] = base["pred"] / g["pred"].transform("sum").replace(0, np.nan)
    base["r"] = g["p"].rank(ascending=False, method="first")

    # エントロピー: -Σ p log p（p=0 の項は 0 とみなす）
    p = base["p"].to_numpy()
    base["_plogp"] = np.where(p > 0, -p * np.log(p), 0.0)

    g2 = base.groupby("レースID", sort=False)
    race = g2.agg(
        出走頭数=("p", "size"),
        エントロピー=("_plogp", "sum"),
        確率標準偏差=("p", "std"),
        最大確率=("p", "max"),
    ).reset_index()

    # 頭数が違うとエントロピーの取りうる上限が変わるので log(頭数) で割る
    race["正規化エントロピー"] = race["エントロピー"] / np.log(race["出走頭数"].clip(lower=2))

    # 上位の確率（1位・2位・上位3頭合計）
    top = base.loc[base["r"] <= 3].pivot_table(
        index="レースID", columns="r", values="p", aggfunc="max"
    )
    top.columns = [f"p{int(c)}" for c in top.columns]
    race = race.merge(top.reset_index(), on="レースID", how="left")
    race["1位2位差"] = race.get("p1", np.nan) - race.get("p2", np.nan)
    race["上位3頭確率和"] = race[[c for c in ["p1", "p2", "p3"] if c in race.columns]].sum(axis=1)

    # ジニ係数: 昇順に並べた確率で G = 2Σ(i·p_i)/(n·Σp) - (n+1)/n
    base["_asc"] = base.groupby("レースID", sort=False)["p"].rank(
        ascending=True, method="first")
    weighted = (base["_asc"] * base["p"]).groupby(base["レースID"], sort=False).sum()
    race = race.merge(weighted.rename("_wsum").reset_index(), on="レースID", how="left")
    n = race["出走頭数"]
    race["ジニ係数"] = 2 * race["_wsum"] / n - (n + 1) / n

    drop = [c for c in ["_wsum", "p1", "p2", "p3"] if c in race.columns]
    return race.drop(columns=drop)


def attach_difficulty(race_table: pd.DataFrame, df: pd.DataFrame,
                      pred: np.ndarray | pd.Series) -> pd.DataFrame:
    """trifecta.build_race_table() の出力に難易度指標を貼り付ける。"""
    metrics = race_difficulty(df, pred)
    metrics = metrics.drop(columns=[c for c in ["出走頭数"] if c in race_table.columns])
    return race_table.merge(metrics, on="レースID", how="left")


# ---------------------------------------------------------------------------
# 依頼A-2：層別分析
# ---------------------------------------------------------------------------
def stratify(race: pd.DataFrame, metric: str, strategy: trifecta.Strategy,
             q: int = 4) -> pd.DataFrame:
    """指標で q 分位に層別し、各層の成績を出す。"""
    target = race.loc[race[metric].notna()].copy()
    if target.empty:
        return pd.DataFrame()

    labels = [f"Q{i + 1}" for i in range(q)]
    try:
        target["層"] = pd.qcut(target[metric], q, labels=labels, duplicates="drop")
    except ValueError:
        return pd.DataFrame()  # 分位に割れない（値がほぼ一定）

    rows = []
    for name, g in target.groupby("層", observed=True):
        res = trifecta.simulate(g, strategy)
        res.update({"指標": metric, "層": str(name),
                    "指標下限": float(g[metric].min()), "指標上限": float(g[metric].max())})
        rows.append(res)

    out = pd.DataFrame(rows)
    keep = ["指標", "層", "指標下限", "指標上限", "レース数", "的中率", "配当中央値",
            "達成率", "達成率下限", "達成率上限", "回収率"]
    return out[[c for c in keep if c in out.columns]]


def stratify_all(race: pd.DataFrame, strategy: trifecta.Strategy,
                 metrics: list[str] | None = None, q: int = 4) -> pd.DataFrame:
    """すべての難易度指標について層別表を作る（依頼A-2の出力）。"""
    metrics = metrics or [m for m in DIFFICULTY_METRICS if m in race.columns]
    tables = [stratify(race, m, strategy, q=q) for m in metrics]
    tables = [t for t in tables if not t.empty]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ---------------------------------------------------------------------------
# 依頼A-3：選別ルールのスイープ
# ---------------------------------------------------------------------------
def _select(race: pd.DataFrame, metric: str, keep: float, higher: bool) -> pd.DataFrame:
    """指標の上位（または下位）keep 割合のレースを残す。"""
    values = race[metric]
    if higher:
        cutoff = values.quantile(1 - keep)
        return race.loc[values >= cutoff]
    cutoff = values.quantile(keep)
    return race.loc[values <= cutoff]


def threshold_sweep(race: pd.DataFrame, metric: str, strategy: trifecta.Strategy,
                    keeps: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
                    higher: bool | None = None) -> pd.DataFrame:
    """買うレースを絞ったときの達成率がどう動くかを見る（依頼A-3）。

    higher=True なら「指標の高い方を残す」、False なら「低い方を残す」。
    省略時は HIGHER_IS_MESSIER に従って混戦側を残す。
    """
    higher = HIGHER_IS_MESSIER.get(metric, True) if higher is None else higher
    side = "高い順" if higher else "低い順"

    rows = []
    for keep in keeps:
        subset = race if keep >= 1.0 else _select(race, metric, keep, higher)
        res = trifecta.simulate(subset, strategy)
        res.update({"選別条件": "全レース" if keep >= 1.0 else f"{metric} {side} 上位{keep:.0%}",
                    "残す割合": keep})
        rows.append(res)

    out = pd.DataFrame(rows)
    keep_cols = ["選別条件", "残す割合", "レース数", "的中率", "配当中央値",
                 "達成率", "達成率下限", "達成率上限", "回収率"]
    return out[[c for c in keep_cols if c in out.columns]]


def sweep_all(race: pd.DataFrame, strategy: trifecta.Strategy,
              metrics: list[str] | None = None,
              keeps: tuple[float, ...] = (0.75, 0.5, 0.25)) -> pd.DataFrame:
    """全指標 × 全閾値を総当たりし、達成率下限の高い順に並べる。

    `attrs["n_conditions"]` に試した条件数が入る（多重比較の申告用）。
    """
    metrics = metrics or [m for m in DIFFICULTY_METRICS if m in race.columns]
    baseline = trifecta.simulate(race, strategy)

    rows = [{"選別条件": "全レース（基準）", "レース数": baseline.get("レース数", 0),
             "的中率": baseline.get("的中率"), "達成率": baseline.get("達成率"),
             "達成率下限": baseline.get("達成率下限"), "達成率上限": baseline.get("達成率上限"),
             "回収率": baseline.get("回収率")}]

    n_conditions = 0
    for metric in metrics:
        for higher in (True, False):
            for keep in keeps:
                subset = _select(race, metric, keep, higher)
                if len(subset) < 30:  # 母数が小さすぎる条件は見ない
                    continue
                res = trifecta.simulate(subset, strategy)
                n_conditions += 1
                side = "高い順" if higher else "低い順"
                rows.append({
                    "選別条件": f"{metric} {side} 上位{keep:.0%}",
                    "レース数": res.get("レース数"), "的中率": res.get("的中率"),
                    "達成率": res.get("達成率"), "達成率下限": res.get("達成率下限"),
                    "達成率上限": res.get("達成率上限"), "回収率": res.get("回収率"),
                })

    out = pd.DataFrame(rows).sort_values("達成率下限", ascending=False).reset_index(drop=True)
    out.attrs["n_conditions"] = n_conditions
    return out


# ---------------------------------------------------------------------------
# 依頼A-4：複合条件（最大2つまで）
# ---------------------------------------------------------------------------
def combo_search(race: pd.DataFrame, strategy: trifecta.Strategy,
                 metrics: list[str] | None = None,
                 keeps: tuple[float, ...] = (0.75, 0.5),
                 min_races: int = 50, top: int = 15) -> pd.DataFrame:
    """2指標の組み合わせを総当たりする（依頼A-4）。

    条件は最大2つまで。過学習を避けるため、
    母数 min_races 未満の条件は最初から候補にしない。
    `attrs["n_conditions"]` に試した組み合わせ数が入る。
    """
    metrics = metrics or [m for m in DIFFICULTY_METRICS if m in race.columns]

    rows = []
    n_conditions = 0
    for m1, m2 in itertools.combinations(metrics, 2):
        for k1, k2 in itertools.product(keeps, repeat=2):
            h1 = HIGHER_IS_MESSIER.get(m1, True)
            h2 = HIGHER_IS_MESSIER.get(m2, True)
            subset = _select(_select(race, m1, k1, h1), m2, k2, h2)
            if len(subset) < min_races:
                continue
            res = trifecta.simulate(subset, strategy)
            n_conditions += 1
            rows.append({
                "条件": f"{m1}{'高' if h1 else '低'}{k1:.0%} × {m2}{'高' if h2 else '低'}{k2:.0%}",
                "レース数": res.get("レース数"), "的中率": res.get("的中率"),
                "配当中央値": res.get("配当中央値"), "達成率": res.get("達成率"),
                "達成率下限": res.get("達成率下限"), "回収率": res.get("回収率"),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("達成率下限", ascending=False).head(top).reset_index(drop=True)
    out.attrs["n_conditions"] = n_conditions
    return out


# ---------------------------------------------------------------------------
# 依頼B：グレード別
# ---------------------------------------------------------------------------
def grade_breakdown(race: pd.DataFrame, strategy: trifecta.Strategy) -> pd.DataFrame:
    """G1 / G2 / G3 / G それぞれの成績（依頼B）。

    母数が小さい（G1は100レース程度）ので、必ず信頼区間とセットで見ること。
    点推定の差はほぼ確実に区間内に収まるはずで、
    「傾向の参考」以上の主張はできない。
    """
    if "格付け" not in race.columns:
        return pd.DataFrame()

    rows = []
    for grade in config.GRADED_VALUES:
        subset = race.loc[race["格付け"] == grade]
        if subset.empty:
            continue
        res = trifecta.simulate(subset, strategy)
        res["格付け"] = grade
        rows.append(res)

    # 比較用に重賞全体も入れる
    graded = race.loc[race["重賞"]] if "重賞" in race.columns else race
    if not graded.empty:
        res = trifecta.simulate(graded, strategy)
        res["格付け"] = "重賞全体"
        rows.append(res)

    out = pd.DataFrame(rows)
    keep = ["格付け", "買い方", "レース数", "的中数", "的中率", "配当中央値",
            "達成率", "達成率下限", "達成率上限", "回収率"]
    return out[[c for c in keep if c in out.columns]]


# ---------------------------------------------------------------------------
# 依頼C：熱い基準の感度分析
# ---------------------------------------------------------------------------
def threshold_sensitivity(race: pd.DataFrame, strategy: trifecta.Strategy,
                          thresholds: tuple[float, ...] = (30_000, 40_000, 50_000,
                                                           70_000, 100_000)
                          ) -> pd.DataFrame:
    """熱い基準額を変えたときの達成率（依頼C）。

    ⚠ 基準を下げれば達成率は必ず上がる。これは解決策ではなく現状把握。
      「10%を超えるのはどの基準額か」を知るための表。
    """
    rows = []
    for th in thresholds:
        res = trifecta.simulate(race, strategy, threshold=th)
        invest = res.get("投資/R", np.nan)
        rows.append({
            "買い方": strategy.name,
            "投資/R": invest,
            "基準額": int(th),
            "倍率": round(th / invest, 1) if invest else np.nan,
            "レース数": res.get("レース数"),
            "熱い当たり": res.get("熱い当たり"),
            "達成率": res.get("達成率"),
            "達成率下限": res.get("達成率下限"),
            "達成率上限": res.get("達成率上限"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 依頼A〜D をまとめて実行
# ---------------------------------------------------------------------------
def run_v3_analysis(race_table: pd.DataFrame, df: pd.DataFrame,
                    pred: np.ndarray | pd.Series,
                    strategies: list[trifecta.Strategy] | None = None,
                    verbose: bool = True) -> dict:
    """v3 の依頼A〜Dを一括で実行し、結果を辞書で返す。

    race_table は trifecta.build_race_table() の出力。
    分析は **重賞のみ** を対象にする（v3の最重要の発見に従う）。
    """
    race = attach_difficulty(race_table, df, pred)
    graded = race.loc[race["重賞"]] if "重賞" in race.columns else race

    main = trifecta.box(6)          # 達成率最高の買い方
    cheap = trifecta.formation(1, 4, 8)  # コスパの良い買い方
    strategies = strategies or trifecta.ALL_STRATEGIES

    out = {
        "race": race,
        "graded": graded,
        # 依頼D：買い方の総当たり（重賞のみ）
        "strategy_table": trifecta.strategy_table(graded, strategies),
        # 依頼A-2：層別
        "strata_box6": stratify_all(graded, main),
        "strata_f148": stratify_all(graded, cheap),
        # 依頼A-3：スイープ
        "sweep_box6": sweep_all(graded, main),
        "sweep_f148": sweep_all(graded, cheap),
        # 依頼A-4：複合条件
        "combo_box6": combo_search(graded, main),
        # 依頼B：グレード別
        "grade_box6": grade_breakdown(graded, main),
        "grade_f148": grade_breakdown(graded, cheap),
        # 依頼C：基準額の感度
        "sensitivity_box6": threshold_sensitivity(graded, main),
        "sensitivity_f148": threshold_sensitivity(
            graded, cheap, thresholds=(5_000, 10_000, 15_000, 20_000, 30_000)),
    }

    if verbose:
        _print_v3(out)
    return out


def _print_v3(out: dict) -> None:
    """v3の結果をまとめて表示する。"""
    fmt = trifecta.format_table

    print("\n=== 依頼D：買い方の総当たり（重賞のみ） ===")
    print(fmt(out["strategy_table"]).to_string(index=False))

    for key, label in [("strata_box6", "上位6頭BOX"), ("strata_f148", "F1→4→8")]:
        if not out[key].empty:
            print(f"\n=== 依頼A-2：層別分析（{label}） ===")
            print(fmt(out[key]).to_string(index=False))

    for key, label in [("sweep_box6", "上位6頭BOX"), ("sweep_f148", "F1→4→8")]:
        table = out[key]
        if not table.empty:
            print(f"\n=== 依頼A-3：選別スイープ（{label}／達成率下限の高い順） ===")
            print(fmt(table).head(15).to_string(index=False))
            print(f"    ※試した条件数: {table.attrs.get('n_conditions', 0)}")

    if not out["combo_box6"].empty:
        table = out["combo_box6"]
        print("\n=== 依頼A-4：複合条件（上位6頭BOX） ===")
        print(fmt(table).to_string(index=False))
        print(f"    ※試した組み合わせ数: {table.attrs.get('n_conditions', 0)}")

    for key, label in [("grade_box6", "上位6頭BOX"), ("grade_f148", "F1→4→8")]:
        if not out[key].empty:
            print(f"\n=== 依頼B：グレード別（{label}） ===")
            print(fmt(out[key]).to_string(index=False))

    for key, label in [("sensitivity_box6", "上位6頭BOX"), ("sensitivity_f148", "F1→4→8")]:
        print(f"\n=== 依頼C：熱い基準の感度（{label}） ===")
        print(fmt(out[key]).to_string(index=False))


def best_candidates(out: dict, min_lower: float = 0.10,
                    min_races: int = 50) -> pd.DataFrame:
    """達成率の**信頼区間下限**が目標を超えた条件だけを抜き出す（依頼書セクション6）。

    空の DataFrame が返ったら「見つからなかった」ということ。
    点推定ではなく下限で判定しているので、
    母数が少ないのに点推定だけ高い条件は自動的に落ちる。
    """
    frames = []
    for key in ["sweep_box6", "sweep_f148", "combo_box6"]:
        table = out.get(key)
        if table is None or table.empty:
            continue
        t = table.copy()
        t["出所"] = key
        col = "選別条件" if "選別条件" in t.columns else "条件"
        t = t.rename(columns={col: "条件"})
        frames.append(t[["出所", "条件", "レース数", "達成率", "達成率下限", "回収率"]])

    # 買い方の総当たり（選別なし）も候補に含める
    st = out.get("strategy_table")
    if st is not None and not st.empty:
        t = st.rename(columns={"買い方": "条件"}).copy()
        t["出所"] = "strategy_table"
        frames.append(t[["出所", "条件", "レース数", "達成率", "達成率下限", "回収率"]])

    if not frames:
        return pd.DataFrame()

    allc = pd.concat(frames, ignore_index=True)
    hit = allc.loc[(allc["達成率下限"] >= min_lower) & (allc["レース数"] >= min_races)]
    return hit.sort_values("達成率下限", ascending=False).reset_index(drop=True)
