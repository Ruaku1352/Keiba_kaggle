"""v3（レース選別・グレード別・基準額感度・3連複）のテスト。

難易度指標は「そのレースの予測確率だけ」から作るので、
検証すべきは主に **計算式が定義どおりか** と
**選別・信頼区間の扱いが正しいか** の2点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocess, selection, trifecta  # noqa: E402
from test_features import make_df  # noqa: E402


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------
def make_race_table(n_races: int = 400, seed: int = 0) -> pd.DataFrame:
    """検証用の1行1レーステーブルを直接組み立てる。"""
    rng = np.random.default_rng(seed)
    grades = rng.choice(["G1", "G2", "G3", "G", None], size=n_races,
                        p=[0.05, 0.05, 0.1, 0.05, 0.75])
    return pd.DataFrame({
        "レースID": [f"R{i:04d}" for i in range(n_races)],
        "日付": pd.date_range("2020-01-01", periods=n_races, freq="D"),
        "出走頭数": rng.integers(8, 19, n_races),
        "1着予測順位": rng.integers(1, 9, n_races),
        "2着予測順位": rng.integers(1, 9, n_races),
        "3着予測順位": rng.integers(1, 9, n_races),
        "格付け": grades,
        "重賞": [g is not None for g in grades],
        "3連単払戻": np.round(np.exp(rng.normal(9.0, 1.5, n_races))),
        "3連複払戻": np.round(np.exp(rng.normal(7.0, 1.2, n_races))),
        # 難易度指標（選別テスト用にダミーで入れておく）
        "エントロピー": rng.random(n_races),
        "正規化エントロピー": rng.random(n_races),
        "確率標準偏差": rng.random(n_races),
        "ジニ係数": rng.random(n_races),
        "1位2位差": rng.random(n_races),
        "上位3頭確率和": rng.random(n_races),
        "最大確率": rng.random(n_races),
    })


# ---------------------------------------------------------------------------
# 依頼A-1：難易度指標
# ---------------------------------------------------------------------------
def test_difficulty_uses_normalized_probabilities():
    """レース内で正規化した確率から計算していること（合計1が前提）。"""
    df = pd.DataFrame({
        "レースID": ["A"] * 4 + ["B"] * 4,
        "レース日付": pd.to_datetime(["2020-01-01"] * 8),
        "馬名": list("abcdefgh"),
        "馬番": [1, 2, 3, 4, 1, 2, 3, 4],
        "着順": [1, 2, 3, 4, 1, 2, 3, 4],
    })
    # A: 完全に横一線 / B: 1頭が突出
    pred = np.array([0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1])
    out = selection.race_difficulty(df, pred).set_index("レースID")

    # 横一線なら エントロピー = log(4)、ジニ = 0、最大確率 = 0.25
    assert out.loc["A", "エントロピー"] == pytest.approx(np.log(4))
    assert out.loc["A", "正規化エントロピー"] == pytest.approx(1.0)
    assert out.loc["A", "ジニ係数"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc["A", "最大確率"] == pytest.approx(0.25)
    assert out.loc["A", "1位2位差"] == pytest.approx(0.0, abs=1e-9)

    # 突出しているほうがエントロピーは低く、ジニは高い
    assert out.loc["B", "エントロピー"] < out.loc["A", "エントロピー"]
    assert out.loc["B", "ジニ係数"] > out.loc["A", "ジニ係数"]
    assert out.loc["B", "最大確率"] == pytest.approx(0.7)


def test_difficulty_scale_invariance():
    """予測確率を定数倍しても指標が変わらないこと（正規化しているため）。"""
    df = pd.DataFrame({
        "レースID": ["A"] * 5,
        "レース日付": pd.to_datetime(["2020-01-01"] * 5),
        "馬名": list("abcde"), "馬番": [1, 2, 3, 4, 5], "着順": [1, 2, 3, 4, 5],
    })
    pred = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    a = selection.race_difficulty(df, pred)
    b = selection.race_difficulty(df, pred * 3.0)
    for col in ["エントロピー", "ジニ係数", "最大確率", "上位3頭確率和"]:
        assert a[col].iloc[0] == pytest.approx(b[col].iloc[0], rel=1e-9)


def test_difficulty_top3_sum():
    """上位3頭確率和が、正規化確率の上位3つの合計になっていること。"""
    df = pd.DataFrame({
        "レースID": ["A"] * 5,
        "レース日付": pd.to_datetime(["2020-01-01"] * 5),
        "馬名": list("abcde"), "馬番": [1, 2, 3, 4, 5], "着順": [1, 2, 3, 4, 5],
    })
    pred = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    out = selection.race_difficulty(df, pred)
    assert out["上位3頭確率和"].iloc[0] == pytest.approx(0.8)


def test_attach_difficulty_keeps_rows():
    """レーステーブルに指標を貼っても行数が変わらないこと。"""
    df = preprocess.basic_clean(make_df(n_races=40))
    rng = np.random.default_rng(0)
    pred = rng.random(len(df))
    payout = pd.DataFrame({
        "レースID": df["レースID"].astype(str).unique(), "3連単払戻": 8000.0})
    race = trifecta.build_race_table(df, pred, payout)
    out = selection.attach_difficulty(race, df, pred)
    assert len(out) == len(race)
    assert out["エントロピー"].notna().all()


# ---------------------------------------------------------------------------
# 依頼A-2/A-3：層別・スイープ
# ---------------------------------------------------------------------------
def test_stratify_partitions_all_races():
    """4分位の層別で、全レースがどこかの層に入ること。"""
    race = make_race_table()
    table = selection.stratify(race, "エントロピー", trifecta.box(6), q=4)
    assert len(table) == 4
    assert table["レース数"].sum() == len(race)
    # 層の境界が単調に上がっている
    assert table["指標下限"].is_monotonic_increasing


def test_sweep_keeps_expected_fraction():
    """スイープで残るレース数が、指定した割合とおおむね一致すること。"""
    race = make_race_table(n_races=400)
    table = selection.threshold_sweep(race, "エントロピー", trifecta.box(6),
                                      keeps=(1.0, 0.5, 0.25))
    assert table.loc[0, "レース数"] == 400
    assert table.loc[1, "レース数"] == pytest.approx(200, abs=5)
    assert table.loc[2, "レース数"] == pytest.approx(100, abs=5)


def test_sweep_direction():
    """higher=True なら指標の高い側、False なら低い側が残ること。"""
    race = make_race_table(n_races=200)
    high = selection._select(race, "エントロピー", 0.25, higher=True)
    low = selection._select(race, "エントロピー", 0.25, higher=False)
    assert high["エントロピー"].min() > low["エントロピー"].max()


def test_sweep_all_reports_condition_count():
    """試した条件数が attrs に記録されること（多重比較の申告用）。"""
    race = make_race_table()
    table = selection.sweep_all(race, trifecta.box(6))
    assert table.attrs["n_conditions"] > 0
    assert table.loc[0, "選別条件"] is not None
    # 達成率下限の降順に並んでいる
    lower = table["達成率下限"].dropna().to_numpy()
    assert np.all(np.diff(lower) <= 1e-12)


def test_combo_search_limits_to_two_conditions():
    """複合条件が2指標までで、母数の小さい条件を除外していること。"""
    race = make_race_table(n_races=600)
    table = selection.combo_search(race, trifecta.box(6), min_races=50)
    assert table.attrs["n_conditions"] > 0
    if not table.empty:
        assert (table["レース数"] >= 50).all()
        assert table["条件"].str.count("×").max() == 1  # 条件は2つまで


def test_best_candidates_uses_lower_bound():
    """成功判定が点推定ではなく信頼区間の下限で行われること。"""
    out = {
        "sweep_box6": pd.DataFrame({
            "選別条件": ["条件A", "条件B"],
            "レース数": [60, 60],
            "達成率": [0.30, 0.11],       # A の方が点推定は高い
            "達成率下限": [0.05, 0.12],   # が、下限は B の方が高い
            "回収率": [0.6, 0.7],
        }),
    }
    best = selection.best_candidates(out, min_lower=0.10, min_races=50)
    assert list(best["条件"]) == ["条件B"]  # 点推定30%のAは下限が低いので落ちる


def test_best_candidates_empty_when_nothing_passes():
    """条件を満たすものがなければ空を返すこと（「見つからなかった」を明示）。"""
    out = {"sweep_box6": pd.DataFrame({
        "選別条件": ["条件A"], "レース数": [500], "達成率": [0.05],
        "達成率下限": [0.03], "回収率": [0.6]})}
    assert selection.best_candidates(out, min_lower=0.10).empty


# ---------------------------------------------------------------------------
# 依頼B：グレード別
# ---------------------------------------------------------------------------
def test_grade_breakdown_covers_all_grades():
    """G1/G2/G3/G と重賞全体が並ぶこと。"""
    race = make_race_table(n_races=600)
    table = selection.grade_breakdown(race, trifecta.box(6))
    assert "重賞全体" in set(table["格付け"])
    assert {"G1", "G2", "G3", "G"} <= set(table["格付け"])
    # 各グレードのレース数の合計 = 重賞全体
    total = table.loc[table["格付け"] == "重賞全体", "レース数"].iloc[0]
    parts = table.loc[table["格付け"] != "重賞全体", "レース数"].sum()
    assert parts == total
    assert (table["達成率下限"] <= table["達成率"]).all()
    assert (table["達成率"] <= table["達成率上限"]).all()


# ---------------------------------------------------------------------------
# 依頼C：基準額の感度
# ---------------------------------------------------------------------------
def test_threshold_sensitivity_is_monotonic():
    """基準額を上げると達成率は必ず下がる（単調性）。"""
    race = make_race_table(n_races=500)
    table = selection.threshold_sensitivity(
        race, trifecta.box(6), thresholds=(10_000, 30_000, 50_000, 100_000))
    rates = table["達成率"].to_numpy()
    assert np.all(np.diff(rates) <= 1e-12)
    # 倍率 = 基準額 ÷ 1レースあたり投資額
    assert table.loc[0, "倍率"] == round(10_000 / table.loc[0, "投資/R"], 1)


def test_threshold_override_in_simulate():
    """simulate に threshold を渡すと基準額が上書きされること。"""
    race = make_race_table(n_races=100)
    auto = trifecta.simulate(race, trifecta.box(6))
    forced = trifecta.simulate(race, trifecta.box(6), threshold=1_000_000)
    assert auto["熱い基準"] == 50_000        # 投資12,000円 → 10,000円の段
    assert forced["熱い基準"] == 1_000_000
    assert forced["熱い当たり"] <= auto["熱い当たり"]


# ---------------------------------------------------------------------------
# 依頼D：買い方のバリエーション
# ---------------------------------------------------------------------------
def test_trio_point_counts():
    """3連複BOXの点数が C(k,3) になること。"""
    assert trifecta.trio(4).points == 4
    assert trifecta.trio(5).points == 10
    assert trifecta.trio(6).points == 20
    assert trifecta.trio(8).points == 56
    # 同じk頭でも3連単の1/6の点数
    assert trifecta.trio(6).points * 6 == trifecta.box(6).points


def test_trio_hits_regardless_of_order():
    """3連複は着順不問。上位k頭に1〜3着が入っていれば的中。"""
    race = pd.DataFrame({
        "レースID": ["A", "B"],
        "日付": pd.to_datetime(["2020-01-01"] * 2),
        "出走頭数": [16, 16],
        # A: 1〜3着が予測4位以内（順序はバラバラ）
        # B: 3着が予測7位 → 4頭では外れ
        "1着予測順位": [3, 1],
        "2着予測順位": [1, 2],
        "3着予測順位": [4, 7],
        "重賞": [True, True],
        "3連単払戻": [50000.0, 50000.0],
        "3連複払戻": [8000.0, 8000.0],
    })
    trio4 = trifecta.simulate(race, trifecta.trio(4))
    assert trio4["的中数"] == 1
    assert trio4["点数"] == 4 and trio4["投資/R"] == 400

    # 同じ4頭でも3連単なら着順まで合わないと当たらない（A は 3-1-4 なので外れ）
    tri4 = trifecta.simulate(race, trifecta.box(4))
    assert tri4["的中数"] == 1  # A は上位4頭に収まっているので3連単BOXでも的中
    assert tri4["点数"] == 24

    trio7 = trifecta.simulate(race, trifecta.trio(7))
    assert trio7["的中数"] == 2  # B も7頭なら入る


def test_trio_uses_trio_payout():
    """3連複の払戻列が使われていること（3連単の払戻ではない）。"""
    race = pd.DataFrame({
        "レースID": ["A"], "日付": pd.to_datetime(["2020-01-01"]),
        "出走頭数": [16], "1着予測順位": [1], "2着予測順位": [2], "3着予測順位": [3],
        "重賞": [True], "3連単払戻": [99999.0], "3連複払戻": [1234.0],
    })
    res = trifecta.simulate(race, trifecta.trio(5))
    assert res["券種"] == "3連複"
    assert res["配当中央値"] == 1234.0


def test_missing_payout_column_is_skipped():
    """3連複の払戻列がないデータでも例外にせずスキップすること。"""
    race = pd.DataFrame({
        "レースID": ["A"], "日付": pd.to_datetime(["2020-01-01"]),
        "出走頭数": [16], "1着予測順位": [1], "2着予測順位": [2], "3着予測順位": [3],
        "重賞": [True], "3連単払戻": [5000.0],
    })
    res = trifecta.simulate(race, trifecta.trio(5))
    assert res["レース数"] == 0


def test_multi2_is_nested_formation():
    """軸2頭マルチ = n1=n2=2 のフォーメーション。点数は 2×1×(n3-2)。"""
    s = trifecta.multi2(8)
    assert (s.n1, s.n2, s.n3) == (2, 2, 8)
    assert s.points == 2 * 1 * 6


def test_all_strategies_have_unique_names():
    """総当たりリストに名前の重複がないこと（表が壊れるため）。"""
    names = [s.name for s in trifecta.ALL_STRATEGIES]
    assert len(names) == len(set(names))


def test_strategy_table_mixes_kinds():
    """3連単と3連複が同じ表に並ぶこと。"""
    race = make_race_table(n_races=300)
    table = trifecta.strategy_table(race, trifecta.ALL_STRATEGIES)
    assert set(table["券種"]) == {"3連単", "3連複"}
    assert (table["レース数"] > 0).all()


# ---------------------------------------------------------------------------
# 全体
# ---------------------------------------------------------------------------
def test_run_v3_analysis_produces_all_reports():
    """v3の分析が一通り動き、必要な表が揃うこと。"""
    df = make_df(n_races=300)
    ids = df["レースID"].unique()
    graded = set(ids[::3])
    df["リステッド・重賞競走"] = [
        "G1" if r in graded else None for r in df["レースID"]]
    df = preprocess.basic_clean(df)

    rng = np.random.default_rng(2)
    pred = rng.random(len(df))
    payout = pd.DataFrame({
        "レースID": df["レースID"].astype(str).unique(),
        "3連単払戻": np.round(np.exp(rng.normal(9, 1.5, len(ids)))),
        "3連複払戻": np.round(np.exp(rng.normal(7, 1.2, len(ids)))),
    })
    race = trifecta.build_race_table(df, pred, payout)
    out = selection.run_v3_analysis(race, df, pred, verbose=False)

    for key in ["strategy_table", "strata_box6", "sweep_box6", "grade_box6",
                "sensitivity_box6"]:
        assert key in out and not out[key].empty, key
    assert out["graded"]["重賞"].all()


def test_format_table_percentages():
    """表示整形が、率だけを100倍し、倍率や境界値は素通しすること。"""
    table = pd.DataFrame({
        "的中率": [0.25], "達成率": [0.05], "達成率下限": [0.01],
        "回収率": [0.6], "倍率": [4.2], "指標下限": [0.35],
        "残す割合": [0.5], "達成率_重賞": [0.07], "達成率下限_平場": [0.02],
        "達成率差": [0.03], "レース数": [100],
    })
    out = trifecta.format_table(table)
    assert out.loc[0, "的中率"] == 25.0
    assert out.loc[0, "達成率下限"] == 1.0
    assert out.loc[0, "達成率_重賞"] == 7.0
    assert out.loc[0, "達成率下限_平場"] == 2.0
    assert out.loc[0, "達成率差"] == 3.0
    assert out.loc[0, "倍率"] == 4.2          # 倍数なのでそのまま
    assert out.loc[0, "指標下限"] == 0.35     # 層の境界値もそのまま
    assert out.loc[0, "残す割合"] == 0.5
    assert out.loc[0, "レース数"] == 100
