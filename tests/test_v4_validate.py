"""v4（時系列CV・仮説判定）のテスト。

v4は「新しい数字を出す」のではなく「再現するかを判定する」コードなので、
テストの主眼も **判定ロジックが正しいか** に置く。
仮説が真のデータ／偽のデータを人工的に作り、
判定が期待どおりの結論を返すことを確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocess, selection, trifecta, validate  # noqa: E402
from test_features import make_df  # noqa: E402


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------
def make_fold_race(n: int = 400, hot_rate_graded: float = 0.2,
                   hot_rate_flat: float = 0.02, seed: int = 0,
                   grade_rates: dict | None = None) -> pd.DataFrame:
    """達成率を狙って作れる、1行1レースのテーブルを組み立てる。

    「上位6頭BOX（120点=12,000円 → 熱い基準50,000円）で当たる」ように
    予測順位と払戻を仕込む。
    """
    rng = np.random.default_rng(seed)
    grades = rng.choice(["G1", "G2", "G3", None], size=n, p=[0.1, 0.1, 0.2, 0.6])
    graded = np.array([g is not None for g in grades])

    # 熱い当たりにしたい行を選ぶ。
    # 割合をサンプリングではなく **決め打ち** にして、テストを決定的にする
    # （乱数だと「差がないはずなのに偶然有意」が5%の確率で起きてテストが不安定になる）
    hot = np.zeros(n, dtype=bool)
    for g in set(grades):
        idx = np.flatnonzero(np.array([x == g for x in grades]))
        if grade_rates is not None and g in grade_rates:
            rate = grade_rates[g]
        else:
            rate = hot_rate_graded if g is not None else hot_rate_flat
        hot[idx[:int(round(len(idx) * rate))]] = True

    # 熱い行は上位6頭に収まり、かつ払戻が基準超え。それ以外は外れ。
    ranks = np.where(hot[:, None], rng.integers(1, 7, (n, 3)), rng.integers(7, 15, (n, 3)))
    payout = np.where(hot, 80_000.0, 3_000.0)

    return pd.DataFrame({
        "レースID": [f"R{i:04d}" for i in range(n)],
        "日付": pd.date_range("2011-01-01", periods=n, freq="D"),
        "出走頭数": 16,
        "1着予測順位": ranks[:, 0],
        "2着予測順位": ranks[:, 1],
        "3着予測順位": ranks[:, 2],
        "格付け": grades,
        "重賞": graded,
        "3連単払戻": payout,
        "3連複払戻": payout / 8,
        "エントロピー": rng.random(n),
        "正規化エントロピー": rng.random(n),
        "確率標準偏差": rng.random(n),
        "ジニ係数": rng.random(n),
        "1位2位差": rng.random(n),
        "上位3頭確率和": rng.random(n),
        "最大確率": rng.random(n),
    })


def make_results(specs: list[dict]) -> list[dict]:
    """fold 結果のリストを組み立てる。"""
    out = []
    for i, spec in enumerate(specs, start=1):
        fold = validate.Fold(str(i), (2003, 2010 + i), (2011 + i, 2013 + i))
        out.append({"fold": fold, "race": make_fold_race(seed=i, **spec), "auc": 0.78})
    return out


# ---------------------------------------------------------------------------
# fold の定義
# ---------------------------------------------------------------------------
def test_default_folds_are_walk_forward():
    """学習期間が伸び、検証期間が前に進み、両者が重ならないこと。"""
    folds = validate.DEFAULT_FOLDS
    assert len(folds) == 4
    for f in folds:
        # 学習の終わり < 検証の始まり（未来を学習していない）
        assert f.train_years[1] < f.test_years[0], f.name
    # 検証期間が前に進んでいる
    starts = [f.test_years[0] for f in folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    # 学習の終端が伸びている（walk-forward）
    ends = [f.train_years[1] for f in folds]
    assert ends == sorted(ends)


def test_slice_years_is_inclusive():
    """年の範囲指定が両端を含むこと。"""
    df = preprocess.basic_clean(make_df(n_races=200))
    years = df["レース日付"].dt.year
    lo, hi = int(years.min()), int(years.min()) + 1
    sub = validate._slice_years(df, (lo, hi))
    assert set(sub["レース日付"].dt.year.unique()) <= {lo, hi}
    assert (sub["レース日付"].dt.year == hi).any()


def test_folds_do_not_share_quantiles():
    """選別条件の分位点が fold ごとに計算されること（fold間で情報を共有しない）。"""
    cond = validate.FIXED_CONDITIONS[0]
    a = make_fold_race(n=200, seed=1)
    b = a.copy()
    b["エントロピー"] = b["エントロピー"] + 100  # 値域を丸ごとずらす

    # 分位点が各テーブル内で計算されるなら、残る行数は同じになるはず
    assert len(cond.apply(a)) == len(cond.apply(b))


# ---------------------------------------------------------------------------
# 依頼A：fold ごとの集計
# ---------------------------------------------------------------------------
def test_fold_summary_shape():
    """fold ごとに1行、必要な列が揃うこと。"""
    results = make_results([{}, {}, {}])
    table = validate.fold_summary(results)
    assert len(table) == 3
    for col in ["AUC", "重賞達成率", "重賞下限", "平場達成率", "G1達成率", "F148達成率"]:
        assert col in table.columns


def test_fold_grade_table_splits_grades():
    """fold × グレードの表になること。"""
    results = make_results([{}, {}])
    table = validate.fold_grade_table(results)
    assert set(table["格付け"]) <= {"G1", "G2", "G3", "G"}
    assert table["fold"].nunique() == 2


# ---------------------------------------------------------------------------
# 依頼B：H1（重賞 > 平場）
# ---------------------------------------------------------------------------
def test_h1_detects_real_difference():
    """重賞が明確に良いデータなら、全foldで再現し有意になること。"""
    results = make_results([{"hot_rate_graded": 0.25, "hot_rate_flat": 0.02}] * 3)
    h1 = validate.test_h1(results)
    assert h1["再現fold数"] == "3/3"
    assert h1["有意"] is True
    assert h1["差"] > 0
    assert h1["差下限"] > 0  # 差の信頼区間が0をまたがない


def test_h1_reports_no_difference_when_none():
    """差がないデータなら有意にならないこと（偽陽性を出さない）。"""
    results = make_results([{"hot_rate_graded": 0.05, "hot_rate_flat": 0.05}] * 3)
    h1 = validate.test_h1(results)
    assert h1["有意"] is False
    assert h1["差下限"] < 0 < h1["差上限"]


def test_two_proportion_test_matches_manual():
    """2群の比率検定が定義どおりに計算されること。"""
    res = trifecta.two_proportion_test(30, 100, 10, 100)
    p_pool = 40 / 200
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / 100 + 1 / 100))
    assert res["z"] == pytest.approx((0.3 - 0.1) / se)
    assert res["差"] == pytest.approx(0.2)
    assert res["有意"] is True
    # 同じ比率なら z=0、p=1
    same = trifecta.two_proportion_test(10, 100, 10, 100)
    assert same["z"] == pytest.approx(0.0)
    assert same["p値"] == pytest.approx(1.0)
    assert same["有意"] is False


def test_two_proportion_test_handles_empty():
    """母数0でも例外にせず NaN を返すこと。"""
    res = trifecta.two_proportion_test(0, 0, 5, 50)
    assert np.isnan(res["z"]) and res["有意"] is False


# ---------------------------------------------------------------------------
# 依頼B：H2（G1 > G2/G3）
# ---------------------------------------------------------------------------
def test_h2_counts_folds_where_g1_best():
    """G1が突出したデータなら、全foldでG1が最高になること。"""
    spec = {"grade_rates": {"G1": 0.5, "G2": 0.02, "G3": 0.02, None: 0.02}}
    results = make_results([spec] * 3)
    h2 = validate.test_h2(results)
    assert h2["G1が最高だったfold"] == "3/3"
    assert h2["有意"] is True


def test_h2_not_significant_when_equal():
    """G1とG2/G3が同じなら有意にならないこと。"""
    spec = {"grade_rates": {"G1": 0.1, "G2": 0.1, "G3": 0.1, None: 0.02}}
    results = make_results([spec] * 3)
    h2 = validate.test_h2(results)
    assert h2["有意"] is False


def test_h2_reports_small_sample_size():
    """G1のレース数が表に出ること（区間の広さを判断するため）。"""
    results = make_results([{}] * 3)
    h2 = validate.test_h2(results)
    assert "G1R数" in h2["per_fold"].columns
    assert (h2["per_fold"]["G1R数"] > 0).all()


# ---------------------------------------------------------------------------
# 依頼B：H3（上位6頭BOXが最良）
# ---------------------------------------------------------------------------
def test_h3_compares_only_budget_strategies():
    """比較対象が F2→6→10 / F3→6→8 / 上位6頭BOX の3つに限られること。"""
    assert [s.points for s in validate.H3_STRATEGIES] == [80, 90, 120]
    results = make_results([{}] * 2)
    h3 = validate.test_h3(results)
    assert set(h3["per_fold"]["買い方"]) == {s.name for s in validate.H3_STRATEGIES}


def test_h3_picks_best_by_lower_bound():
    """達成率と下限、両方の観点で最良を数えていること。"""
    results = make_results([{}] * 3)
    h3 = validate.test_h3(results)
    assert "/3" in h3["達成率で最良だったfold"]
    assert "/3" in h3["下限で最良だったfold"]
    # 各foldの最良が3つの買い方のいずれか
    names = {s.name for s in validate.H3_STRATEGIES}
    assert set(h3["各foldの最良（下限）"].values()) <= names


# ---------------------------------------------------------------------------
# 依頼B：H4（選別は効かない）
# ---------------------------------------------------------------------------
def test_h4_uses_exactly_three_fixed_conditions():
    """条件が固定3つで、探索していないこと。"""
    assert len(validate.FIXED_CONDITIONS) == 3
    for cond in validate.FIXED_CONDITIONS:
        assert len(cond.rules) == 2  # 複合条件は2指標まで


def test_h4_concludes_no_effect_on_random_metrics():
    """指標がランダム（＝無関係）なら「効いていない」と結論すること。"""
    results = make_results([{}] * 4)
    h4 = validate.test_h4(results)
    # 全foldで下限が改善する条件はまず出ない
    assert h4["全foldで下限が改善した条件"] == []
    assert "H4を支持" in h4["結論"]
    # 基準行が各foldに1つずつ入っている
    base = h4["per_fold"].loc[h4["per_fold"]["条件"] == "全重賞（基準）"]
    assert len(base) == 4


def test_h4_detects_a_genuinely_working_condition():
    """本当に効く指標を仕込めば、ちゃんと「効いた」と検出できること。

    （検出力がないから「効かない」と言っているわけではない、という確認）
    """
    results = make_results([{}] * 3)
    for r in results:
        race = r["race"]
        # エントロピーが高い行ほど熱くなるよう仕込む
        hot = race["3連単払戻"] >= 50_000
        race.loc[hot, "エントロピー"] = 0.9 + np.random.default_rng(0).random(hot.sum()) * 0.1
        race.loc[~hot, "エントロピー"] = np.random.default_rng(1).random((~hot).sum()) * 0.5

    h4 = validate.test_h4(results)
    entropy_conditions = [c for c in h4["全foldで下限が改善した条件"] if "エントロピー" in c]
    assert entropy_conditions, "効く条件を検出できていない"
    assert "H4を支持" not in h4["結論"]


def test_condition_apply_narrows_races():
    """複合条件を適用するとレース数が減ること。"""
    race = make_fold_race(n=400, seed=3)
    cond = validate.FIXED_CONDITIONS[0]  # 高50% × 低50%
    out = cond.apply(race)
    assert 0 < len(out) < len(race)
    # だいたい 50% × 50% = 25% 前後に落ちる
    assert 0.15 < len(out) / len(race) < 0.35


def test_condition_missing_metric_returns_empty():
    """指標の列がない場合は空を返すこと（誤って全件を買わない）。"""
    race = make_fold_race(n=100).drop(columns=["エントロピー"])
    assert validate.FIXED_CONDITIONS[0].apply(race).empty


# ---------------------------------------------------------------------------
# 依頼C：結論
# ---------------------------------------------------------------------------
def test_recommendation_follows_evidence():
    """重賞が明確に良いデータなら、結論が「重賞のみ」になること。"""
    results = make_results([{"hot_rate_graded": 0.25, "hot_rate_flat": 0.02}] * 3)
    h1, h2 = validate.test_h1(results), validate.test_h2(results)
    h3, h4 = validate.test_h3(results), validate.test_h4(results)
    rec = validate.recommend(results, h1, h2, h3, h4)

    assert rec["買う対象"].startswith("重賞のみ")
    assert "有意" in rec["対象の根拠"]
    assert rec["選別条件を使うか"] == "使わない"
    assert 0 <= rec["期待達成率（全fold プール）"] <= 1
    lo, hi = rec["信頼区間"]
    assert lo <= rec["期待達成率（全fold プール）"] <= hi
    assert len(rec["限界"]) >= 3


def test_recommendation_marks_unproven_g1():
    """G1が有意でなければ、G1に絞る根拠がないと明記すること。"""
    spec = {"grade_rates": {"G1": 0.1, "G2": 0.1, "G3": 0.1, None: 0.02}}
    results = make_results([spec] * 3)
    h1, h2 = validate.test_h1(results), validate.test_h2(results)
    h3, h4 = validate.test_h3(results), validate.test_h4(results)
    rec = validate.recommend(results, h1, h2, h3, h4)
    assert "根拠は得られなかった" in rec["G1について"]
    assert "とくにG1" not in rec["買う対象"]


# ---------------------------------------------------------------------------
# 全体
# ---------------------------------------------------------------------------
def test_run_v4_end_to_end():
    """学習器を差し替えて、全体が通ることを確認する（LightGBM不要）。"""
    df = make_df(n_races=400)
    # 年をばらけさせて fold を作れるようにする
    df["レース日付"] = pd.to_datetime("2011-01-01") + pd.to_timedelta(
        np.arange(len(df)) // 8 * 3, unit="D")
    graded_ids = set(df["レースID"].unique()[::4])
    df["リステッド・重賞競走"] = [
        "G1" if r in graded_ids else None for r in df["レースID"]]
    df = preprocess.basic_clean(df)

    ids = df["レースID"].astype(str).unique()
    rng = np.random.default_rng(0)
    payout = pd.DataFrame({
        "レースID": ids,
        "3連単払戻": np.round(np.exp(rng.normal(9.5, 1.5, len(ids)))),
    })

    folds = [
        validate.Fold("1", (2011, 2011), (2012, 2012)),
        validate.Fold("2", (2011, 2012), (2013, 2013)),
        validate.Fold("3", (2011, 2013), (2014, 2014)),
    ]

    def fake_model(train_df, test_df, feature_cols):
        """LightGBMの代わりに乱数を返す（判定パイプラインの疎通確認）。"""
        return np.random.default_rng(len(train_df)).random(len(test_df)), 0.75

    out = validate.run_v4(df, payout, folds=folds, model_fn=fake_model, verbose=False)

    assert len(out["results"]) == 3
    assert len(out["fold_summary"]) == 3
    for key in ["H1", "H2", "H3", "H4"]:
        assert "仮説" in out[key]
    assert "買い方" in out["recommendation"]
    # 難易度指標が各foldのレーステーブルに付いている
    assert "エントロピー" in out["results"][0]["race"].columns


def test_run_time_series_cv_reuses_features():
    """特徴量が既にある DataFrame を渡したら作り直さないこと。"""
    df = preprocess.basic_clean(make_df(n_races=200))
    df["レース日付"] = pd.to_datetime("2011-01-01") + pd.to_timedelta(
        np.arange(len(df)) // 8 * 5, unit="D")
    featured = df.copy()
    featured["通算勝率"] = 0.1  # 「特徴量あり」の目印

    calls = []

    def fake_model(train_df, test_df, feature_cols):
        calls.append(len(feature_cols))
        return np.zeros(len(test_df)), 0.5

    payout = pd.DataFrame({"レースID": featured["レースID"].astype(str).unique(),
                           "3連単払戻": 10000.0})
    folds = [validate.Fold("1", (2011, 2011), (2012, 2012))]
    validate.run_time_series_cv(featured, payout, folds=folds,
                                feature_cols=["馬齢"], model_fn=fake_model,
                                verbose=False)
    assert calls == [1]  # 渡した feature_cols がそのまま使われた


def test_run_v4_requires_payout():
    """配当データがなければ、黙って進まず明示的に失敗すること。"""
    df = preprocess.basic_clean(make_df(n_races=40))
    with pytest.raises(ValueError, match="配当"):
        validate.run_v4(df, payout_df=None, verbose=False)
