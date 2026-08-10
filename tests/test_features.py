"""合成データで「未来の情報が漏れていないか」を検証するテスト。

実データがなくても走る。`python -m pytest tests/ -q` で実行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import betting, evaluate, features, preprocess  # noqa: E402


def make_df(n_races: int = 60, n_horses: int = 8, seed: int = 0) -> pd.DataFrame:
    """小さな架空のレース結果を作る。"""
    rng = np.random.default_rng(seed)
    rows = []
    horses = [f"馬{i}" for i in range(20)]
    jockeys = [f"騎手{i}" for i in range(5)]
    trainers = [f"厩舎{i}" for i in range(4)]

    for r in range(n_races):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=r * 7)
        runners = rng.choice(horses, size=n_horses, replace=False)
        order = rng.permutation(n_horses) + 1
        for i, (h, rank) in enumerate(zip(runners, order)):
            rows.append({
                "レースID": f"R{r:04d}",
                "レース日付": date,
                "馬名": h,
                "馬番": i + 1,
                "枠番": (i // 2) + 1,
                "着順": int(rank),
                "タイム": 120.0 + rng.normal(),
                "騎手": rng.choice(jockeys),
                "調教師": rng.choice(trainers),
                "競馬場名": rng.choice(["東京", "中山"]),
                "芝・ダート区分": rng.choice(["芝", "ダート"]),
                "右左回り・直線区分": "左",
                "馬場状態1": rng.choice(["良", "重"]),
                "距離(m)": int(rng.choice([1200, 1600, 2000, 2400])),
                "馬齢": int(rng.integers(3, 8)),
                "斤量": 55.0,
                "馬体重": 480 + rng.integers(-30, 30),
                "場体重増減": rng.integers(-10, 10),
                # 1着ほどオッズが低くなるよう、ゆるく相関させる
                "単勝オッズ": float(np.round(rank * 2 + rng.random() * 3 + 1, 1)),
                "人気": int(rank),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    df = preprocess.basic_clean(make_df())
    return features.add_all_features(df)


def test_first_appearance_has_no_history(featured: pd.DataFrame):
    """各馬の初出走行では、過去実績が空（0回 / NaN）でなければならない。"""
    first = featured.groupby("馬名", sort=False).head(1)
    assert (first["通算出走回数"] == 0).all()
    assert first["通算勝率"].isna().all()
    assert first["前走着順"].isna().all()


def test_cumulative_stats_exclude_current_row(featured: pd.DataFrame):
    """通算勝率が「自分より前の行だけ」から計算されていることを直接確認する。"""
    df = featured.sort_values(["レース日付", "レースID", "馬番"]).reset_index(drop=True)
    for horse, g in df.groupby("馬名", sort=False):
        wins = (g["着順"] == 1).astype(float).to_numpy()
        for i in range(1, len(g)):
            expected = wins[:i].mean()  # i 行目より前だけの勝率
            actual = g["通算勝率"].to_numpy()[i]
            assert actual == pytest.approx(expected, abs=1e-6), f"{horse} の {i} 行目"


def test_recent_window_mean(featured: pd.DataFrame):
    """過去3走平均着順が、直前3走（自分を含まない）の平均と一致すること。"""
    df = featured.sort_values(["レース日付", "レースID", "馬番"]).reset_index(drop=True)
    for _, g in df.groupby("馬名", sort=False):
        ranks = g["着順"].to_numpy(dtype=float)
        got = g["過去3走平均着順"].to_numpy()
        for i in range(1, len(g)):
            expected = ranks[max(0, i - 3):i].mean()
            assert got[i] == pytest.approx(expected, abs=1e-4)


def test_shuffling_future_results_does_not_change_features():
    """未来の着順を書き換えても過去の特徴量が変わらない＝リークなしの決定的な確認。"""
    base = preprocess.basic_clean(make_df())
    f1 = features.add_all_features(base)

    # 後半のレースの着順だけをシャッフルする
    tampered = base.copy()
    half = tampered["レース日付"].quantile(0.5)
    mask = tampered["レース日付"] > half
    rng = np.random.default_rng(123)
    tampered.loc[mask, "着順"] = rng.permutation(tampered.loc[mask, "着順"].to_numpy())
    tampered["1着フラグ"] = (tampered["着順"] == 1).astype("int8")
    f2 = features.add_all_features(tampered)

    check_cols = ["通算勝率", "騎手通算勝率", "コンビ勝率", "過去5走平均着順"]
    front = ~mask.to_numpy()
    for col in check_cols:
        a, b = f1.loc[front, col], f2.loc[front, col]
        assert np.allclose(a.fillna(-1), b.fillna(-1)), f"{col} が未来の改変で変化した"


def test_jockey_change_flag(featured: pd.DataFrame):
    """乗り替わりフラグが前走の騎手との比較になっていること。"""
    df = featured.sort_values(["レース日付", "レースID", "馬番"])
    for _, g in df.groupby("馬名", sort=False):
        jockeys = g["騎手"].astype(str).to_numpy()
        flags = g["乗り替わりフラグ"].to_numpy()
        assert np.isnan(flags[0])
        for i in range(1, len(g)):
            assert flags[i] == float(jockeys[i] != jockeys[i - 1])


def test_strict_daily_lag_runs():
    """厳密モード（前日終了時点の騎手成績）でも例外なく動くこと。"""
    df = preprocess.basic_clean(make_df())
    out = features.add_all_features(df, strict_daily_lag=True)
    assert len(out) == len(df)
    assert out["騎手通算勝率"].notna().any()


def test_feature_columns_excludes_odds(featured: pd.DataFrame):
    """既定ではオッズ・人気が特徴量に入らないこと。"""
    cols = features.feature_columns(featured)
    assert "単勝オッズ" not in cols and "人気" not in cols
    assert "単勝オッズ" in features.feature_columns(featured, include_odds=True)


def test_backtest_math(featured: pd.DataFrame):
    """回収率の計算が定義どおりか（払戻＝オッズ×100、外れは0）。"""
    rng = np.random.default_rng(1)
    pred = rng.random(len(featured))
    r = evaluate.backtest_top_n(featured, pred, n=1)

    assert r.n_bets == featured["レースID"].nunique()
    assert r.invested == r.n_bets * 100
    hits = r.bets[r.bets["rank"] == 1]
    assert r.returned == pytest.approx((hits["odds"] * 100).sum())
    assert r.roi == pytest.approx(r.returned / r.invested)


def test_probability_normalized_per_race(featured: pd.DataFrame):
    """レース内で正規化した確率の合計が 1 になること。"""
    rng = np.random.default_rng(2)
    ev = betting.compute_expected_value(featured, rng.random(len(featured)))
    sums = ev.groupby("race_id")["prob"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_value_bet_filters(featured: pd.DataFrame):
    """期待値・オッズのフィルタが効いていること。"""
    rng = np.random.default_rng(3)
    pred = rng.random(len(featured))
    picks = betting.select_value_bets(featured, pred, ev_threshold=1.2, min_odds=5)
    assert (picks["ev"] >= 1.2).all()
    assert (picks["odds"] >= 5).all()

    capped = betting.select_value_bets(featured, pred, ev_threshold=0.0,
                                       max_bets_per_race=2)
    assert capped.groupby("race_id").size().max() <= 2


def test_kelly_is_bounded(featured: pd.DataFrame):
    """ケリー比率が 0〜上限に収まること。"""
    f = betting.kelly_fraction(np.array([0.5, 0.01, 0.9]), np.array([3.0, 2.0, 1.1]))
    assert (f >= 0).all() and (f <= 0.05).all()
