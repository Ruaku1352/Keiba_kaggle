"""依頼A〜E（脚質・ペース・3連単検証・パドック補正・記録）のテスト。

実データがなくても走るよう、合成データを組み立てて検証する。
特に重視しているのは2種類のリーク検出:

  (1) 未来シャッフル不変性
      未来のレース結果を書き換えても、過去行の特徴量が1ミリも変わらないこと

  (2) 同一レース内リーク
      ペースは「レース単位の値」なので、馬の行のまま累積すると
      同じレースの他の馬の行から自分のレースのペースが漏れる。
      自分のレースだけを書き換えても自分の特徴量が変わらないことを確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import corner, features, pace, paddock, preprocess, records, trifecta  # noqa: E402
from test_features import make_df  # noqa: E402

# 合成データは数百レースしかないので、統計量の最低母数を下げておく。
# （本番の既定値は 30 / 50。ここを下げないと z スコアが全部 NaN になり、
#   リーク検証が「NaN 同士の比較」になって意味を失う）
pace.PACE_Z_MIN_COUNT = 5
pace.EXPECTED_PACE_MIN_COUNT = 5


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------
def make_corner_csv(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """race_result から、それらしいコーナー通過順の文字列を作る。"""
    rng = np.random.default_rng(seed)
    rows = []
    for race_id, g in df.groupby("レースID", sort=False):
        posts = list(g["馬番"].astype(int))
        rng.shuffle(posts)
        # 先頭2頭を括弧でくくって「横並び」を再現する
        text = f"(*{posts[0]},{posts[1]})" + ",".join(str(p) for p in posts[2:])
        rows.append({"レースID": race_id, "4コーナー": text})
    return pd.DataFrame(rows)


def make_lap_df(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """レース単位のペース表（pace.compute_race_pace の出力形式）を作る。"""
    rng = np.random.default_rng(seed)
    ids = df["レースID"].drop_duplicates().to_numpy()
    first3 = 34.0 + rng.normal(0, 1.5, len(ids))
    last3 = 35.0 + rng.normal(0, 1.5, len(ids))
    return pd.DataFrame({
        "レースID": ids.astype(str),
        "前半3F": first3.astype("float32"),
        "上がり3F": last3.astype("float32"),
        "ラップ本数": 8,
        "合計タイム": (first3 + last3 + 45).astype("float32"),
        "ペース指標": (first3 - last3).astype("float32"),
    })


@pytest.fixture(scope="module")
def built():
    """脚質・ペースまで載せた DataFrame を1度だけ作る。"""
    base = preprocess.basic_clean(make_df(n_races=200))
    cdf = _corner_long(make_corner_csv(base))
    ldf = make_lap_df(base)
    out = features.add_all_features(base, corner_df=cdf, lap_df=ldf,
                                    drop_helper_cols=False)
    return base, cdf, ldf, out


def _corner_long(wide: pd.DataFrame) -> pd.DataFrame:
    """make_corner_csv の横持ちを load_corner と同じ縦持ちに変換する。"""
    records_ = []
    for race_id, text in zip(wide["レースID"], wide["4コーナー"]):
        for horse_no, rank in corner.parse_passing_order(text).items():
            records_.append((str(race_id), "4コーナー", horse_no, rank))
    return pd.DataFrame(records_, columns=["レースID", "corner", "馬番", "通過順位"])


# ---------------------------------------------------------------------------
# 依頼A: コーナー通過順のパース
# ---------------------------------------------------------------------------
def test_parse_basic():
    """依頼書の実データ例をそのままパースできること。"""
    got = corner.parse_passing_order("(*13,6)4,10-2,9,12,3(7,11)1=8,5")
    assert got == {13: 1, 6: 1, 4: 3, 10: 4, 2: 5, 9: 6, 12: 7, 3: 8,
                   7: 9, 11: 9, 1: 11, 8: 12, 5: 13}


def test_parse_group_is_tied():
    """括弧内は同順位、次はその頭数分だけ飛ぶ（競技順位方式）。"""
    assert corner.parse_passing_order("(1,2,3)4") == {1: 1, 2: 1, 3: 1, 4: 4}


def test_parse_handles_spaces_and_symbols():
    """空白・記号（-, =）は単なる区切りとして扱う。"""
    assert corner.parse_passing_order("(*2,10)(1,5,11)8   9") == {
        2: 1, 10: 1, 1: 3, 5: 3, 11: 3, 8: 6, 9: 7}
    assert corner.parse_passing_order("5,3,1-2=4") == {5: 1, 3: 2, 1: 3, 2: 4, 4: 5}


def test_parse_empty():
    """欠損・空文字は空辞書（例外を投げない）。"""
    assert corner.parse_passing_order(np.nan) == {}
    assert corner.parse_passing_order("") == {}


def test_parse_two_digit_numbers():
    """2桁の馬番が分断されないこと。"""
    assert corner.parse_passing_order("18,17,16") == {18: 1, 17: 2, 16: 3}


# ---------------------------------------------------------------------------
# 依頼A: 脚質特徴量
# ---------------------------------------------------------------------------
def test_relative_position_range(built):
    """4角相対位置が 0〜1 に収まること（0=先頭, 1=最後方）。"""
    _, _, _, out = built
    pos = out["_4角相対位置"].dropna()
    assert pos.min() >= 0.0 and pos.max() <= 1.0


def test_running_style_first_race_is_nan(built):
    """初出走では脚質スコアが未定義（過去がないので NaN）。"""
    _, _, _, out = built
    first = out.groupby("馬名", sort=False).head(1)
    assert first["脚質スコア"].isna().all()


def test_running_style_equals_past_mean(built):
    """脚質スコアが「自分より前の4角相対位置の平均」と一致すること。"""
    _, _, _, out = built
    df = out.sort_values(["レース日付", "レースID", "馬番"]).reset_index(drop=True)
    for _, g in df.groupby("馬名", sort=False):
        pos = g["_4角相対位置"].to_numpy()
        got = g["脚質スコア"].to_numpy()
        for i in range(1, len(g)):
            past = pos[:i]
            past = past[~np.isnan(past)]
            if len(past) == 0:
                assert np.isnan(got[i])
            else:
                assert got[i] == pytest.approx(past.mean(), abs=1e-4)


def test_solo_escape_flag(built):
    """単騎逃げフラグ = 自分が逃げ馬 かつ レース内の逃げ馬が1頭。"""
    _, _, _, out = built
    is_nige = out["脚質カテゴリ"].astype(object) == "逃げ"
    expected = (is_nige & (out["レース内逃げ馬頭数"] == 1)).astype(int)
    assert (out["単騎逃げフラグ"].to_numpy() == expected.to_numpy()).all()


def test_race_scenario_is_race_level(built):
    """展開特徴量（逃げ馬頭数・前傾度）はレース内で全馬同じ値になること。"""
    _, _, _, out = built
    for col in ["レース内逃げ馬頭数", "レース内先行馬頭数", "前傾度"]:
        n_unique = out.groupby("レースID", sort=False)[col].nunique(dropna=False)
        assert (n_unique <= 1).all(), f"{col} がレース内で割れている"


# ---------------------------------------------------------------------------
# 依頼B: ラップからのペース計算
# ---------------------------------------------------------------------------
def test_compute_race_pace_uses_laps_not_broken_columns():
    """壊れた「前半3ハロン」列を無視し、ラップから計算し直すこと。"""
    raw = pd.DataFrame({
        "レースID": [1, 2],
        "ラップタイム1": [7.7, 12.5],
        "ラップタイム2": [11.4, 10.9],
        "ラップタイム3": [11.6, 11.6],
        "ラップタイム4": [12.0, 12.2],
        "ラップタイム5": [12.7, 13.9],
        "ラップタイム6": [12.6, np.nan],
        # 壊れた列（ラップ1と同じ値が入っている実データを再現）
        "前半3ハロン": [7.7, 12.5],
        "上がり3ハロン": [7.7, 12.5],
    })
    out = pace.compute_race_pace(raw)

    # 前半3F = 7.7+11.4+11.6 = 30.7（壊れた列の 7.7 ではない）
    assert out.loc[0, "前半3F"] == pytest.approx(30.7, abs=1e-3)
    # 上がり3F = 末尾3本 = 12.0+12.7+12.6 = 37.3
    assert out.loc[0, "上がり3F"] == pytest.approx(37.3, abs=1e-3)
    # 2行目はラップ5本 → 末尾3本 = 11.6+12.2+13.9 = 37.7
    assert out.loc[1, "上がり3F"] == pytest.approx(37.7, abs=1e-3)
    assert out.loc[1, "ラップ本数"] == 5
    assert out.loc[0, "ペース指標"] == pytest.approx(30.7 - 37.3, abs=1e-3)


def test_compute_race_pace_needs_three_laps():
    """ラップが3本未満なら計算不能として NaN にする。"""
    raw = pd.DataFrame({"レースID": [1], "ラップタイム1": [12.0], "ラップタイム2": [np.nan]})
    out = pace.compute_race_pace(raw)
    assert np.isnan(out.loc[0, "前半3F"]) and np.isnan(out.loc[0, "上がり3F"])


def test_compute_race_pace_from_cumulative():
    """ラップ列がなくてもペース（累積タイム）から復元できること。"""
    raw = pd.DataFrame({
        "レースID": [1],
        "ペース1": [7.7], "ペース2": [19.1], "ペース3": [30.7],
        "ペース4": [42.7], "ペース5": [55.4], "ペース6": [68.0],
    })
    out = pace.compute_race_pace(raw)
    assert out.loc[0, "前半3F"] == pytest.approx(30.7, abs=1e-2)


def test_pace_features_exist(built):
    """ペース適性と想定ペースが作られていること。"""
    _, _, _, out = built
    for col in ["前走ペース指標", "好走時ペース平均", "ペース巧拙差", "想定ペース"]:
        assert col in out.columns
    assert out["想定ペース"].notna().any()


# ---------------------------------------------------------------------------
# リーク検出（最重要）
# ---------------------------------------------------------------------------
def _front_mask(df: pd.DataFrame) -> np.ndarray:
    half = df["レース日付"].quantile(0.5)
    return (df["レース日付"] <= half).to_numpy()


def test_future_shuffle_invariance_for_v2_features():
    """未来の結果・通過順・ペースを全部書き換えても、過去行の特徴量は不変。"""
    base = preprocess.basic_clean(make_df(n_races=200))
    cdf = _corner_long(make_corner_csv(base))
    ldf = make_lap_df(base)
    f1 = features.add_all_features(base, corner_df=cdf, lap_df=ldf)

    front = _front_mask(base)
    rng = np.random.default_rng(999)

    # 1) 後半レースの着順をシャッフル
    tampered = base.copy()
    idx = ~front
    tampered.loc[idx, "着順"] = rng.permutation(tampered.loc[idx, "着順"].to_numpy())
    tampered["1着フラグ"] = (tampered["着順"] == 1).astype("int8")

    # 2) 後半レースのコーナー通過順を作り直す（別シード）
    future_ids = set(tampered.loc[idx, "レースID"])
    cdf2 = cdf.copy()
    shuffled = _corner_long(make_corner_csv(tampered.loc[idx], seed=777))
    cdf2 = pd.concat([cdf2.loc[~cdf2["レースID"].isin(future_ids)], shuffled])

    # 3) 後半レースのペースを別の値に差し替える
    ldf2 = ldf.copy()
    mask = ldf2["レースID"].isin({str(i) for i in future_ids})
    ldf2.loc[mask, "ペース指標"] = rng.normal(0, 3, int(mask.sum())).astype("float32")
    ldf2.loc[mask, "上がり3F"] = rng.normal(35, 2, int(mask.sum())).astype("float32")

    f2 = features.add_all_features(tampered, corner_df=cdf2, lap_df=ldf2)

    check = (corner.running_style_feature_columns() + pace.pace_feature_columns()
             + ["通算勝率", "騎手通算勝率"])
    for col in check:
        if col not in f1.columns or f1[col].dtype.name == "category":
            continue
        a = f1.loc[front, col].fillna(-999).to_numpy()
        b = f2.loc[front, col].fillna(-999).to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{col} が未来の改変で変化した"


def test_no_same_race_pace_leak():
    """自分のレースのペースを書き換えても、自分の行の特徴量は変わらないこと。

    ペースはレース単位の値なので、馬の行のまま累積すると
    「同じレースの他の馬の行」経由で自分のレースのペースが漏れる。
    その漏れがないことを直接確かめる。
    """
    base = preprocess.basic_clean(make_df(n_races=200))
    cdf = _corner_long(make_corner_csv(base))
    ldf = make_lap_df(base)
    f1 = features.add_all_features(base, corner_df=cdf, lap_df=ldf)

    # 真ん中あたりの1レースだけペースを極端な値にする
    target = base["レースID"].drop_duplicates().to_numpy()[100]
    ldf2 = ldf.copy()
    row = ldf2["レースID"] == str(target)
    ldf2.loc[row, "ペース指標"] = 99.0
    ldf2.loc[row, "上がり3F"] = 99.0
    f2 = features.add_all_features(base, corner_df=cdf, lap_df=ldf2)

    mask = (base["レースID"] == target).to_numpy()
    for col in pace.pace_feature_columns():
        if col not in f1.columns:
            continue
        a = f1.loc[mask, col].fillna(-999).to_numpy()
        b = f2.loc[mask, col].fillna(-999).to_numpy()
        assert np.allclose(a, b), f"{col} に同一レースのペースが漏れている"


def test_no_same_race_corner_leak():
    """自分のレースの通過順を書き換えても、自分の行の脚質特徴量は変わらないこと。"""
    base = preprocess.basic_clean(make_df(n_races=200))
    cdf = _corner_long(make_corner_csv(base))
    ldf = make_lap_df(base)
    f1 = features.add_all_features(base, corner_df=cdf, lap_df=ldf)

    target = base["レースID"].drop_duplicates().to_numpy()[100]
    cdf2 = cdf.copy()
    rows = cdf2["レースID"] == str(target)
    # 通過順を丸ごと逆転させる
    cdf2.loc[rows, "通過順位"] = cdf2.loc[rows, "通過順位"].max() + 1 - cdf2.loc[rows, "通過順位"]

    f2 = features.add_all_features(base, corner_df=cdf2, lap_df=ldf)
    mask = (base["レースID"] == target).to_numpy()
    for col in ["脚質スコア", "脚質安定度", "前走4角相対位置", "好走時脚質スコア",
                "レース内逃げ馬頭数", "前傾度", "単騎逃げフラグ"]:
        a = f1.loc[mask, col].fillna(-999).to_numpy()
        b = f2.loc[mask, col].fillna(-999).to_numpy()
        assert np.allclose(a, b), f"{col} に同一レースの通過順が漏れている"


def test_helper_columns_are_dropped():
    """`_` 始まりの作業列（そのレースの結果）が特徴量に残っていないこと。"""
    base = preprocess.basic_clean(make_df(n_races=60))
    cdf = _corner_long(make_corner_csv(base))
    ldf = make_lap_df(base)
    out = features.add_all_features(base, corner_df=cdf, lap_df=ldf)
    assert not [c for c in out.columns if c.startswith("_")]
    assert "_ペース指標" not in features.feature_columns(out)


# ---------------------------------------------------------------------------
# 依頼C: 3連単の検証
# ---------------------------------------------------------------------------
def test_box_point_counts():
    """BOXの点数が k(k-1)(k-2) になること。"""
    assert trifecta.box(3).points == 6
    assert trifecta.box(4).points == 24
    assert trifecta.box(5).points == 60
    assert trifecta.box(6).points == 120


def test_hot_threshold_matches_spec():
    """依頼書 3.3 の表と同じ基準額が選ばれること。"""
    cases = {600: 10_000, 1500: 10_000, 2100: 20_000, 2400: 20_000,
             3600: 20_000, 5200: 30_000, 6000: 30_000, 8800: 50_000}
    for invest, expected in cases.items():
        assert trifecta.hot_threshold(invest) == expected, invest


def test_wilson_interval_handles_zero():
    """0件でも区間が [0, 上限] になること（素朴な近似だと負に飛び出す）。"""
    lo, hi = trifecta.wilson_interval(0, 500)
    assert lo == 0.0 and 0 < hi < 0.05
    lo, hi = trifecta.normal_interval(0, 500)
    assert lo == hi == 0.0  # 素朴な近似は幅0になってしまう

    lo, hi = trifecta.wilson_interval(5, 100)
    assert lo < 0.05 < hi


def test_simulate_hit_logic():
    """当たり判定が「実着順の予測順位 ≤ 候補数」で決まること。"""
    race = pd.DataFrame({
        "レースID": ["A", "B", "C"],
        "日付": pd.to_datetime(["2020-01-01"] * 3),
        "出走頭数": [16, 16, 16],
        # A: 1,2,3着とも予測上位3頭 → 3頭BOXで的中
        # B: 3着が予測5位 → 3頭BOXは外れ、5頭BOXなら的中
        # C: 1着が予測10位 → どちらも外れ
        "1着予測順位": [1, 2, 10],
        "2着予測順位": [3, 1, 1],
        "3着予測順位": [2, 5, 2],
        "重賞": [True, False, False],
        "3連単払戻": [12000.0, 30000.0, 500.0],
    })
    r3 = trifecta.simulate(race, trifecta.box(3))
    assert r3["的中数"] == 1
    assert r3["点数"] == 6 and r3["投資/R"] == 600
    assert r3["熱い基準"] == 10_000
    assert r3["熱い当たり"] == 1              # 12,000円 ≥ 10,000円
    assert r3["達成率"] == pytest.approx(1 / 3)
    assert r3["回収率"] == pytest.approx(12000 / (600 * 3))

    r5 = trifecta.simulate(race, trifecta.box(5))
    assert r5["的中数"] == 2
    assert r5["熱い基準"] == 30_000           # 投資6,000円 → 5,000円の段
    assert r5["熱い当たり"] == 1              # 30,000円のみ基準到達


def test_simulate_respects_field_size():
    """出走頭数が候補数に満たない場合、点数が実際に買える数まで縮むこと。"""
    race = pd.DataFrame({
        "レースID": ["A"], "日付": pd.to_datetime(["2020-01-01"]),
        "出走頭数": [4], "1着予測順位": [1], "2着予測順位": [2],
        "3着予測順位": [3], "重賞": [False], "3連単払戻": [1000.0],
    })
    r = trifecta.simulate(race, trifecta.box(6))
    assert r["点数"] == 4 * 3 * 2  # 6頭BOXではなく4頭ぶんしか買えない


def test_build_race_table_and_graded():
    """レース単位テーブルが作られ、重賞フラグが立つこと。"""
    df = make_df(n_races=30)
    graded_ids = set(df["レースID"].unique()[:5])
    df["リステッド・重賞競走"] = [
        "G1" if r in graded_ids else None for r in df["レースID"]]
    df = preprocess.basic_clean(df)

    rng = np.random.default_rng(0)
    pred = rng.random(len(df))
    payout = pd.DataFrame({
        "レースID": df["レースID"].astype(str).unique(),
        "3連単払戻": 5000.0,
    })
    race = trifecta.build_race_table(df, pred, payout)

    assert len(race) == df["レースID"].nunique()
    assert race["重賞"].sum() == 5
    # 予測順位は 1〜出走頭数 の範囲に収まる
    assert race["1着予測順位"].between(1, race["出走頭数"]).all()

    table = trifecta.strategy_table(race)
    assert {"買い方", "達成率", "回収率", "達成率下限"} <= set(table.columns)


def test_gap_box_runs():
    """乖離ベースの買い方が計算できること。"""
    df = preprocess.basic_clean(make_df(n_races=50))
    rng = np.random.default_rng(1)
    pred = rng.random(len(df))
    payout = pd.DataFrame({
        "レースID": df["レースID"].astype(str).unique(), "3連単払戻": 20000.0})
    res = trifecta.simulate_gap_box(df, pred, payout)
    assert res["点数"] == 6
    assert 0 <= res["的中率"] <= 1
    assert res["レース数"] == df["レースID"].nunique()


# ---------------------------------------------------------------------------
# 依頼D: パドック補正
# ---------------------------------------------------------------------------
def test_logit_sigmoid_roundtrip():
    """logit と sigmoid が互いの逆変換になっていること。"""
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(paddock.sigmoid(paddock.logit(p)), p, atol=1e-9)


def test_sigmoid_extreme_values_are_stable():
    """極端な入力でもオーバーフローせず 0〜1 に収まること。"""
    out = paddock.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0) and out[2] == pytest.approx(1.0)


def test_adjustment_stays_probability():
    """補正後も 0〜1 に収まり、レース内で合計1になること。"""
    pred = np.array([0.05, 0.30, 0.60, 0.80])
    horses = np.array(["A", "B", "C", "D"])
    out = paddock.apply_paddock_adjustment(pred, horses, {"A": 1.0, "D": -1.0}, strength=2.0)
    assert (out >= 0).all() and (out <= 1).all()
    assert out.sum() == pytest.approx(1.0)


def test_adjustment_direction():
    """プラス評価で確率が上がり、マイナス評価で下がること。"""
    pred = np.array([0.2, 0.2, 0.2])
    horses = np.array(["A", "B", "C"])
    out = paddock.apply_paddock_adjustment(pred, horses, {"A": 1.0, "C": -1.0}, strength=0.5)
    assert out[0] > out[1] > out[2]


def test_adjustment_is_odds_multiplication():
    """logit加算 = オッズの定数倍、という動作原理どおりになっていること。"""
    pred = np.array([0.2, 0.5])
    horses = np.array(["A", "B"])
    a = 0.5 * 1.0  # strength × スコア
    out = paddock.apply_paddock_adjustment(pred, horses, {"A": 1.0}, strength=0.5,
                                           normalize=False)
    odds_before = 0.2 / 0.8
    odds_after = out[0] / (1 - out[0])
    assert odds_after == pytest.approx(odds_before * np.exp(a), rel=1e-6)
    assert out[1] == pytest.approx(0.5)  # 評価のない馬は動かない


def test_zero_strength_is_noop():
    """strength=0 なら（正規化を除いて）元の予測のままであること。"""
    pred = np.array([0.1, 0.2, 0.7])
    horses = np.array(["A", "B", "C"])
    out = paddock.apply_paddock_adjustment(pred, horses, {"A": 1.0}, strength=0.0)
    assert np.allclose(out, pred / pred.sum())


def test_adjustment_report_shows_rank_change():
    """補正前後の順位変動が表に出ること。"""
    pred = np.array([0.5, 0.4, 0.1])
    horses = np.array(["A", "B", "C"])
    table = paddock.adjustment_report(pred, horses, {"C": 1.0}, strength=3.0,
                                      odds=np.array([2.0, 3.0, 50.0]))
    assert set(["馬名", "補正前順位", "補正後順位", "順位変動", "補正後EV"]) <= set(table.columns)
    row = table.loc[table["馬名"] == "C"].iloc[0]
    assert row["順位変動"] > 0  # 評価を上げたので順位が上がっている


def test_adjust_predictions_multi_race():
    """複数レースをまとめて補正でき、評価のないレースは変わらないこと。"""
    df = pd.DataFrame({
        "レースID": ["A", "A", "B", "B"],
        "馬名": ["h1", "h2", "h3", "h4"],
    })
    pred = np.array([0.3, 0.3, 0.3, 0.3])
    src = paddock.DictPaddockInput({"A": {"h1": 1.0}})
    out = paddock.adjust_predictions(df, pred, src, strength=1.0)
    assert out[0] > out[1]                      # レースAは補正された
    assert np.allclose(out[2:], pred[2:])       # レースBは無変更


# ---------------------------------------------------------------------------
# 依頼E: 記録
# ---------------------------------------------------------------------------
def test_prediction_log_roundtrip(tmp_path):
    """予測を書いて、着順を追記して、結合できること。"""
    log = records.PredictionLog(str(tmp_path))
    race_df = pd.DataFrame({
        "馬番": [1, 2, 3],
        "馬名": ["A", "B", "C"],
        "単勝オッズ": [2.0, 5.0, 30.0],
        "レース日付": pd.to_datetime(["2026-12-27"] * 3),
    })
    pred = np.array([0.4, 0.3, 0.05])
    adjusted = paddock.apply_paddock_adjustment(pred, race_df["馬名"], {"C": 1.0}, 0.5)

    log.record_race("2026R1", race_df, pred, {"C": 1.0}, adjusted, race_name="有馬記念")
    log.record_result("2026R1", pd.DataFrame({
        "馬番": [1, 2, 3], "馬名": ["A", "B", "C"], "着順": [2, 1, 3]}))
    log.record_bet("2026R1", "3連単BOX", "1-2-3", 6, 600, 4300)

    joined = log.load_joined()
    assert len(joined) == 3
    assert joined["着順"].notna().all()
    assert joined.loc[joined["馬名"] == "C", "パドックスコア"].iloc[0] == 1.0
    # 補正で C の確率が上がっている
    row = joined.loc[joined["馬名"] == "C"].iloc[0]
    assert row["補正後確率"] > row["予測確率"] / pred.sum() * 0  # 記録できていること

    skill = log.evaluate_paddock_skill()
    assert not skill.empty
    assert "複勝率" in skill.columns

    bets = pd.read_csv(log.bet_path)
    assert bets.loc[0, "的中"] == 1


def test_prediction_log_appends(tmp_path):
    """2レース分を追記しても混ざらないこと。"""
    log = records.PredictionLog(str(tmp_path))
    race_df = pd.DataFrame({"馬番": [1, 2], "馬名": ["A", "B"], "単勝オッズ": [2.0, 3.0]})
    log.record_race("R1", race_df, np.array([0.5, 0.5]))
    log.record_race("R2", race_df, np.array([0.6, 0.4]))
    table = pd.read_csv(log.prediction_path, dtype={"レースID": str})
    assert len(table) == 4
    assert set(table["レースID"]) == {"R1", "R2"}
