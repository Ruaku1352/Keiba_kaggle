"""特徴量エンジニアリング（依頼1）。

■ 設計の大原則：Data Leakage を絶対に起こさない
   すべての「過去実績系」特徴量は、**その行自身を含まない**過去だけから作る。
   実装は次の 2 つの道具で統一している。

     過去の件数  : groupby.cumcount()                → 自分より前の行数
     過去の合計  : groupby.cumsum() - 自分の値        → 自分より前の合計

   cumsum() は「自分を含む累積和」なので、自分の値を引けば
   「自分より前だけの累積和」になる。これが漏れ防止の中核。

   直近N走は「累積和の差分」で取る:
       直近N走の合計 = (自分より前の累積和) - (N行前の時点での累積和)
   groupby.rolling は 100 万行超だと非常に遅いので使わない。

■ 前提
   preprocess.basic_clean() で
   [レース日付, レースID, 馬番] の順にソート済みであること。
   この順序が保証されていれば cumsum は必ず「過去 → 未来」の向きに走る。

■ 同日レースの扱い
   騎手・調教師の集計は「同じ開催日の、より前のレース」を過去として含む。
   実運用（当日朝に予測）で厳密にしたい場合は
   add_all_features(..., strict_daily_lag=True) を使うと、
   騎手・調教師系を「前日終了時点」の成績に切り替える。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 低レベルのヘルパー
# ---------------------------------------------------------------------------
def _past_count(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """自分より前に、そのキーで何行あったか（＝過去出走回数）。"""
    return df.groupby(keys, observed=True, sort=False).cumcount()


def _past_sum(df: pd.DataFrame, keys: list[str], value: pd.Series) -> pd.Series:
    """自分より前の合計（自分自身は含まない）。"""
    tmp = value.fillna(0)
    cum = tmp.groupby([df[k] for k in keys], observed=True, sort=False).cumsum()
    return cum - tmp


def _past_rate(df: pd.DataFrame, keys: list[str], value: pd.Series,
               min_count: int = 1) -> pd.Series:
    """自分より前の平均（＝勝率など）。母数が min_count 未満なら NaN。"""
    cnt = _past_count(df, keys)
    tot = _past_sum(df, keys, value)
    rate = tot / cnt.replace(0, np.nan)
    return rate.where(cnt >= min_count)


def _past_window_mean(df: pd.DataFrame, keys: list[str], value: pd.Series,
                      window: int) -> pd.Series:
    """直近 window 走の平均（自分自身は含まない）。

    累積和の差分で計算するので O(n)。groupby.rolling より桁違いに速い。
    """
    tmp = value.fillna(0)
    grouper = [df[k] for k in keys]
    past_cum = tmp.groupby(grouper, observed=True, sort=False).cumsum() - tmp
    # window 行前の時点での「過去累積和」
    lagged = past_cum.groupby(grouper, observed=True, sort=False).shift(window).fillna(0)

    cnt = _past_count(df, keys)
    denom = np.minimum(cnt, window)  # 出走数が window に満たない馬に対応
    win_sum = past_cum - lagged
    return (win_sum / denom.replace(0, np.nan)).astype("float32")


def _label_encode(s: pd.Series) -> pd.Series:
    """groupby キー用に文字列/カテゴリを整数コード化する（速度・メモリ対策）。"""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes
    return s.astype("category").cat.codes


# ---------------------------------------------------------------------------
# 馬の実績系
# ---------------------------------------------------------------------------
def add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの過去実績特徴量を追加する。"""
    cols = config.resolve_columns(df)
    horse, rank, date = cols["horse"], cols["rank"], cols["date"]

    win = (df[rank] == 1).astype("float32")      # 1着
    quinella = (df[rank] <= 2).astype("float32")  # 連対（2着以内）
    show = (df[rank] <= 3).astype("float32")      # 複勝圏（3着以内）

    keys = [horse]

    # --- 通算成績（その時点まで）------------------------------------------
    df["通算出走回数"] = _past_count(df, keys).astype("int32")
    df["通算勝率"] = _past_rate(df, keys, win).astype("float32")
    df["通算連対率"] = _past_rate(df, keys, quinella).astype("float32")
    df["通算複勝率"] = _past_rate(df, keys, show).astype("float32")

    # --- 直近N走の平均着順 --------------------------------------------------
    rank_f = df[rank].astype("float32")
    for n in config.RECENT_WINDOWS:
        df[f"過去{n}走平均着順"] = _past_window_mean(df, keys, rank_f, n)

    # --- 前走系 -------------------------------------------------------------
    g_horse = df.groupby(horse, observed=True, sort=False)
    df["前走着順"] = g_horse[rank].shift(1).astype("float32")
    df["前走からの日数"] = (df[date] - g_horse[date].shift(1)).dt.days.astype("float32")

    if "distance" in cols:
        dist = cols["distance"]
        prev_dist = g_horse[dist].shift(1)
        df["前走との距離差"] = (df[dist] - prev_dist).astype("float32")

    if "last3f" in cols:
        df["前走上がり3F"] = g_horse[cols["last3f"]].shift(1).astype("float32")

    # --- 条件別の過去勝率 ---------------------------------------------------
    # 距離帯：±200m のスライド窓は groupby にできないので、200m 刻みのビンで代用。
    # （1600m の馬は 1500-1699m の馬とだけ比較される。厳密な ±200m ではない）
    if "distance" in cols:
        df["距離帯"] = (df[cols["distance"]] // 200).astype("float32")
        df["同距離帯_過去勝率"] = _past_rate(df, [horse, "距離帯"], win).astype("float32")
        df["同距離帯_過去出走数"] = _past_count(df, [horse, "距離帯"]).astype("int32")

    if "surface" in cols:
        code = _label_encode(df[cols["surface"]])
        df["_surface_code"] = code
        df["同芝ダ_過去勝率"] = _past_rate(df, [horse, "_surface_code"], win).astype("float32")
        df["同芝ダ_過去出走数"] = _past_count(df, [horse, "_surface_code"]).astype("int32")

    if "going" in cols:
        code = _label_encode(df[cols["going"]])
        df["_going_code"] = code
        df["同馬場状態_過去勝率"] = _past_rate(df, [horse, "_going_code"], win).astype("float32")
        df["同馬場状態_過去出走数"] = _past_count(df, [horse, "_going_code"]).astype("int32")

    if "course" in cols:
        code = _label_encode(df[cols["course"]])
        df["_course_code"] = code
        df["同競馬場_過去勝率"] = _past_rate(df, [horse, "_course_code"], win).astype("float32")

    return df


# ---------------------------------------------------------------------------
# 騎手・調教師系
# ---------------------------------------------------------------------------
def _daily_lagged_rate(df: pd.DataFrame, key_cols: list[str], value: pd.Series,
                       date_col: str) -> pd.Series:
    """「前日終了時点」の勝率を返す（同日レースの結果を一切使わない厳密版）。

    日単位で集計 → その日の分を除いた累積 → 元の行に結合、という手順。
    """
    tmp = pd.DataFrame({"_v": value.fillna(0).to_numpy(), "_d": df[date_col].to_numpy()})
    for c in key_cols:
        tmp[c] = df[c].to_numpy()

    daily = tmp.groupby(key_cols + ["_d"], observed=True, sort=True).agg(
        _sum=("_v", "sum"), _cnt=("_v", "size")
    ).reset_index()

    g = daily.groupby(key_cols, observed=True, sort=False)
    # その日を含まない累積 = 累積 - 当日分
    daily["_past_sum"] = g["_sum"].cumsum() - daily["_sum"]
    daily["_past_cnt"] = g["_cnt"].cumsum() - daily["_cnt"]
    daily["_rate"] = daily["_past_sum"] / daily["_past_cnt"].replace(0, np.nan)

    merged = tmp.merge(
        daily[key_cols + ["_d", "_rate", "_past_cnt"]], on=key_cols + ["_d"], how="left"
    )
    return pd.Series(merged["_rate"].to_numpy(), index=df.index, dtype="float32")


def add_jockey_features(df: pd.DataFrame, strict_daily_lag: bool = False) -> pd.DataFrame:
    """騎手系の特徴量を追加する。"""
    cols = config.resolve_columns(df)
    if "jockey" not in cols:
        return df
    jockey, rank, date = cols["jockey"], cols["rank"], cols["date"]

    win = (df[rank] == 1).astype("float32")
    df["_jockey_code"] = _label_encode(df[jockey])

    if strict_daily_lag:
        df["騎手通算勝率"] = _daily_lagged_rate(df, ["_jockey_code"], win, date)
    else:
        df["騎手通算勝率"] = _past_rate(df, ["_jockey_code"], win).astype("float32")

    df["騎手通算騎乗数"] = _past_count(df, ["_jockey_code"]).astype("int32")
    df[f"騎手直近{config.JOCKEY_RECENT_WINDOW}走勝率"] = _past_window_mean(
        df, ["_jockey_code"], win, config.JOCKEY_RECENT_WINDOW
    )

    if "_course_code" in df.columns:
        df["騎手×競馬場_勝率"] = _past_rate(
            df, ["_jockey_code", "_course_code"], win, min_count=10
        ).astype("float32")

    if "_surface_code" in df.columns:
        df["騎手×芝ダ_勝率"] = _past_rate(
            df, ["_jockey_code", "_surface_code"], win, min_count=10
        ).astype("float32")

    return df


def add_trainer_features(df: pd.DataFrame, strict_daily_lag: bool = False) -> pd.DataFrame:
    """調教師系の特徴量を追加する。"""
    cols = config.resolve_columns(df)
    if "trainer" not in cols:
        return df
    trainer, rank, date = cols["trainer"], cols["rank"], cols["date"]

    win = (df[rank] == 1).astype("float32")
    df["_trainer_code"] = _label_encode(df[trainer])

    if strict_daily_lag:
        df["調教師通算勝率"] = _daily_lagged_rate(df, ["_trainer_code"], win, date)
    else:
        df["調教師通算勝率"] = _past_rate(df, ["_trainer_code"], win).astype("float32")

    df["調教師通算出走数"] = _past_count(df, ["_trainer_code"]).astype("int32")
    return df


# ---------------------------------------------------------------------------
# 馬 × 騎手
# ---------------------------------------------------------------------------
def add_horse_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """コンビ回数・コンビ勝率・乗り替わりフラグ。"""
    cols = config.resolve_columns(df)
    if "jockey" not in cols:
        return df
    horse, jockey, rank = cols["horse"], cols["jockey"], cols["rank"]

    win = (df[rank] == 1).astype("float32")
    if "_jockey_code" not in df.columns:
        df["_jockey_code"] = _label_encode(df[jockey])

    keys = [horse, "_jockey_code"]
    df["コンビ回数"] = _past_count(df, keys).astype("int32")
    df["コンビ勝率"] = _past_rate(df, keys, win).astype("float32")

    # 乗り替わり：前走と騎手が違えば 1。初出走（前走なし）は NaN。
    prev_jockey = df.groupby(horse, observed=True, sort=False)["_jockey_code"].shift(1)
    df["乗り替わりフラグ"] = np.where(
        prev_jockey.isna(), np.nan, (df["_jockey_code"] != prev_jockey).astype("float32")
    ).astype("float32")

    return df


# ---------------------------------------------------------------------------
# レース単位の特徴量（当日情報のみを使うので漏れではない）
# ---------------------------------------------------------------------------
def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """出走頭数など、レース内で完結する特徴量。"""
    cols = config.resolve_columns(df)
    race_id = cols["race_id"]
    g_race = df.groupby(race_id, observed=True, sort=False)

    df["出走頭数"] = g_race[cols["post"] if "post" in cols else race_id].transform("size").astype("int16")

    # レース内での相対値（当日出走馬同士の比較。結果は使っていないので安全）
    if "通算勝率" in df.columns:
        df["通算勝率_レース内順位"] = g_race["通算勝率"].rank(
            ascending=False, method="min"
        ).astype("float32")
    if "騎手通算勝率" in df.columns:
        df["騎手勝率_レース内順位"] = g_race["騎手通算勝率"].rank(
            ascending=False, method="min"
        ).astype("float32")
    if "weight_carried" in cols:
        df["斤量_レース内平均差"] = (
            df[cols["weight_carried"]] - g_race[cols["weight_carried"]].transform("mean")
        ).astype("float32")

    return df


# ---------------------------------------------------------------------------
# まとめて実行
# ---------------------------------------------------------------------------
def add_all_features(df: pd.DataFrame, strict_daily_lag: bool = False,
                     drop_helper_cols: bool = True) -> pd.DataFrame:
    """すべての特徴量を追加する（この関数だけ呼べばよい）。

    Parameters
    ----------
    strict_daily_lag :
        True なら騎手・調教師の勝率を「前日終了時点」で計算する（より厳密）。
    drop_helper_cols :
        True なら内部用の `_xxx_code` 列を最後に削除する。
    """
    df = df.copy()

    df = add_horse_features(df)
    df = add_jockey_features(df, strict_daily_lag=strict_daily_lag)
    df = add_trainer_features(df, strict_daily_lag=strict_daily_lag)
    df = add_horse_jockey_features(df)
    df = add_race_features(df)

    if drop_helper_cols:
        helper = [c for c in df.columns if c.startswith("_") and c.endswith("_code")]
        df = df.drop(columns=helper)

    return df


def feature_columns(df: pd.DataFrame, include_odds: bool = False) -> list[str]:
    """学習に使う特徴量の列名リストを返す。

    include_odds=False（既定）なら単勝オッズ・人気を除外する。
    市場（オッズ）と独立した予測を作るのが目的なので、通常は除外して学習し、
    オッズは「期待値の計算時」にだけ使う。
    """
    cols = config.resolve_columns(df)

    base_logical = ["bracket", "post", "age", "weight_carried", "horse_weight",
                    "horse_weight_diff", "distance", "course", "surface", "turn", "going"]
    if include_odds:
        base_logical += ["win_odds", "popularity"]

    engineered = [
        "通算出走回数", "通算勝率", "通算連対率", "通算複勝率",
        "前走着順", "前走からの日数", "前走との距離差", "前走上がり3F",
        "同距離帯_過去勝率", "同距離帯_過去出走数",
        "同芝ダ_過去勝率", "同芝ダ_過去出走数",
        "同馬場状態_過去勝率", "同馬場状態_過去出走数",
        "同競馬場_過去勝率",
        "騎手通算勝率", "騎手通算騎乗数", f"騎手直近{config.JOCKEY_RECENT_WINDOW}走勝率",
        "騎手×競馬場_勝率", "騎手×芝ダ_勝率",
        "調教師通算勝率", "調教師通算出走数",
        "コンビ回数", "コンビ勝率", "乗り替わりフラグ",
        "出走頭数", "通算勝率_レース内順位", "騎手勝率_レース内順位", "斤量_レース内平均差",
    ]
    engineered += [f"過去{n}走平均着順" for n in config.RECENT_WINDOWS]

    result = [cols[k] for k in base_logical if k in cols]
    result += [c for c in engineered if c in df.columns]
    # 重複を除きつつ順序を維持
    seen: set[str] = set()
    return [c for c in result if not (c in seen or seen.add(c))]
