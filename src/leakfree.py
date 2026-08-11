"""リークを起こさない「過去だけの集計」を作る道具箱。

このプロジェクトの全ての過去実績系特徴量は、ここにある関数だけで作る。
道具を1箇所に集めておけば「どこかで未来を見ていないか」を目視で追える。

■ 中核となる考え方

    groupby.cumcount()            → 自分より前の行数
    groupby.cumsum() - 自分の値    → 自分より前の合計

  cumsum() は「自分を含む累積和」なので、自分の値を引けば
  「自分より前だけの累積和」になる。これだけで漏れは防げる。

■ 前提
  DataFrame が [レース日付, レースID, 馬番] 順にソート済みであること
  （preprocess.basic_clean が保証する）。この順序があるので、
  cumsum は必ず「過去 → 未来」の向きに走る。

■ 欠損の扱い
  past_mean_ignore_nan / past_std_ignore_nan は
  「値がある行だけ」を母数にする。コーナー通過順のように
  一部の年しかデータがない特徴量では、これを使わないと
  欠損を0とみなして平均が歪む。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _grouper(df: pd.DataFrame, keys: list[str]) -> list[pd.Series]:
    """groupby に渡すキー列のリスト（Series で渡すと index が揃う）。"""
    return [df[k] for k in keys]


def past_count(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """自分より前に、そのキーで何行あったか。"""
    return df.groupby(keys, observed=True, sort=False).cumcount()


def past_sum(df: pd.DataFrame, keys: list[str], value: pd.Series) -> pd.Series:
    """自分より前の合計（自分自身は含まない）。NaN は 0 として扱う。"""
    tmp = value.fillna(0)
    cum = tmp.groupby(_grouper(df, keys), observed=True, sort=False).cumsum()
    return cum - tmp


def past_rate(df: pd.DataFrame, keys: list[str], value: pd.Series,
              min_count: int = 1) -> pd.Series:
    """自分より前の平均（勝率など）。母数が min_count 未満なら NaN。

    value が 0/1 のフラグであることを前提にした「率」用。
    欠損がある連続値には past_mean_ignore_nan を使うこと。
    """
    cnt = past_count(df, keys)
    tot = past_sum(df, keys, value)
    rate = tot / cnt.replace(0, np.nan)
    return rate.where(cnt >= min_count)


def past_window_mean(df: pd.DataFrame, keys: list[str], value: pd.Series,
                     window: int) -> pd.Series:
    """直近 window 行の平均（自分自身は含まない）。

    累積和の差分で計算するので O(n)。
        直近N走の合計 = (自分より前の累積和) - (N行前時点の累積和)
    groupby.rolling は162万行だと現実的な時間で終わらないため使わない。
    """
    tmp = value.fillna(0)
    g = _grouper(df, keys)
    past_cum = tmp.groupby(g, observed=True, sort=False).cumsum() - tmp
    lagged = past_cum.groupby(g, observed=True, sort=False).shift(window).fillna(0)

    cnt = past_count(df, keys)
    denom = np.minimum(cnt, window)  # 出走数が window に満たない馬に対応
    return ((past_cum - lagged) / denom.replace(0, np.nan)).astype("float32")


def past_mean_ignore_nan(df: pd.DataFrame, keys: list[str], value: pd.Series,
                         min_count: int = 1) -> pd.Series:
    """自分より前の平均。**値が入っている行だけ**を母数にする。

    仕組み: 合計は NaN を 0 とみなして足し、母数は「値があった回数」を数える。
    こうすると欠損レース（コーナー情報のない年など）が平均を薄めない。
    """
    valid = value.notna().astype("float32")
    total = past_sum(df, keys, value)
    n = past_sum(df, keys, valid)
    mean = total / n.replace(0, np.nan)
    return mean.where(n >= min_count).astype("float32")


def past_std_ignore_nan(df: pd.DataFrame, keys: list[str], value: pd.Series,
                        min_count: int = 2) -> pd.Series:
    """自分より前の標準偏差（母集団標準偏差）。値がある行だけが母数。

    Var(X) = E[X^2] - E[X]^2 を使うので、2回の累積和だけで求まる。
    浮動小数の丸めで極小の負値が出ることがあるため 0 で下限を切っている。
    """
    valid = value.notna().astype("float32")
    n = past_sum(df, keys, valid)
    s1 = past_sum(df, keys, value)
    s2 = past_sum(df, keys, value.astype("float64") ** 2)

    denom = n.replace(0, np.nan)
    var = s2 / denom - (s1 / denom) ** 2
    var = var.clip(lower=0)
    return np.sqrt(var).where(n >= min_count).astype("float32")


def past_conditional_mean(df: pd.DataFrame, keys: list[str], value: pd.Series,
                          condition: pd.Series, min_count: int = 1) -> pd.Series:
    """条件を満たした過去レースだけでの平均。

    例）「複勝圏に入ったレースの平均ペース指標」＝ハイペース巧者かどうか。
    合計も母数も condition を掛けてから累積するだけ。
    """
    cond = condition.fillna(0).astype("float32")
    valid = (value.notna().astype("float32") * cond)
    total = past_sum(df, keys, value.fillna(0) * cond)
    n = past_sum(df, keys, valid)
    mean = total / n.replace(0, np.nan)
    return mean.where(n >= min_count).astype("float32")


def label_encode(s: pd.Series) -> pd.Series:
    """groupby キー用に文字列/カテゴリを整数コード化する（速度・メモリ対策）。"""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes
    return s.astype("category").cat.codes


def normalize_id(s: pd.Series) -> pd.Series:
    """レースIDを文字列に正規化する。

    CSV ごとに int で読まれたり str で読まれたりするため、
    結合前に必ずこれを通して型を揃える。float で読まれた場合の
    "198601010101.0" も int を経由して潰す。
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int64").astype(str)
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
