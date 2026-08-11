"""odds.csv から3連単の配当を取り出す。

■ 方針：配当金額だけを使い、当たり組み合わせは race_result から作る

  「どの組み合わせが当たりか」は race_result の着順（1着・2着・3着の馬番）から
  100%確実に分かる。だから odds.csv からは **払戻金額だけ** もらえばよい。
  こうするとカラム名解決の当たり判定を1列に減らせて壊れにくい。

■ カラム名が不明な場合
  Kaggle 版 odds.csv のカラム名は環境によってぶれるため、
  `describe_odds_columns(path)` で候補を目視してから
  `load_trifecta_payout(path, payout_col="...")` で明示指定できるようにしてある。

■ 単位
  JRA の払戻金は「100円あたり」で表記される。
  1点100円で買う本プロジェクトでは、払戻＝そのままの金額になる。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config, leakfree

# 3連単らしい列名（「3連単」「三連単」「trifecta」の表記ゆれを吸収）
_TRIFECTA = re.compile(r"(3連単|３連単|三連単|trifecta)", re.IGNORECASE)
# 3連複らしい列名
_TRIO = re.compile(r"(3連複|３連複|三連複|trio)", re.IGNORECASE)

# 券種名 -> 列名パターン
KIND_PATTERNS = {"3連単": _TRIFECTA, "3連複": _TRIO}
# 金額らしい列名
_PAYOUT = re.compile(r"(払戻|配当|払い戻し|payout|refund)")
# オッズらしい列名（払戻が無い場合の代替。オッズ×100 が払戻に相当する）
_ODDS = re.compile(r"(オッズ|odds)")


def describe_odds_columns(path: str | None = None, nrows: int = 5) -> pd.DataFrame:
    """odds.csv の列名を一覧し、3連単らしい列に印を付けて返す。

    実データを見る前にどの列を使うか決められないので、まずこれを実行して
    目で確認する。列名が分かったら load_trifecta_payout に渡す。
    """
    path = path or config.ODDS_CSV
    head = pd.read_csv(path, nrows=nrows, low_memory=False)
    return pd.DataFrame({
        "column": head.columns,
        "3連単らしい": [bool(_TRIFECTA.search(str(c))) for c in head.columns],
        "3連複らしい": [bool(_TRIO.search(str(c))) for c in head.columns],
        "金額らしい": [bool(_PAYOUT.search(str(c))) for c in head.columns],
        "サンプル": [head[c].iloc[0] if len(head) else None for c in head.columns],
    })


def _guess_payout_column(columns, kind: str = "3連単") -> str:
    """指定した券種の払戻列を推測する。"""
    pattern = KIND_PATTERNS[kind]
    candidates = [c for c in columns if pattern.search(str(c))]
    if kind == "3連単":
        # 「3連複」を「3連単」と誤認しないよう、複の方は除外しておく
        candidates = [c for c in candidates if not _TRIO.search(str(c))]
    if not candidates:
        raise KeyError(
            f"{kind}らしい列が見つかりません。"
            "describe_odds_columns() で列名を確認し、payout_col= で明示してください"
        )
    # 「払戻/配当」を最優先、次点でオッズ列
    for sub in (_PAYOUT, _ODDS):
        hit = [c for c in candidates if sub.search(str(c))]
        if hit:
            return hit[0]
    # 数字だけが入っていそうな列にフォールバック
    return candidates[0]


def _to_payout(raw: pd.DataFrame, column: str, is_odds: bool | None) -> pd.Series:
    """払戻列を数値に変換する（カンマ・「円」を除去、オッズなら100倍）。"""
    value = pd.to_numeric(
        raw[column].astype(str).str.replace(r"[,円]", "", regex=True),
        errors="coerce",
    )
    if is_odds is None:
        # 3連単の払戻は普通「数百〜数百万円」。中央値が3桁未満ならオッズ表記とみなす。
        is_odds = bool(_ODDS.search(str(column))) and (value.median() < 1000)
    if is_odds:
        value = value * 100  # オッズ（倍率）を100円あたりの払戻に直す
    return value.astype("float64")


def load_payouts(path: str | None = None, kinds: tuple[str, ...] = ("3連単", "3連複"),
                 columns: dict[str, str] | None = None,
                 is_odds: bool | None = None) -> pd.DataFrame:
    """レースIDごとの払戻（100円あたり）を券種ごとに返す。

    Parameters
    ----------
    kinds   : 取り出す券種。既定は3連単と3連複。
    columns : {券種: 列名} で明示指定できる。省略した券種は自動推測。
              自動推測に失敗した券種は、例外にせず単にスキップする
              （3連複の列がないデータでも3連単だけで動くようにするため）。

    戻り値の列: レースID, 3連単払戻, 3連複払戻（見つかったものだけ）
    """
    path = path or config.ODDS_CSV
    raw = pd.read_csv(path, low_memory=False)
    id_col = "レースID" if "レースID" in raw.columns else raw.columns[0]
    columns = columns or {}

    out = pd.DataFrame({"レースID": leakfree.normalize_id(raw[id_col])})
    for kind in kinds:
        col = columns.get(kind)
        if col is None:
            try:
                col = _guess_payout_column(raw.columns, kind)
            except KeyError:
                continue  # その券種の列がないだけなので無視する
        out[f"{kind}払戻"] = _to_payout(raw, col, is_odds)

    value_cols = [c for c in out.columns if c != "レースID"]
    if not value_cols:
        raise KeyError(
            "払戻列が1つも見つかりません。describe_odds_columns() で列名を確認してください"
        )
    return out.dropna(subset=value_cols, how="all").drop_duplicates(subset="レースID")


def load_trifecta_payout(path: str | None = None, payout_col: str | None = None,
                         is_odds: bool | None = None) -> pd.DataFrame:
    """レースIDごとの3連単払戻だけを返す（load_payouts の3連単版）。"""
    columns = {"3連単": payout_col} if payout_col else None
    return load_payouts(path, kinds=("3連単",), columns=columns, is_odds=is_odds)


def synthetic_payout_from_odds(df: pd.DataFrame, pred_free: bool = True) -> pd.DataFrame:
    """odds.csv が無いときの近似払戻（検証用のフォールバック）。

    単勝オッズから求めた各馬の市場勝率をもとに、Harville モデル
    （1着が決まったら残りで確率を再正規化する）で3連単の確率を出し、
    その逆数に控除率0.725（3連単の払戻率）を掛けて払戻を近似する。

    ⚠ あくまで近似。本番の検証は必ず実データの odds.csv で行うこと。
    """
    cols = config.resolve_columns(df)
    race_id, rank, odds = cols["race_id"], cols["rank"], cols["win_odds"]

    sub = df.loc[df[rank] <= 3, [race_id, rank, odds]].copy()
    sub["p"] = 1.0 / pd.to_numeric(sub[odds], errors="coerce")

    # レース全体の市場勝率合計（控除率込みなので1を超える）で正規化する
    total = df.groupby(race_id, observed=True)[odds].apply(
        lambda s: (1.0 / pd.to_numeric(s, errors="coerce")).sum()
    )
    sub = sub.merge(total.rename("total"), left_on=race_id, right_index=True, how="left")
    sub["p"] = sub["p"] / sub["total"]

    rows = []
    for rid, g in sub.groupby(race_id, observed=True, sort=False):
        g = g.sort_values(rank)
        p = g["p"].to_numpy()
        if len(p) < 3 or not np.isfinite(p).all():
            continue
        # Harville: P(1着A) * P(2着B|A除く) * P(3着C|A,B除く)
        prob = p[0] * (p[1] / max(1 - p[0], 1e-9)) * (p[2] / max(1 - p[0] - p[1], 1e-9))
        if prob <= 0:
            continue
        rows.append((rid, 0.725 / prob * 100))

    return pd.DataFrame(rows, columns=["レースID", "3連単払戻"]).assign(
        レースID=lambda d: leakfree.normalize_id(d["レースID"])
    )
