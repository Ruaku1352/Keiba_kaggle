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
        "金額らしい": [bool(_PAYOUT.search(str(c))) for c in head.columns],
        "サンプル": [head[c].iloc[0] if len(head) else None for c in head.columns],
    })


def _guess_payout_column(columns) -> str:
    """3連単の払戻列を推測する。"""
    trifecta = [c for c in columns if _TRIFECTA.search(str(c))]
    if not trifecta:
        raise KeyError(
            "3連単らしい列が見つかりません。"
            "describe_odds_columns() で列名を確認し、payout_col= で明示してください"
        )
    # 「3連単」かつ「払戻/配当」を最優先、次点でオッズ列
    for pattern in (_PAYOUT, _ODDS):
        hit = [c for c in trifecta if pattern.search(str(c))]
        if hit:
            return hit[0]
    # 数字だけが入っていそうな列にフォールバック
    return trifecta[0]


def load_trifecta_payout(path: str | None = None, payout_col: str | None = None,
                         is_odds: bool | None = None) -> pd.DataFrame:
    """レースIDごとの3連単払戻（100円あたり）を返す。

    Parameters
    ----------
    payout_col : 払戻列の名前。省略時は自動推測。
    is_odds    : その列が「払戻金」ではなく「オッズ（倍率）」の場合 True。
                 省略時は列名と値の大きさから判定する。

    戻り値の列: レースID, 3連単払戻
    """
    path = path or config.ODDS_CSV
    raw = pd.read_csv(path, low_memory=False)

    id_col = "レースID" if "レースID" in raw.columns else raw.columns[0]
    payout_col = payout_col or _guess_payout_column(raw.columns)

    value = pd.to_numeric(
        raw[payout_col].astype(str).str.replace(r"[,円]", "", regex=True),
        errors="coerce",
    )

    if is_odds is None:
        # 3連単の払戻は普通「数百〜数百万円」。中央値が3桁未満ならオッズ表記とみなす。
        median = value.median()
        is_odds = bool(_ODDS.search(str(payout_col))) and (median < 1000)
    if is_odds:
        value = value * 100  # オッズ（倍率）を100円あたりの払戻に直す

    out = pd.DataFrame({
        "レースID": leakfree.normalize_id(raw[id_col]),
        "3連単払戻": value.astype("float64"),
    })
    return out.dropna(subset=["3連単払戻"]).drop_duplicates(subset="レースID")


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
