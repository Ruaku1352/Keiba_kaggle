"""依頼D：パドック評価による予測の事後補正。

パドックの良し悪しは過去データに存在しないので、学習には組み込めない。
そこで「学習済みモデルの出力を、当日の目視評価で後から動かす」形にする。

■ なぜ logit 空間で足すのか（重要）

  予測確率を直接 1.5 倍すると、元が 0.8 の馬は 1.2 になって確率でなくなる。
  そこで確率を **logit（対数オッズ）** に変換してから加算する。

      logit(p) = log(p / (1-p))        …… 0〜1 を -∞〜+∞ に開く
      sigmoid(x) = 1 / (1 + exp(-x))   …… その逆変換

  logit 空間での加算は「オッズを定数倍する」ことに等しい。

      logit(p) + a  ⇔  odds → odds × e^a

  つまり a=+0.5 なら「オッズが約1.65倍になる」補正。
  確率が 0 や 1 を超えることは絶対にないし、
  元々 0.02 の馬と 0.6 の馬に同じ a を足しても、
  「掛け算として同じ強さ」の補正になる。これが確率を直接いじるより自然な理由。

■ 補正の強さ
  strength は logit に足す量のスケール。
      補正量 = strength × スコア
  スコアを -1.0〜+1.0 で入力する想定なので、strength=0.5 なら
  最大でオッズ1.65倍相当。初期値は控えめにしてある。

■ 入力の差し替え
  当面は辞書入力。将来スマホから入れられるように、
  入力部分は PaddockInput クラスに分離してある
  （Googleスプレッドシート連携はここのサブクラスを足すだけで済む）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 変換の基本
# ---------------------------------------------------------------------------
def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """確率 → 対数オッズ。0/1 ちょうどで発散しないよう eps でクリップする。"""
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """対数オッズ → 確率。オーバーフローしない書き方にしてある。"""
    x = np.asarray(x, dtype="float64")
    out = np.empty_like(x)
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


# ---------------------------------------------------------------------------
# 入力（将来スプレッドシート連携に差し替える部分）
# ---------------------------------------------------------------------------
class PaddockInput:
    """パドック評価の入力元。`get(race_id) -> {馬名: スコア}` を実装する。"""

    def get(self, race_id: str) -> dict[str, float]:
        raise NotImplementedError


@dataclass
class DictPaddockInput(PaddockInput):
    """辞書で直接渡す入力（当面はこれを使う）。

    >>> src = DictPaddockInput({"202612250811": {"ドウデュース": 0.8}})
    """

    data: dict[str, dict[str, float]] = field(default_factory=dict)

    def get(self, race_id: str) -> dict[str, float]:
        return self.data.get(str(race_id), {})

    def set(self, race_id: str, scores: dict[str, float]) -> None:
        self.data[str(race_id)] = scores


@dataclass
class CsvPaddockInput(PaddockInput):
    """CSV から読む入力。列: レースID, 馬名, パドックスコア

    スマホからスプレッドシートに入力 → CSV書き出し、という運用を想定。
    Googleスプレッドシート直結にする場合も、この get() を差し替えるだけで済む。
    """

    path: str
    race_col: str = "レースID"
    horse_col: str = "馬名"
    score_col: str = "パドックスコア"

    def get(self, race_id: str) -> dict[str, float]:
        table = pd.read_csv(self.path, dtype={self.race_col: str})
        sub = table.loc[table[self.race_col].astype(str) == str(race_id)]
        return dict(zip(sub[self.horse_col], pd.to_numeric(sub[self.score_col], errors="coerce")))


# ---------------------------------------------------------------------------
# 補正の本体
# ---------------------------------------------------------------------------
def apply_paddock_adjustment(pred, horses, adjustments: dict[str, float],
                             strength: float = 0.5,
                             normalize: bool = True) -> np.ndarray:
    """予測確率にパドック評価を反映する。

    Parameters
    ----------
    pred        : モデルの予測確率（1レース分）
    horses      : 馬名の配列（pred と同じ並び）
    adjustments : {馬名: スコア}。スコアは -1.0〜+1.0 程度
    strength    : 補正の強さ。logit に `strength × スコア` を足す
    normalize   : True ならレース内で合計1に正規化する

    Returns
    -------
    補正後の確率（pred と同じ長さ）
    """
    pred = np.asarray(pred, dtype="float64")
    horses = np.asarray(horses)

    # 馬名 → スコアの対応を配列に展開（評価のない馬は 0 = 補正なし）
    scores = np.array([float(adjustments.get(str(h), 0.0)) for h in horses])

    adjusted = sigmoid(logit(pred) + strength * scores)

    if normalize:
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
    return adjusted


def adjustment_report(pred, horses, adjustments: dict[str, float],
                      strength: float = 0.5, odds=None) -> pd.DataFrame:
    """補正の前後で予測順位がどう動いたかを表にする。

    「自分のパドック評価がモデルをどれだけ動かしたか」を毎回見るための機能。
    順位が動きすぎるなら strength が強すぎる。
    """
    before = np.asarray(pred, dtype="float64")
    before_norm = before / before.sum() if before.sum() > 0 else before
    after = apply_paddock_adjustment(pred, horses, adjustments, strength=strength)

    table = pd.DataFrame({
        "馬名": np.asarray(horses),
        "スコア": [float(adjustments.get(str(h), 0.0)) for h in horses],
        "補正前確率": before_norm,
        "補正後確率": after,
    })
    table["補正前順位"] = table["補正前確率"].rank(ascending=False, method="first").astype(int)
    table["補正後順位"] = table["補正後確率"].rank(ascending=False, method="first").astype(int)
    table["順位変動"] = table["補正前順位"] - table["補正後順位"]  # プラスなら評価が上がった

    if odds is not None:
        odds = np.asarray(odds, dtype="float64")
        table["オッズ"] = odds
        table["補正前EV"] = before_norm * odds
        table["補正後EV"] = after * odds

    return table.sort_values("補正後確率", ascending=False).reset_index(drop=True)


def adjust_predictions(df: pd.DataFrame, pred, source: PaddockInput,
                       strength: float = 0.5,
                       race_col: str = "レースID",
                       horse_col: str = "馬名") -> np.ndarray:
    """複数レースをまとめて補正する（レース単位でループする実運用向け）。"""
    pred = np.asarray(pred, dtype="float64")
    out = pred.copy()

    race_ids = df[race_col].astype(str).to_numpy()
    horses = df[horse_col].to_numpy()

    for rid in pd.unique(race_ids):
        mask = race_ids == rid
        scores = source.get(rid)
        if not scores:
            continue  # 評価が入っていないレースは触らない
        out[mask] = apply_paddock_adjustment(
            pred[mask], horses[mask], scores, strength=strength
        )
    return out
