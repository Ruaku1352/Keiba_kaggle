"""依頼E：予測とパドック評価の記録を残す。

目的は1つ。**自分のパドック評価に予測力があるかを後から検証すること。**

そのためには「補正前の確率」と「補正後の確率」を両方残す必要がある。
片方だけでは、当たったときにモデルの手柄なのか自分の目の手柄なのか分からない。

■ ファイル構成（CSV追記）
  predictions.csv : レース前に書く。1行 = 1頭
  results.csv     : レース後に着順を追記（predictions と結合して検証する）
  bets.csv        : 実際に買った買い目と結果

1年運用すれば evaluate_paddock_skill() で
「スコアを付けた馬が実際に走ったか」を数字で確認できる。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = [
    "記録日時", "レースID", "日付", "レース名", "馬番", "馬名",
    "予測確率", "パドックスコア", "補正後確率", "単勝オッズ",
    "補正前順位", "補正後順位",
]

RESULT_COLUMNS = ["レースID", "馬番", "馬名", "着順"]

BET_COLUMNS = [
    "記録日時", "レースID", "券種", "買い目", "点数", "投資額", "払戻額", "的中",
]


def _append_csv(path: str, rows: pd.DataFrame, columns: list[str]) -> None:
    """CSV に追記する。無ければヘッダ付きで新規作成。"""
    rows = rows.reindex(columns=columns)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    header = not os.path.exists(path)
    rows.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


@dataclass
class PredictionLog:
    """予測記録の読み書き。

    >>> log = PredictionLog("/content/drive/MyDrive/keiba_log")
    >>> log.record_race(race_id, race_df, pred, scores, adjusted)
    """

    directory: str = "."

    @property
    def prediction_path(self) -> str:
        return os.path.join(self.directory, "predictions.csv")

    @property
    def result_path(self) -> str:
        return os.path.join(self.directory, "results.csv")

    @property
    def bet_path(self) -> str:
        return os.path.join(self.directory, "bets.csv")

    # -- レース前 ----------------------------------------------------------
    def record_race(self, race_id: str, race_df: pd.DataFrame,
                    pred: np.ndarray, paddock_scores: dict[str, float] | None = None,
                    adjusted: np.ndarray | None = None,
                    race_name: str = "", date: str | None = None,
                    horse_col: str = "馬名", post_col: str = "馬番",
                    odds_col: str = "単勝オッズ") -> pd.DataFrame:
        """1レース分の予測を記録する。戻り値は書き込んだ内容。"""
        paddock_scores = paddock_scores or {}
        pred = np.asarray(pred, dtype="float64")
        horses = race_df[horse_col].astype(str).to_numpy()

        # 補正後が渡されなければ補正なしとして扱う
        adjusted = pred if adjusted is None else np.asarray(adjusted, dtype="float64")

        rows = pd.DataFrame({
            "記録日時": datetime.now().isoformat(timespec="seconds"),
            "レースID": str(race_id),
            "日付": date or (race_df["レース日付"].iloc[0] if "レース日付" in race_df else ""),
            "レース名": race_name,
            "馬番": race_df[post_col].to_numpy() if post_col in race_df else np.arange(1, len(race_df) + 1),
            "馬名": horses,
            "予測確率": pred,
            "パドックスコア": [paddock_scores.get(h, np.nan) for h in horses],
            "補正後確率": adjusted,
            "単勝オッズ": race_df[odds_col].to_numpy() if odds_col in race_df else np.nan,
        })
        rows["補正前順位"] = rows["予測確率"].rank(ascending=False, method="first").astype(int)
        rows["補正後順位"] = rows["補正後確率"].rank(ascending=False, method="first").astype(int)

        _append_csv(self.prediction_path, rows, PREDICTION_COLUMNS)
        return rows

    # -- レース後 ----------------------------------------------------------
    def record_result(self, race_id: str, result_df: pd.DataFrame,
                      horse_col: str = "馬名", post_col: str = "馬番",
                      rank_col: str = "着順") -> None:
        """レース後に確定着順を追記する。"""
        rows = pd.DataFrame({
            "レースID": str(race_id),
            "馬番": result_df[post_col].to_numpy(),
            "馬名": result_df[horse_col].astype(str).to_numpy(),
            "着順": pd.to_numeric(result_df[rank_col], errors="coerce").to_numpy(),
        })
        _append_csv(self.result_path, rows, RESULT_COLUMNS)

    def record_bet(self, race_id: str, kind: str, combination: str, points: int,
                   invested: int, payout: float = 0.0) -> None:
        """買った買い目とその結果を残す。"""
        rows = pd.DataFrame([{
            "記録日時": datetime.now().isoformat(timespec="seconds"),
            "レースID": str(race_id),
            "券種": kind,
            "買い目": combination,
            "点数": points,
            "投資額": invested,
            "払戻額": payout,
            "的中": int(payout > 0),
        }])
        _append_csv(self.bet_path, rows, BET_COLUMNS)

    # -- 検証 --------------------------------------------------------------
    def load_joined(self) -> pd.DataFrame:
        """予測記録と着順を結合して返す（検証用）。"""
        if not os.path.exists(self.prediction_path):
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        preds = pd.read_csv(self.prediction_path, dtype={"レースID": str})
        if not os.path.exists(self.result_path):
            return preds
        results = pd.read_csv(self.result_path, dtype={"レースID": str})
        return preds.merge(results[["レースID", "馬名", "着順"]],
                           on=["レースID", "馬名"], how="left")

    def evaluate_paddock_skill(self) -> pd.DataFrame:
        """パドック評価に予測力があったかを集計する。

        見方:
          - スコア帯ごとの複勝率が右肩上がりなら、目に予測力がある
          - `補正後AUC > 補正前AUC` なら、補正が予測を改善している
        """
        joined = self.load_joined()
        if "着順" not in joined.columns or joined["着順"].isna().all():
            return pd.DataFrame()

        joined = joined.dropna(subset=["着順"])
        joined["複勝"] = (joined["着順"] <= 3).astype(int)
        joined["勝ち"] = (joined["着順"] == 1).astype(int)

        # スコア帯ごとの成績
        bins = [-1.01, -0.5, -0.01, 0.01, 0.5, 1.01]
        labels = ["-1.0〜-0.5", "-0.5〜0", "0(評価なし)", "0〜+0.5", "+0.5〜+1.0"]
        joined["スコア帯"] = pd.cut(
            joined["パドックスコア"].fillna(0), bins=bins, labels=labels
        )
        table = joined.groupby("スコア帯", observed=True).agg(
            頭数=("複勝", "size"), 勝率=("勝ち", "mean"), 複勝率=("複勝", "mean"),
        ).reset_index()
        return table

    def compare_auc(self) -> dict:
        """補正前後の AUC を比較する（sklearn があれば）。"""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return {"error": "scikit-learn が必要です"}

        joined = self.load_joined().dropna(subset=["着順"])
        if joined.empty or joined["着順"].nunique() < 2:
            return {"error": "着順の記録が足りません"}

        y = (joined["着順"] == 1).astype(int)
        if y.nunique() < 2:
            return {"error": "1着の記録が足りません"}
        return {
            "n": int(len(joined)),
            "補正前AUC": float(roc_auc_score(y, joined["予測確率"])),
            "補正後AUC": float(roc_auc_score(y, joined["補正後確率"])),
        }
