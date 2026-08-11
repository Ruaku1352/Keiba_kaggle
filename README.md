# 競馬予想AI（Keiba_kaggle）

Kaggle「JRA日本中央競馬会 Horse Racing Dataset」(1986-2021) を使い、
**オッズに対して割安な馬（市場が過小評価している馬）** を見つけて
期待値プラスの買い目を出すためのコード一式。

進捗まとめの依頼1〜3に対応している。

| 依頼 | 対応ファイル |
|---|---|
| 依頼1：特徴量エンジニアリングの拡充 | `src/features.py` |
| 依頼2：回収率ベースの評価関数 | `src/evaluate.py` |
| 依頼3：期待値ベースの買い目抽出 | `src/betting.py` |
| （前処理・時系列分割） | `src/preprocess.py` |
| （学習〜評価の一気通貫） | `src/train.py` |

---

## Colab での使い方

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ruaku1352/keiba_kaggle.git
%cd keiba_kaggle

from src import train
result = train.run_pipeline("/content/drive/MyDrive/keiba/race_result.csv")
```

これで
**読み込み → 前処理 → 特徴量作成 → 時系列分割 → LightGBM学習 → 回収率評価 → 期待値検証**
まで走り、重要度・回収率・期待値スイープの表が出力される。

戻り値 `result` から個別に触れる：

```python
result.importance        # gain 重要度
result.pred              # テスト期間の予測確率
result.test_df           # テスト期間のデータ
result.feature_cols      # 使った特徴量
```

### メモリ・時間がきついとき

```python
# 特徴量は全期間から計算した上で、2000年以降だけを学習に使う
result = train.run_pipeline(csv_path, start_year=2000)
```

162万行 × 40特徴量で概ね 1.5〜2.5 GB 程度。Colab 無料枠（12GB）に収まる設計だが、
学習が遅い場合は `start_year` で絞るのが一番効く。
`preprocess.load_race_result(path, usecols=[...])` で読む列を絞るのも有効。

---

## 依頼1：追加した特徴量

### Data Leakage を防ぐ仕組み

全ての過去実績系は **その行自身を含まない過去だけ** から作る。実装は 2 つの道具に統一：

```python
過去の件数 = groupby.cumcount()              # 自分より前の行数
過去の合計 = groupby.cumsum() - 自分の値      # 自分より前の合計
```

`cumsum()` は自分を含む累積和なので、自分の値を引けば「自分より前だけ」になる。
直近N走は **累積和の差分** で取る（`groupby.rolling` は162万行だと遅すぎるため）：

```
直近N走の合計 = (自分より前の累積和) - (N行前時点の累積和)
```

この前提として `preprocess.basic_clean()` が
`[レース日付, レースID, 馬番]` でソートしてから返している。

### 特徴量一覧

**馬の実績系**

| 特徴量 | 内容 |
|---|---|
| 通算出走回数 | その時点までの出走数 |
| 通算勝率 / 通算連対率 / 通算複勝率 | 1着 / 2着以内 / 3着以内の率 |
| 過去3走平均着順 / 過去5走平均着順 | 直近N走の着順平均 |
| 前走着順 | 直前レースの着順 |
| 前走からの日数 | 休養明けかどうか |
| 前走との距離差 | 距離延長／短縮 |
| 前走上がり3F | 末脚の速さ（`後３Ｆタイム` があれば） |
| 同距離帯_過去勝率 | 200m刻みのビンごとの勝率 |
| 同芝ダ_過去勝率 | 芝／ダート別の勝率 |
| 同馬場状態_過去勝率 | 良／重などの適性 |
| 同競馬場_過去勝率 | コース適性 |

> 依頼にあった「同距離帯（±200m）」はスライド窓で groupby できないため、
> **200m刻みのビン**（`距離 // 200`）で代用している。
> 1600m の馬は 1500〜1699m の実績とだけ比較される点に注意。

**騎手系**：騎手通算勝率 / 騎手通算騎乗数 / 騎手直近100走勝率 / 騎手×競馬場_勝率 / 騎手×芝ダ_勝率

**馬×騎手系**：コンビ回数 / コンビ勝率 / 乗り替わりフラグ

**調教師系**：調教師通算勝率 / 調教師通算出走数

**レース単位**：出走頭数 / 通算勝率_レース内順位 / 騎手勝率_レース内順位 / 斤量_レース内平均差

### 同日レースの扱い（重要）

騎手・調教師の勝率は既定で「同じ開催日の、より前のレース」も過去に含む。
実際の運用（当日朝に予測を出す）で厳密にしたい場合：

```python
df = features.add_all_features(df, strict_daily_lag=True)
```

とすると **前日終了時点** の成績に切り替わる。

### 単独で使う場合

```python
from src import preprocess, features

df = preprocess.load_race_result(path)
df = preprocess.basic_clean(df)
df = features.add_all_features(df)
cols = features.feature_columns(df)          # オッズ・人気は除外される
cols = features.feature_columns(df, include_odds=True)  # 比較用に含める
```

---

## 依頼2：回収率で評価する

正解率は無意味（1着は7.6%なので「全部0予測」で92.4%）。実際の馬券収支で見る。

```python
from src import evaluate

# 予測上位1頭に単勝100円
r = evaluate.backtest_top_n(result.test_df, result.pred, n=1)
print(r)                    # レース数 / 的中率 / 回収率 / 収支

# 上位1〜3頭を横並び比較
evaluate.backtest_by_top_n(result.test_df, result.pred, ns=(1, 2, 3))

# オッズ帯ごとの成績（どのゾーンで勝てているか）
evaluate.odds_band_report(result.test_df, result.pred)

# 収支推移のグラフ
evaluate.plot_profit_curve(r)
```

**基準となる数字**

| 指標 | 意味 |
|---|---|
| 回収率 80% 前後 | 適当に買った場合（JRAの単勝控除率が約20%） |
| 回収率 100% | 損益トントン |
| `evaluate.favorite_baseline(test_df)` | 1番人気を買い続けた場合。**モデルはこれを超えて初めて意味がある** |
| `evaluate.random_baseline(test_df)` | ランダムに1頭選んだ場合 |

検証は必ず時系列で後半20%（`preprocess.time_series_split`）だけを使う。

---

## 依頼3：期待値ベースの買い目抽出

```
期待値 EV = 予測確率 × 単勝オッズ
```

EV > 1.0 の馬 = 市場が過小評価している馬。

```python
from src import betting

# 期待値つきの一覧（prob は レース内で合計1に正規化済み）
ev = betting.compute_expected_value(result.test_df, result.pred)

# 期待値1.0以上を抽出
picks = betting.select_value_bets(result.test_df, result.pred, ev_threshold=1.0)

# 高配当（オッズ10倍以上）だけに絞ったときの回収率
betting.backtest_value_bets(result.test_df, result.pred,
                            ev_threshold=1.0, min_odds=10)

# 閾値を振って安定性を確認
betting.ev_threshold_sweep(result.test_df, result.pred)

# オッズ下限ごとの成績
betting.high_odds_report(result.test_df, result.pred)
```

**確率の正規化について**：LightGBM の binary 出力はレース内で合計1にならない。
そのまま掛けると期待値が系統的にずれるので、`compute_expected_value` は
レース単位で正規化してから期待値を計算している。

**閾値スイープの読み方**：閾値を上げるほど点数が減って回収率が上がるなら、
モデルの期待値はある程度信頼できている。バラバラならノイズを拾っているだけ。

**賭け金の配分**

```python
picks = betting.select_value_bets(test_df, pred, ev_threshold=1.2, min_odds=5)
betting.bet_amounts(picks, bankroll=100000, use_kelly=True)  # ケリー基準（上限5%）
```

**本番レースの確認**

```python
betting.race_picks(test_df, pred, race_id="2026...", top=8)
```

---

## テスト

リーク検出を含めたテストを同梱している（実データ不要、合成データで走る）。

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

主な検証内容：

- 初出走行では過去実績が空（0回 / NaN）になっている
- 通算勝率が「自分より前の行だけ」の平均と厳密に一致する
- 過去3走平均着順が直前3走の平均と一致する
- **後半のレース結果をシャッフルしても前半の特徴量が1ミリも変わらない**（＝未来を見ていない決定的な証拠）
- 回収率の計算が定義どおり（払戻＝オッズ×100、外れは0）
- レース内の正規化確率の合計が1

pandas 2.2 / 3.0 の両方で通ることを確認済み。

---

---

# v2：熱い当たりを狙う（依頼A〜E）

**目的関数が変わった。** 回収率の最大化ではなく
「投資額に対して大きなリターンが返る確率」＝**達成率**の最大化。
回収率60%でも、当たったときに熱ければ勝ち。

| 予算 | 熱い基準 |
|---|---|
| 1,000円 | 10,000円 |
| 3,000円 | 20,000円 |
| 5,000円 | 30,000円 |
| 10,000円 | 50,000円 |

| 依頼 | 対応ファイル |
|---|---|
| A：脚質・展開特徴量 | `src/corner.py` |
| B：ペース特徴量 | `src/pace.py` |
| C：重賞限定の3連単検証 | `src/trifecta.py` / `src/payout.py` |
| D：パドック補正 | `src/paddock.py` |
| E：予測記録 | `src/records.py` |
| （リーク対策の道具箱を分離） | `src/leakfree.py` |

## v2の使い方

```python
from src import train

result = train.run_pipeline(
    "/content/drive/MyDrive/race_result.csv",
    corner_path="/content/drive/MyDrive/20020615-20210731_corner_passing_order.csv",
    lap_path="/content/drive/MyDrive/19860105-20210731_laptime.csv",
    odds_path="/content/drive/MyDrive/19860105-20210731_odds.csv",
)

result.verdict          # v2ベースラインと比べて改善したかの判定表
result.strategy_all     # 買い方の総当たり（全レース）
result.strategy_graded  # 同（重賞のみ）
result.graded_vs_flat   # 重賞 vs 平場
result.gap_result       # 乖離ベースの成績（最重要の判定対象）
```

`odds.csv` の列名が読めない場合は先に確認する：

```python
from src import payout
payout.describe_odds_columns()   # 3連単らしい列に印が付く
payout.load_trifecta_payout(path, payout_col="3連単払戻金")  # 明示指定
```

---

## 依頼A：脚質・展開特徴量

### 通過順文字列のパース

```
(*13,6)4,10-2,9,12,3(7,11)1=8,5
   ↓
{13:1, 6:1, 4:3, 10:4, 2:5, 9:6, 12:7, 3:8, 7:9, 11:9, 1:11, 8:12, 5:13}
```

括弧内は同順位、次は頭数分だけ飛ぶ（競技順位方式）。
`-` `=` `*` 連続空白は**すべて単なる区切り**として捨てている。
差の大小まで数値化しても、下流の「相対位置の平均」ではほぼ効かないため、
まず単純で壊れにくい実装を選んだ。

### 位置の正規化

```
4角相対位置 = (通過順位 - 1) / (出走頭数 - 1)     0.0=先頭, 1.0=最後方
```

頭数が8頭のレースと18頭のレースで「5番手」の意味が違うので、必ず割って揃える。

### 特徴量

**馬ごと（過去のみ）**：脚質スコア / 脚質カテゴリ（逃げ・先行・差し・追込） /
脚質安定度（過去の相対位置の標準偏差） / 脚質サンプル数 / 脚質スコア直近3走 /
前走4角相対位置 / 好走時脚質スコア

**レース内の展開**：レース内逃げ馬頭数 / レース内先行馬頭数 /
脚質スコア_レース内順位 / 脚質スコア_レース内平均差 / 単騎逃げフラグ / 前傾度

**相互作用**：枠相対位置 / 枠×脚質 / 内枠先行度 / 距離×脚質

> 決定木は掛け算を自力で作れないので、`枠×脚質` のような積は明示的に列にしている。

**展開特徴量がリークしない理由**：材料は「過去実績から作った脚質スコア」だけで、
当日の通過順は一切使っていない。レース前に確定する情報のみで構成されている。

---

## 依頼B：ペース特徴量

### 壊れた列への対処

`laptime.csv` の「前半3ハロン」「上がり3ハロン」は**ラップ1と同じ値**が入っていて壊れている。
使わずにラップから計算し直す：

```
前半3F   = ラップ1 + ラップ2 + ラップ3       （= ペース3 と一致）
上がり3F = 有効なラップの末尾3本の合計
ペース指標 = 前半3F - 上がり3F              （プラス＝ハイペース）
```

末尾3本の取り出しは `np.take_along_axis` で行ごとに違う位置を一括抽出している
（有効ラップ数 n の行なら index n-3, n-2, n-1）。
ラップ列がない場合はペース列（累積タイム）の差分から復元するフォールバックも入れた。

### 条件別の標準化

生の秒数は距離・馬場で全然違うので、**(芝ダート × 400m刻みの距離帯) ごとに z 化**する。
この平均・標準偏差にも過去のレースだけを使う。

### 特徴量

前走ペース指標 / 経験ペース平均 / 好走時ペース平均 /
ハイペース時複勝率 / スローペース時複勝率 / **ペース巧拙差** /
過去上がり3F偏差 / 前走上がり3F偏差 /
想定ペース / 想定ペース適性差 / 前走比ペース変化 / 想定ペース×巧拙

### 想定ペースの作り方

事前に実ペースは分からないので、脚質構成から推定する。
ヒューリスティックな係数を手で決めるのではなく、**過去に同じような構成のレースが
実際に何 z のペースになったか**を学習する形にした。

1. 前傾度（逃げ・先行馬の密度）を 0.05 刻みのビンに分ける
2. 過去のレースで、そのビンが実際に何 z になったかの平均を取る
3. その平均を今日の想定ペースとして使う

### ⚠ ペース特有のリーク（対処済み）

ペースは**レース単位の値**なので、馬の行のまま `cumsum` すると
**同じレースの他の馬の行**が「過去」として混ざり、
自分のレースのペースを見てしまう。

対策として `_race_frame()` で1行1レースに畳んでから累積し、最後に馬の行へ配り直している。
`test_no_same_race_pace_leak` が「自分のレースのペースだけを99に書き換えても
自分の行の特徴量が変わらない」ことを検証している。

---

## 依頼C：3連単の買い方検証

### 計算量の話

3連単BOXは点数が多い（5頭BOXで60点）が、**当たる組み合わせは必ず1つだけ**
（実際の1-2-3着の並び）なので全点を展開する必要はない：

```
的中 ⇔ 実際の1着馬の予測順位 ≤ n1
      かつ 2着馬の予測順位 ≤ n2
      かつ 3着馬の予測順位 ≤ n3
```

22,956レースでも一瞬で終わる。

### 点数

```
点数 = n1 × (n2 - 1) × (n3 - 2)
```

1着に n1 通り、2着は1着で使った1頭を除いて n2-1 通り、3着はさらに2頭除いて n3-2 通り。
BOX(k) は k(k-1)(k-2) で、3頭BOX=6点・4頭BOX=24点・5頭BOX=60点と既存の表に一致する。

> **フォーメーションの点数について**：依頼書の「軸1→3→4 (6点)」などは
> 定義が読み取れなかった（上の式だと 1×2×2 = 4点になる）。
> こちらは上の定義で計算し、**点数を表に出している**ので、
> 投資額でv2の表と突き合わせてほしい。BOXの点数は完全に一致している。

### 使い方

```python
from src import trifecta

race = trifecta.build_race_table(test_df, pred, payout_df)  # 1行1レース

trifecta.strategy_table(race)                    # 全レース
trifecta.strategy_table(race, graded_only=True)  # 重賞のみ
trifecta.compare_graded_vs_flat(race)            # 重賞 vs 平場（有意判定つき）
trifecta.simulate_gap_box(test_df, pred, payout_df)  # 乖離ベース
```

学習は全レースのまま、**評価だけ**を重賞に絞る設計（学習データを減らさない）。

### 信頼区間は Wilson を使った

依頼書にあった `p ± 1.96*sqrt(p(1-p)/n)` も `normal_interval()` として残してあるが、
既定は **Wilson score interval** にした。理由は、達成率が数%と極端に小さく、
重賞は母数も小さいため、素朴な近似だと

- 下限が負に飛び出す（達成率がマイナス、という無意味な値になる）
- 0件のとき幅が0になる（「絶対に起きない」と誤読される）

Wilson 区間は0件でも `[0, 上限]` という妥当な区間を返す。
`compare_graded_vs_flat()` の `有意` 列は、重賞と平場の区間が重ならないかで判定している。

### 判定

`trifecta.verdict()` が依頼書セクション6の4指標をv2ベースラインと自動比較する。

| 指標 | v2ベースライン |
|---|---|
| valid AUC | 0.7787 |
| 上位3頭BOX 達成率 | 1.015% |
| 上位5頭BOX 達成率 | 2.919% |
| **乖離BOX 達成率** | **0.065%** ← 最重要 |

乖離BOXが改善するなら、新特徴量が市場の知らない情報を捉えたことになる。

---

## 依頼D：パドック補正

```python
from src import paddock

adjusted = paddock.apply_paddock_adjustment(
    pred, horses, {"ドウデュース": 0.8, "スターズオンアース": -0.5}, strength=0.5)

paddock.adjustment_report(pred, horses, scores, strength=0.5, odds=odds)
```

### なぜ logit で足すのか

確率を直接1.5倍すると、元が0.8の馬は1.2になって確率でなくなる。
そこで logit（対数オッズ）に変換してから加算する：

```
logit(p) = log(p / (1-p))        0〜1 を -∞〜+∞ に開く
sigmoid(x) = 1 / (1 + exp(-x))   その逆変換
```

**logit空間での加算は、オッズを定数倍することと等価**：

```
logit(p) + a  ⇔  odds → odds × e^a
```

つまり `strength=0.5`・スコア`+1.0` なら「オッズが約1.65倍」の補正。
確率が0や1を超えることは原理的にありえないし、
元が0.02の馬と0.6の馬に同じ補正を掛けても「掛け算として同じ強さ」になる。
これが確率を直接いじるより自然な理由。

`test_adjustment_is_odds_multiplication` がこの等価性を数値で検証している。

### 入力の差し替え

`PaddockInput` を継承すれば入力元を差し替えられる。
`DictPaddockInput`（辞書）と `CsvPaddockInput`（CSV）を同梱。
Googleスプレッドシート連携は `get()` を実装したクラスを1つ足すだけで済む。

---

## 依頼E：予測記録

```python
from src import records

log = records.PredictionLog("/content/drive/MyDrive/keiba_log")

log.record_race(race_id, race_df, pred, scores, adjusted, race_name="有馬記念")
log.record_result(race_id, result_df)      # レース後に着順を追記
log.record_bet(race_id, "3連単BOX", "1-2-3", 6, 600, 4300)

log.evaluate_paddock_skill()   # スコア帯ごとの勝率・複勝率
log.compare_auc()              # 補正前AUC vs 補正後AUC
```

補正前と補正後の確率を**両方**残すのがポイント。
片方だけでは、当たったときにモデルの手柄なのか自分の目の手柄なのか区別できない。
1年運用すれば、スコア帯ごとの複勝率が右肩上がりかどうかで
自分のパドック評価に予測力があるかを判定できる。

---

## v2で追加したテスト

```bash
python -m pytest tests/ -q     # 46件（v1: 11件 + v2: 35件）
```

新規のリーク検証は2種類：

1. **未来シャッフル不変性**（拡張）
   着順に加えて、未来レースの**通過順とペースも**書き換えて、
   過去行の脚質・ペース特徴量が変わらないことを確認

2. **同一レース内リーク**（新規）
   自分のレースの通過順／ペースだけを書き換えて、
   自分の行の特徴量が変わらないことを確認。
   レース単位の値を扱うとき特有の漏れ方で、これを踏むと
   「未来シャッフル不変性は通るのにリークしている」状態になる

ほかに、通過順パーサ（依頼書の実データ例そのまま）、壊れたラップ列の無視、
熱い基準の対応表、BOX点数、Wilson区間、logit補正の等価性、記録の往復を検証している。
