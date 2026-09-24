# test40 — SiO₄四面体を1ビーズにしたガラス・結晶の条件付き生成

**1つのモデルを両相で学習し、`--phase glass` / `--phase crystal` で生成を切り替えます。**
標準の粗視化単位は SiO₄ 四面体です。Si と O の全原子参照から四面体の中心を計算し、
そのビーズ配置を学習します。生成は指定相の周期セル内の一様ランダム配置から開始します。

学習済み・品質検証済みのモデルではなく、スパコンで学習・評価するための実装です。
`examples/reference/`はガラス1構造・結晶1構造の**動作確認用データ**です。このデータでの検証lossは
独立な構造への汎化を測りません。既定の`qsub run_test40.pbs`（`STAGE=full`）は、同梱のLAMMPS入力
から結晶(NPT複数レプリカ)・ガラス(NVT複数レプリカ)の多フレーム軌跡を生成して`input/dataset`を
作りますが、それでも汎化性能の保証にはなりません。多数の独立な参照構造を使った評価が必要です。

## 粗視化の定義

Si の周囲の Si–O 距離 2.2 Å 未満の酸素4個を1つの SiO₄ 四面体とします。
各酸素を中心Siに対して最小像でアンラップしてから、次の重心を計算します。

\[
M_i=m_{Si}+\sum_{o\in SiO_4(i)}m_O/n_o,\qquad
R_i=r_{Si,i}+\frac{\sum_o(m_O/n_o)\,\Delta r_{i,o}}{M_i}.
\]

`n_o` はその酸素を共有する四面体の数です。通常は2なので、四面体1個あたりの有効質量は
`28.0855 + 4 × 15.9994 / 2 = 60.0843 amu` です。SiO₄全質量を各ビーズへ割り当てると
共有酸素を二重計上するため、この分配を使います。中心をSi座標で代用する処理ではありません。

- 標準では、全Siが4配位、全Oが2配位であることを確認します。
- `--allow-sharing-defects` を指定すると、Oの共有数が1や3の場合も実際の共有数で質量を分配し、
  欠陥とビーズごとの質量を記録します。Siの4配位と、すべてのOが少なくとも1つの四面体に属する条件は維持します。
- 同梱ガラスには共有数が2でないOが8個あり、このオプションで変換しました。総質量は保存しています。
- 各相のフレーム間で同じ原子順序・直交セル・Si–O接続を要求します。結合交換を含む軌跡は対象外です。
- 1ビーズは四面体の位置だけを表します。配向、内部Si–O結合、O–Si–O角、酸素の逆写像は生成しません。
- 出力の元素ラベル `Si` はビーズ識別用です。`final.extxyz` の質量と `final.data` のMassesはCG質量です。
  欠陥を許した場合の質量・接続表は参照由来の属性で、生成配置が同じ接続を再現した保証ではありません。

## スパコンで使う

Python 3.11、PyTorch 2.6.0 / CUDA 12.4を基準にしています。PBSの既定値は既存の `sg8`、
GPU 1台、CPU 8、メモリ32 GB、20時間です（他のジョブスクリプト同様、このリポジトリでは
`sg8`キューの上限に合わせています）。`STAGE=full`はMD生成も同じ20時間枠に含むため、
レプリカ数・ステップ数（`NREP_CRYSTAL`/`NREP_GLASS`/`MD_NSTEPS`など）が大きいと学習に
残る時間が少なくなります。MD生成に時間がかかる場合は、先に`STAGE=md`だけを独立ジョブとして
実行し（同じ20時間枠を使い切ってもよい。既に生成済みの軌跡は次回スキップされます）、
完了後に`STAGE=full`（またはprepare→train）を投げ直してください。
必要なモジュールは利用環境に合わせてロードしてください。
コンテナ作成は外部ネットワークに接続でき、Singularity/Apptainerのビルドを許可されたホストで実行します。

```bash
git clone git@github.com:haru2225/test40.git
cd test40
singularity build test40.sif Singularity.def
# Apptainerでも可: apptainer build test40.sif Singularity.def

# 既定(STAGE=full)は1回のqsubで完結: 同梱のLAMMPS入力(md/*.in, md/glass_seed.dat)から
# 結晶(NPT)・ガラス(NVT)の多フレーム軌跡を生成し(足りないレプリカのみ; 既にあればスキップ)、
# input/dataset へprepareし、学習まで行います。LAMMPS(Vashishta potential)は
# コンテナの外で動くので、run_test40.pbs内の「site-specific」ブロックで
# 自サイトのLAMMPSモジュール名に書き換え、LAMMPS_POTENTIALSも指定してください。
qsub -P <課題番号> -v LAMMPS_POTENTIALS=/path/to/vashishta/potentials run_test40.pbs

# 段階を分けたい場合はSTAGEで指定できます:
#   STAGE=md      同梱LAMMPS入力から軌跡だけ生成（コンテナ不要）
#   STAGE=prepare 既存の軌跡からinput/datasetをprepareするだけ
#   STAGE=train   既存のinput/dataset（既定）で学習だけ
#   STAGE=generate 生成のみ（下記）
qsub -P <課題番号> -v STAGE=md,LAMMPS_POTENTIALS=/path/to/vashishta/potentials run_test40.pbs
qsub -P <課題番号> -v STAGE=prepare run_test40.pbs

# まず短いGPU動作確認（同梱の examples/reference を使う）。生成品質の検証ではありません。
qsub -P <課題番号> -v STAGE=train,DATASET_PATH=$PWD/examples/reference,UPDATES=20,LOG_EVERY=10,CHECKPOINT_EVERY=10 run_test40.pbs

# 同じ学習を3万更新まで継続
qsub -P <課題番号> -v RESUME=1,UPDATES=30000 run_test40.pbs

# 学習ジョブが完了してから、それぞれ別ジョブで生成
qsub -P <課題番号> -v STAGE=generate,PHASE=glass run_test40.pbs
qsub -P <課題番号> -v STAGE=generate,PHASE=crystal run_test40.pbs
```

MD生成のレプリカ数は`NREP_CRYSTAL`/`NREP_GLASS`（既定5・5）、各軌跡の長さは`MD_NSTEPS`/`MD_DUMP_EVERY`/
`MD_EQ_STEPS`（既定は2 ns、200 fs毎ダンプ、20 ps平衡化）で調整できます。`input/dataset`は既に
存在すれば`STAGE=full`/`prepare`はスキップします（作り直す場合は削除してから実行してください）。

リポジトリが非公開の場合はGitHub認証が必要です。GitHubにアクセスできない計算機では、
認証済みの端末でcloneしたフォルダと作成済み `test40.sif` を転送してください。

既定の入力は同梱の `examples/reference/` です。生成系はガラス1000ビーズ、結晶64ビーズで、
参照相ごとのセルとビーズ数を使います。任意サイズ・任意密度への外挿をするCLIではありません。
同じセルと組成の両相データを用意すれば、サイズ・密度の違いに頼らず相条件の効果を比較できます。

```bash
# 中断したジョブを同じ設定・出力先から再開（学習・生成両方に対応）
qsub -P <課題番号> -v RESUME=1 run_test40.pbs
qsub -P <課題番号> -v STAGE=generate,PHASE=crystal,RESUME=1 run_test40.pbs

# 独立したサンプル
qsub -P <課題番号> -v STAGE=generate,PHASE=glass,SEED=2026 run_test40.pbs

# 複数フレーム(セルが違いうる)データセットで、特定フレームのセルへ生成する
qsub -P <課題番号> -v STAGE=generate,PHASE=crystal,FRAME=2 run_test40.pbs

# 自分のデータで学習を開始（新しい出力先を使う）
qsub -P <課題番号> -v DATASET_PATH=/absolute/path/dataset,TRAIN_DIR=/absolute/path/train run_test40.pbs
```

外部ディレクトリを使う場合は `EXTRA_BIND=/absolute/path` を追加してください。
`SIF_IMAGE`、`CONTAINER_RUNTIME`、`CUTOFF`、`SIGMA_MIN`、`SIGMA_MAX`、`WIDTH`、`LAYERS`、
`LEARNING_RATE`、`TIME_BUDGET_HOURS`、生成時の `STEPS` も環境変数で設定できます。
SIGTERM/SIGINT、または時間予算19.5時間でチェックポイントを保存して終了コード75で終了します。
SIGKILLや突然のノード停止の場合は最後の定期保存から再開します。自動再投入はしません。
CPUでは乱数状態を含む再開一致をテスト済みです。CUDAのscatter演算ではビット単位の再現性は保証しません。

## 自分の全原子データからSiO₄ビーズを作る

固定直交セルの全原子 `extxyz` 軌跡を推奨します。座標・セルはÅ、元素はSi/O、PBCは3方向です。
LAMMPS dataも読み込めます。`--glass`/`--crystal` はそれぞれ複数ファイルを取れます。

各フレームは自分自身のセル長を保持します（NPTの「セル呼吸」があるMDトラジェクトリでもそのまま使えます）。
学習・生成はフレームごとのセル長をモデルへの条件入力として使います。ただし**Si-O結合トポロジー（どのOがどのSiに属すか）は
相ごとに全フレームで完全一致している必要があります**——結合交換は表現できません。これは、独立に別々にクエンチした
複数のガラス構造（ネットワークがサンプルごとに異なる）を「複数フレーム」として束ねることはできない、という意味でもあります。
複数フレームが必要な場合は、同じ構造を追跡した1本の軌跡（時間的にデコリレートした複数スナップショット）を使ってください。

```bash
singularity exec --bind "$PWD:$PWD" test40.sif python test40.py prepare \
  --glass input/glass.extxyz --crystal input/crystal.extxyz \
  --glass-index ::10 --crystal-index ::10 --split-gap 5 --output input/dataset
```

`--glass-index`/`--crystal-index` は相ごとに指定します（例: 1構造だけのクエンチ結果と、長いMD軌跡を同じ`prepare`呼び出しで
混ぜる場合、必要なスライスが相ごとに異なるため）。フレームの末尾10%を検証用に取り分け、`--split-gap` で境界のフレームを
間引けます。近接フレームの時間相関は残るので、十分な時間間隔や独立した軌跡を用意してください。
同じ座標の重複フレームは取り除きます。1構造しかない相は、`--allow-single-reference` を明示した場合だけ受け付けます。
`--repeat-glass 2 2 2` / `--repeat-crystal 2 2 2` で全原子構造を複製してからマッピングできます。
学習cutoffは、全フレームを通して最短セル辺の半分より小さくしてください。既定は5 Åです。

手元のtest32/test33参照と同じファイルから作る例（入力はリポジトリ外の既存ファイル）：

```bash
python test40.py prepare \
  --glass ../DM2/demo/demo_training/simu_data/sio2_3000_glass_0_1k_sample0.dat \
  --glass-input-format lammps-data --allow-single-reference --allow-sharing-defects \
  --crystal ../ScoreMD/md/silica_beta_cristobalite_init.data \
  --crystal-input-format lammps-data \
  --output input/reference
```

LAMMPSの原子種番号はMassesからASEが推定します。Massesがない場合は `--glass-lammps-types 14 8` のように相ごとに
指定できます（`lammps-data`ではMassesからの自動推定に対して`Z_of_type`優先の上書き、`lammps-dump-text`のようにMassesを
持たない形式では必須です）。元のガラスと結晶はtype順序が逆（ガラス: 1=O, 2=Si／結晶: 1=Si, 2=O）なので、
相ごとに別々の`--glass-lammps-types`/`--crystal-lammps-types`を使います。
四面体マッピングに失敗した場合は原子種・距離・参照の配位欠陥を確認してください。
`--mapping-cutoff` で距離を変更できますが、単に検査を通すために広げるのは適切ではありません。

`md/silica_beta_cristobalite.in`（NPT、結晶）のような固定トポロジーMD軌跡を`lammps-dump-text`として複数フレーム
取り込む例（同じ物質でも相ごとに条件が違うことが多いため`--crystal-*`だけを軌跡用に指定）：

```bash
python test40.py prepare \
  --glass ../DM2/demo/demo_training/simu_data/sio2_3000_glass_0_1k_sample0.dat \
  --glass-input-format lammps-data --allow-single-reference --allow-sharing-defects \
  --crystal ../ScoreMD/md/traj_0.lammpstrj ../ScoreMD/md/traj_1.lammpstrj \
  --crystal-input-format lammps-dump-text --crystal-lammps-types 14 8 \
  --crystal-index 1000::300 --split-gap 2 \
  --output input/reference
```

ガラス側で同様に独立フレームを増やすには、既存の`md/silica_beta_cristobalite.in`と同じ考え方の
`md/silica_glass_nvt.in`（新規）でNVT軌跡を生成してください——独立にクエンチした複数のガラス構造を直接束ねることは
できません（上記のトポロジー制約）。

## 結果と評価

- `results/train/checkpoint.pt`: 共有モデル、optimizer、乱数状態、両相メタデータ。
- `results/train/training.json`: 相別・ノイズ別のlossと、ゼロ予測器のloss。
- `results/{phase}-seed1337/final.extxyz`: 最終CG構造とビーズ質量。
- `final.data`: LAMMPS形式の配置。力場は付属しません。
- `positions.npy`: 初期ランダム配置を含む生成過程。物理時間のMD軌跡ではありません。
- `generation.json`: 完了状態と有効フレーム数。中断時は `valid_frames` 以降を読まないでください。

```bash
for phase in glass crystal; do
  singularity exec --bind "$PWD:$PWD" test40.sif python test40.py evaluate \
    --dataset examples/reference --phase "$phase" \
    --sample "results/$phase-seed1337/final.extxyz" \
    --output "results/$phase-evaluation.json"
done
```

JSONとPNGでビーズRDF・離散逆格子点の構造因子を比較します。近接ビーズ数と最短距離も出します。
既定の近接判定は4 Åで、RDFの第1極小に応じて `--bond-cutoff` を調整してください。
この近接数は**ビーズ間**の幾何学的な指標で、内部Siの4配位を検証する値ではありません。
構造因子は `-4 <= h,k,l <= 4` の方向別診断で、粉末XRDや結晶相同定ではありません。
ガラスと結晶それぞれで複数seedを生成し、参照のRDF、ピーク、近接数、異常な重なりを確認してください。
lossが低いだけで生成成功とは判定しません。

## モデルと検証範囲

PyTorchのみのスカラー・ベクトル型の回転同変メッセージパッシングモデルです。
[PaiNN](https://arxiv.org/abs/2102.03150)の考え方を参考にした小型実装で、NequIPそのものではありません。
相ラベル、log σ、セル辺長、ビーズ密度を条件にし、両相のlossを1:1で学習します。
ノイズ後の近傍を毎回作り直すので、test33で見つかった学習・生成のcutoff不一致を避けています。

前向き過程は直交周期セル上のブラウン運動です。wrapped Gaussianの条件付きスコアを画像和・Fourier和で計算し、
`-σ score` を教師にします。最大σはセル内の終端分布がほぼ一様になる値を使用し、生成は対応する逆VE-SDEです。
最小σまで積分して終了し、長いノイズなしpolishは行いません。結果にはσ_minの平滑化が残ります。
背景: [score-based generation](https://arxiv.org/abs/1907.05600)。

局所グラフと相条件だけで結晶の長距離秩序や適切なガラス分布を保証するものではありません。
有限ステップの積分誤差、参照数、モデル容量の影響を検証する必要があります。
まず小ノイズでの復元性能、相別lossがゼロ予測を上回ること、生成時の距離分布を確認し、
その後に学習量・生成ステップ数を増やしてください。

```bash
# ローカルCPUで実装を検証
python -m venv .venv
. .venv/bin/activate
pip install torch==2.6.0
pip install -r requirements.txt
python -m pytest -q
```

テストは周期スコアの解析微分との一致、回転・置換・並進同変性、相/σ条件への応答、
共有酸素の質量保存と周期重心、両相の学習・生成・評価、中断再開を確認します。
合成テストの「glass」は配線確認用の歪ませた結晶であり、ガラス生成の科学的な検証には使いません。

## 同梱参照の出典

`examples/reference/` は以下の1フレームずつから作ったCG配置とマッピングです。
元の原子座標や学習済み重みは同梱していません。入力SHA256は `metadata.json` に保存しています。

- Glass: [DM2](https://github.com/digital-synthesis-lab/DM2), commit `ab5a7e65d0879c5de23859fa191e318f9f70fae0`,
  `demo/demo_training/simu_data/sio2_3000_glass_0_1k_sample0.dat`。元のMIT noticeは `licenses/DM2-MIT.txt`。
- Crystal: ユーザーのScoreMD作業ツリーの `md/silica_beta_cristobalite_init.data`。
  元のScoreMD MIT noticeは `licenses/ScoreMD-MIT.txt`。

`md/`には、`STAGE=md`/`full`が実際のMD軌跡を生成するためのLAMMPS入力・参照構造も同梱しています
（学習済み重みではなく、生成される`.lammpstrj`もリポジトリには含めません、`.gitignore`参照）:

- `md/silica_beta_cristobalite.in` / `md/silica_beta_cristobalite_init.data`: ユーザーのScoreMD
  作業ツリー由来（β-cristobalite 2×2×2、Vashishta SiO2、NPT production）。`licenses/ScoreMD-MIT.txt`。
- `md/silica_glass_nvt.in`: 同じくScoreMD作業ツリー由来のNVT生成スクリプト（新規作成）。
- `md/glass_seed.dat`: [DM2](https://github.com/digital-synthesis-lab/DM2), commit
  `ab5a7e65d0879c5de23859fa191e318f9f70fae0`, `demo/demo_training/simu_data/sio2_3000_glass_100k_sample0.dat`
  （NVT軌跡の開始構造1つ、`examples/reference`の同梱ガラスとは別サンプル）。`licenses/DM2-MIT.txt`。

再配布時にもこれらのnoticeを保持してください。
