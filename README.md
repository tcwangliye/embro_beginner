# 胚胎多模态活产概率预测（embro_beginner）

用 **临床表格** + **胚胎 timelapse 视频**（D1–D5 延时摄影）预测胚胎移植后的**活产（Live Birth）概率**。纯入门练习项目。

## 项目简介

| 项 | 说明 |
|---|---|
| 输入模态 1 | 临床表格：年龄、BMI、激素水平等移植前可获得的数值特征 |
| 输入模态 2 | timelapse 视频：胚胎培养过程的延时帧序列（经 FEMI 预训练 ViT 提取视觉特征） |
| 预测目标 | LB（活产）概率，可选 FH（胎心） |
| 建模方法 | CatBoost（梯度提升树）+ 五折**患者分组**交叉验证 |
| 核心对比 | 纯临床 / 纯视频 / 多模态融合 三路公平对比 |

**两条红线**：
1. **时间一致性**——只用移植前可获得的信息（D1–D5 帧 + 移植前临床指标）
2. **患者分组防泄漏**——同一患者的多个胚胎不得跨折（`StratifiedGroupKFold`）

## 目录结构

```
embro_beginner/
├── PLAN.md                # 执行计划
├── README.md              # 本文件
├── requirements.txt       # 一键安装依赖（pip install -r requirements.txt）
├── data/
│   ├── raw/               # 原始数据（手工放入：临床表 + 视频匹配表 + 帧目录）
│   ├── processed/         # 中间产物（帧清单、帧/视频特征）
│   └── external/          # 预训练权重缓存
├── scripts/               # 全部脚本（见下）
├── results/
│   ├── audit/             # 01 数据审计报告
│   ├── oof/               # 各实验 OOF 预测
│   ├── models/            # 每折模型 + PCA pipeline
│   └── metrics/           # 指标 JSON/CSV + 汇总表
├── logs/                  # 运行日志
└── experiments.csv        # 实验注册表（06 生成）
```

## 快速开始

```bash
# 0. 一键安装全部依赖（表格 + 视频建模）
pip install -r requirements.txt
# 无 GPU 时先装 CPU 版 torch 再装其余依赖：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt

# 1. 检查环境（含真实数据就绪状态）
python scripts/00_check_environment.py

# 2. 数据审计（临床表 + 视频 manifest）
python scripts/01_audit_data.py

# 3. 临床基线（先立下限，当前唯一可跑的视频无关管线）
python scripts/02_baseline_clinical.py --task lb

# 4. 视频特征提取 → 聚合（⚠️ 需帧图像，见下方“数据现状”）
python scripts/03_extract_frame_feats.py --limit 10   # 先小样本试跑
python scripts/04_aggregate_video_feats.py

# 5. 三路对比（临床 / 视频 / 融合）
python scripts/05_train_fusion.py --mode clinical --task lb
python scripts/05_train_fusion.py --mode video    --task lb
python scripts/05_train_fusion.py --mode fusion   --task lb

# 6. 汇总对比表
python scripts/06_report.py
```

## 数据现状（真实数据已接入）

真实数据解压于 **`/home/storage/wy/embro`**（代码与数据分离），路径由 `scripts/common.py` 的 `EXT_DATA_ROOT` 统一管理：

| 数据 | 位置 | 状态 |
|---|---|---|
| 临床主表 | `data/cy0626/embryo_new/processed_0507_clinical/clinical_cleaned_full.csv` | ✅ 1597 行×128 列 |
| 标签冲突表 | `.../label_conflict_report.csv` | ✅ 存在 |
| 视频匹配表 | `data/cy0626/embryo_new/Timelapse_femi_processed/femi_processed_manifest.csv` | ✅ 16934 条 |
| FEMI 帧图像 | `.../Timelapse_femi_processed/**/F0/` | ⚠️ **缺失**：16934 个 F0 全为断开的符号链接（指向包内未含的 `Timelapse_1246/`） |
| VaTEP 预训练权重 | `zcy/embryo_live2/experiments_0507_best_table/pretrain_pseudo/encoder.pth` | ✅ 25 MB |
| VaTEP 视频缓存 | `.../video_cache_paper_12x48/` | ✅ 1332 个 `.npy`（19.2 GB） |

> **结论**：表格管线（01/02/05-clinical/06）可直接运行；**帧特征管线（03/04/05-video/fusion）因帧图像缺失不可用**——若需视频建模，建议走 VaTEP 路线（`video_cache` 已就绪）。

## 数据契约（已按真实表修正）

`scripts/common.py` 集中管理列名约定与数据路径：

| 常量 | 约定 | 说明 |
|---|---|---|
| `PATIENT` | `clean_patient_id` | 患者 ID 列（如 `'140592'`） |
| `EMBRYO` | `clean_embryo_no` | 胚胎号列（患者内唯一，如 `'6.3'`） |
| `KEY` | `patient_embryo_key` | 患者-胚胎唯一键（表内已有，如 `'140592__6.3'`） |
| `LABELS` | `{"fh": "clean_fetal_heartbeat_label", "lb": "clean_live_birth_label"}` | 任务名 → 标签列名 |
| `CLINICAL_FEATURES` | 10 个 `clean_` 前缀列 | 女/男年龄、不孕年限、BMI、FSH、LH、E2、AMH、Gn 总量、内膜厚度 |
| `EXT_DATA_ROOT` | `/home/storage/wy/embro` | 外部数据根目录（改机器只需改这一处） |

视频匹配表需含胚胎目录列（`embryo_dir` / `processed_f0` / `video_dir` 任一即可）；manifest 内旧机器绝对路径会自动重映射（`remap_legacy_path`）。

---

## 脚本详解（bash 用法与参数）

### 00_check_environment.py —— 环境就绪检查

逐项检查依赖包与 GPU，一次报告全部缺失项，并给出安装建议。

```bash
python scripts/00_check_environment.py
python scripts/00_check_environment.py --min-cuda
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--min-cuda` | 关 | 开启后 GPU 不可用时返回退出码 1（可作 CI 门禁） |

检查内容：Python 版本、8 个依赖包（pandas/numpy/scikit-learn/catboost/joblib/Pillow/torch/transformers）、GPU（CUDA 可用性、型号、显存、cuDNN）、磁盘剩余、`data/raw/` 就绪状态。

---

### 01_audit_data.py —— 双模态数据审计

在建模前摸清数据质量：临床表全局统计、逐列缺失报告、重复胚胎键、视频帧数分布。

```bash
python scripts/01_audit_data.py
python scripts/01_audit_data.py --clinical data/raw/clinical_table.csv
python scripts/01_audit_data.py --video-matches data/raw/video_matches.csv
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--clinical` | `data/raw/clinical_table.csv` | 临床表路径 |
| `--video-matches` | 自动探测 | 视频匹配表路径（缺省时探测 `video_matches.csv` / `unique_video_matches.csv`） |

输出到 `results/audit/`：`dataset_summary.csv`、`column_report.csv`、`duplicate_embryo_keys.csv`、`video_frame_stats.csv`、`audit_summary.json`。

---

### 02_baseline_clinical.py —— 纯临床表格基线

CatBoost + 五折患者分组 CV，先立参考下限。所有后续融合方案都与它公平对比。

```bash
python scripts/02_baseline_clinical.py --task lb
python scripts/02_baseline_clinical.py --task fh --iterations 500 --seed 42
python scripts/02_baseline_clinical.py --task lb --features age bmi amh
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--task` | `lb` | 预测目标：`fh`（胎心）或 `lb`（活产） |
| `--clinical` | `data/raw/clinical_table.csv` | 临床表路径 |
| `--features` | `CLINICAL_FEATURES` | 自定义特征列（空格分隔多个） |
| `--seed` | `2026` | 随机种子 |
| `--folds` | `5` | 交叉验证折数 |
| `--iterations` | `300` | CatBoost 迭代轮数 |

输出到 `results/`：`oof/baseline_clinical_{task}_oof.csv`、`models/..._fold{i}.cbm`、`metrics/..._metrics.json` 等。

---

### 03_extract_frame_feats.py —— FEMI 帧级特征提取

用预训练 ViT（默认 FEMI）提取每帧视觉嵌入，去 [CLS] token 后对 patch 均值池化。**需要 GPU**。若帧清单不存在会自动生成（扫描胚胎目录 + 自然排序 + stride 采样）。

```bash
python scripts/03_extract_frame_feats.py --limit 10              # 小样本试跑
python scripts/03_extract_frame_feats.py --limit 0 --stride 10   # 全量，更密采样
python scripts/03_extract_frame_feats.py --model google/vit-base-patch16-224-in21k  # 换编码器消融
python scripts/03_extract_frame_feats.py --batch-size 32 --device cuda:0
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `ihlab/FEMI` | HuggingFace 预训练模型 ID |
| `--batch-size` | `16` | 推理批大小 |
| `--device` | `auto` | `cuda` / `cpu` / `auto`（自动检测） |
| `--limit` | `0` | 处理的胚胎数（`0` = 全量） |
| `--stride` | `20` | 帧采样间隔（每隔 N 帧取 1 帧） |
| `--regenerate` | 关 | 强制重新生成帧清单（默认复用已有） |
| `--out` | `data/processed/frame_feats.csv` | 帧级特征输出路径 |

---

### 04_aggregate_video_feats.py —— 视频级特征聚合

把每胚胎的不定长帧序列聚合成固定维度向量（mean/std/delta/last 四种统计，维度 ×4）。

```bash
python scripts/04_aggregate_video_feats.py
python scripts/04_aggregate_video_feats.py --feats data/processed/frame_feats_ihlab_FEMI.csv
python scripts/04_aggregate_video_feats.py --out data/processed/video_feats_v2.csv
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--feats` | `data/processed/frame_feats.csv` | 帧级特征 CSV；默认路径不存在时自动探测最新 `frame_feats_*.csv` |
| `--out` | `data/processed/video_feats.csv` | 视频级特征输出路径 |

---

### 05_train_fusion.py —— 三路建模对比

同一套五折患者分组 CV + 相同 CatBoost 超参下对比三种模式，回答"视频特征能否提升纯临床基线"。

```bash
# 三种模式各跑一次（核心实验）
python scripts/05_train_fusion.py --mode clinical --task lb
python scripts/05_train_fusion.py --mode video    --task lb
python scripts/05_train_fusion.py --mode fusion   --task lb

# 调参示例
python scripts/05_train_fusion.py --mode fusion --task lb --pca-dim 64 --iterations 500
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `fusion` | `clinical`（纯临床）/ `video`（纯视频）/ `fusion`（拼接融合） |
| `--task` | `lb` | 预测目标：`fh` 或 `lb` |
| `--clinical` | `data/raw/clinical_table.csv` | 临床表路径 |
| `--video-feats` | `data/processed/video_feats.csv` | 视频级特征路径（clinical 模式忽略） |
| `--pca-dim` | `32` | 视频特征 PCA 降维目标维数（video/fusion 模式） |
| `--seed` | `2026` | 随机种子 |
| `--folds` | `5` | 交叉验证折数 |
| `--iterations` | `300` | CatBoost 迭代轮数 |

视频特征降维管线（`SimpleImputer → StandardScaler → PCA`）在**每折训练集上 fit**，验证集仅 transform，杜绝数据泄漏。

---

### 06_report.py —— 实验指标汇总

扫描 `results/metrics/*_metrics.json`，打印对比表（AUROC 降序），写入汇总 CSV 并维护实验注册表。

```bash
python scripts/06_report.py
python scripts/06_report.py --metric-dir results/metrics
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--metric-dir` | `results/metrics` | 指标 JSON 所在目录 |

输出：`results/metrics/summary.csv`（本次汇总）、`experiments.csv`（注册表，按 tag 覆盖旧实验）。

---

## 依赖清单

一键安装：`pip install -r requirements.txt`（文件内已按表格/视频分组注释，含版本下限）。下表为各包用途说明：

| 包 | 用途 | 必需阶段 |
|---|---|---|
| pandas / numpy | 表格处理 | 全部 |
| scikit-learn | StratifiedGroupKFold、PCA、评估 | 全部 |
| catboost | 表格/融合模型 | 全部 |
| joblib | pipeline 序列化 | 全部 |
| Pillow | 帧图像加载 | 视频管线 |
| torch | 深度学习后端 | 视频管线（需 GPU） |
| torchvision | torch 配套（图像变换） | 视频管线（需 GPU） |
| transformers | FEMI / ViT 加载 | 视频管线（需 GPU） |

## 常见问题

- **`01` 报临床表缺字段**：按报错提示对照 `common.py` 的 `CLINICAL_FEATURES` / `PATIENT` / `EMBRYO` 修改列名约定。
- **`03` 提示找不到匹配表**：确认 `data/raw/` 下匹配表含 `embryo_dir`（或 `processed_f0`/`video_dir`）列。
- **`03` 在 CPU 上跑**：可以跑但极慢，`--limit` 调小先验证流程。
- **`05` fusion 样本数少于 clinical**：说明部分胚胎缺视频匹配，属正常，见 metrics 里的 `n_matched`。
