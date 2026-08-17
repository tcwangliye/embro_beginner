"""公共数据契约模块（被各脚本导入，不单独运行）。

集中管理：
  - 全项目路径常量
  - 列名约定（患者 ID / 胚胎键 / 标签映射）
  - 移植前临床特征列表
  - 数据清洗管道 load_cohort()
  - 统一评估指标 metric_dict()

拿到真实数据后，如果列名与下方约定不符，只需修改本文件的常量即可。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # 原始数据（手工放入）
PROCESSED_DIR = DATA_DIR / "processed"    # 中间产物（清单 / 特征）
EXTERNAL_DIR = DATA_DIR / "external"      # 预训练权重缓存
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_DIR = RESULTS_DIR / "oof"
MODEL_DIR = RESULTS_DIR / "models"
METRIC_DIR = RESULTS_DIR / "metrics"
AUDIT_DIR = RESULTS_DIR / "audit"
LOGS_DIR = PROJECT_ROOT / "logs"

# ------------------------------------------------------------
# 外部数据根目录（解压位置，与项目代码分离）
# ------------------------------------------------------------
EXT_DATA_ROOT = Path("/home/storage/wy/embro")   # embryo_all_data.tgz 解压根目录
EMBRYO_NEW = EXT_DATA_ROOT / "data" / "cy0626" / "embryo_new"
CLINICAL_DIR = EMBRYO_NEW / "processed_0507_clinical"   # 临床主表 + 标签冲突表
FEMI_ROOT = EMBRYO_NEW / "Timelapse_femi_processed"     # FEMI 处理后帧目录（含 manifest）
TIMELAPSE_ROOT = EMBRYO_NEW / "Timelapse_1246"          # 原始延时帧（包内缺失，symlink 目标）
VATEP_DIR = EXT_DATA_ROOT / "zcy" / "embryo_live2" / "experiments_0507_best_table"
VIDEO_CACHE_DIR = VATEP_DIR / "video_cache_paper_12x48"  # VaTEP 训练缓存（1332 个 .npy）
PRETRAINED_PTH = VATEP_DIR / "pretrain_pseudo" / "encoder.pth"  # VaTEP 预训练编码器

# 默认数据路径（按真实数据修正）
CLINICAL_CSV = CLINICAL_DIR / "clinical_cleaned_full.csv"
CONFLICT_CSV = CLINICAL_DIR / "label_conflict_report.csv"      # 标签冲突样本（VaTEP 用）
VIDEO_MATCHES_CSV = FEMI_ROOT / "femi_processed_manifest.csv"  # 胚胎 -> 视频目录 匹配表
UNIQUE_VIDEO_MATCHES_CSV = VIDEO_MATCHES_CSV                    # 兼容旧命名（真实数据只有 manifest）

# 中间产物路径
FRAME_MANIFEST_CSV = PROCESSED_DIR / "frame_manifest.csv"
FRAME_FEATS_CSV = PROCESSED_DIR / "frame_feats.csv"     # 帧级特征（03 输出）
VIDEO_FEATS_CSV = PROCESSED_DIR / "video_feats.csv"     # 视频级特征（04 输出）


def remap_legacy_path(p: str) -> str:
    """把旧机器绝对路径（/home/data/cy0626/embryo_new/...）映射到当前机器。

    压缩包内 manifest / 匹配表的目录列是源机器绝对路径，本机解压后
    需要把前缀替换为本机实际位置。
    """
    old_root = "/home/data/cy0626/embryo_new"
    if old_root in str(p):
        return str(p).replace(old_root, str(EMBRYO_NEW))
    return str(p)

# ============================================================
# 数据契约（列名约定，按真实临床表修正）
# ============================================================
PATIENT = "clean_patient_id"            # 患者 ID 列（清洗后，如 '140592'）
EMBRYO = "clean_embryo_no"              # 胚胎号列（患者内唯一，如 '6.3'）
KEY = "patient_embryo_key"              # 患者-胚胎唯一键（表内已有，如 '140592__6.3'）
LABELS = {"fh": "clean_fetal_heartbeat_label", "lb": "clean_live_birth_label"}  # 任务名 -> 标签列名
TASKS = tuple(LABELS)

# 移植前可获得的临床数值特征（10 个，全部来自清洗后列）
CLINICAL_FEATURES = [
    "clean_女年龄",        # age
    "clean_男年龄",        # male age
    "clean_不孕年限",      # infertility years
    "clean_BMI",
    "clean_基础FSH",
    "clean_基础LH",
    "clean_基础E2",
    "clean_AMH",
    "clean_Gn总量",
    "clean_转内膜日内膜厚度",
]

# 帧图像扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# 数据清洗管道
# ============================================================
def load_cohort(
    clinical_path: Path | str = CLINICAL_CSV,
    task: str = "lb",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """加载并清洗临床数据。

    返回 (df, y, groups, audit)：
      df      清洗后的 DataFrame（含 KEY / PATIENT / 标签 / 特征列）
      y       标签数组（0/1）
      groups  患者分组数组（用于 StratifiedGroupKFold，防同患者跨折泄漏）
      audit   清洗过程行数变化审计字典
    """
    if task not in TASKS:
        raise ValueError(f"task 必须是 {TASKS} 之一，收到 {task!r}")
    label_col = LABELS[task]
    feats = list(features) if features else list(CLINICAL_FEATURES)
    required = [PATIENT, EMBRYO, label_col] + feats
    audit: dict = {"input_rows": 0}

    path = Path(clinical_path)
    if not path.exists():
        raise FileNotFoundError(
            f"临床数据不存在：{path}\n请将原始临床表放入 data/raw/ 并确认文件名。"
        )
    df = pd.read_csv(path)
    audit["input_rows"] = len(df)

    # fail-fast：必选字段缺失直接抛异常，不静默处理
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"临床表缺少字段：{missing}\n表内实际列：{list(df.columns)}")

    # 容错类型转换：脏数据（文本/空串）-> NaN
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # 过滤：二分类标签 + 有效患者 ID + 有效胚胎号
    n0 = len(df)
    df = df[df[label_col].isin([0, 1])]
    audit["removed_bad_label"] = n0 - len(df)
    df = df[df[PATIENT].astype(str).str.strip() != ""]
    df = df[df[EMBRYO].astype(str).str.strip() != ""]

    # 患者-胚胎唯一键：表内已有则直接用（真实数据为 '140592__6.3'），否则构造
    if KEY not in df.columns:
        df[KEY] = df[PATIENT].astype(str).str.strip() + "__" + df[EMBRYO].astype(str).str.strip()
    n1 = len(df)
    df = df.drop_duplicates(subset=KEY, keep="first")
    audit["duplicates_removed"] = n1 - len(df)

    y = df[label_col].to_numpy(dtype=float).astype(int)
    groups = df[PATIENT].to_numpy()
    audit["final_rows"] = len(df)
    audit["n_patients"] = int(df[PATIENT].nunique())
    audit["n_positive"] = int(y.sum())
    return df.reset_index(drop=True), y, groups, audit


# ============================================================
# 统一评估
# ============================================================
def metric_dict(y, probability: np.ndarray) -> dict:
    """统一评估：返回 AUROC + AUPRC。

    类别不平衡场景下 AUPRC 比 AUROC 更真实地反映模型对少数类
    （成功妊娠）的排序能力。
    """
    y = np.asarray(y)
    prob = np.asarray(probability)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
    }


# ============================================================
# 通用工具
# ============================================================
def natural_key(name: str) -> list:
    """自然排序键：frame_2 < frame_10（数字按数值比较而非字典序）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def ensure_dirs() -> None:
    """创建全部输出目录（幂等）。"""
    for d in (
        RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR,
        OOF_DIR, MODEL_DIR, METRIC_DIR, AUDIT_DIR, LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_no_leakage(groups: np.ndarray, train_idx, valid_idx, patients_col: str = PATIENT) -> None:
    """双重断言：患者不跨折（每组患者只出现在一个折中）。"""
    tr_patients = set(groups[train_idx])
    va_patients = set(groups[valid_idx])
    assert tr_patients.isdisjoint(va_patients), "患者跨折泄漏：同一患者同时出现在训练集与验证集！"


def write_oof_csv(df: pd.DataFrame, y: np.ndarray, oof: np.ndarray, folds: np.ndarray, path: Path | str, label_col: str) -> None:
    """将 OOF 预测统一落盘。"""
    out = pd.DataFrame({
        KEY: df[KEY],
        PATIENT: df[PATIENT],
        label_col: y,
        "fold": folds,
        "oof_probability": oof,
    })
    out.to_csv(path, index=False)
