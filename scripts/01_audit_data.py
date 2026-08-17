"""01_audit_data.py —— 双模态数据审计（临床表 + 视频目录）。

在建模之前摸清数据质量，产出（results/audit/）：
  dataset_summary.csv        全局统计：行数 / 患者数 / 标签分布
  column_report.csv          逐列类型 / 缺失数 / 缺失率 / 唯一值数（缺失率降序）
  duplicate_embryo_keys.csv  重复胚胎键的完整记录
  video_frame_stats.csv      每胚胎帧数统计（自然排序后按图像扩展名统计）
  audit_summary.json         机器可读的审计摘要

用法示例：
  python scripts/01_audit_data.py
  python scripts/01_audit_data.py --clinical data/raw/clinical_table.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    AUDIT_DIR, CLINICAL_CSV, FEMI_ROOT, IMAGE_EXTS, KEY, LABELS, PATIENT,
    TASKS, VIDEO_MATCHES_CSV, UNIQUE_VIDEO_MATCHES_CSV, ensure_dirs, natural_key,
    remap_legacy_path,
)


# ------------------------------------------------------------
# 临床审计
# ------------------------------------------------------------
def audit_clinical(clinical_path: Path) -> dict:
    """审计临床表：全局统计 + 逐列报告 + 重复键。"""
    df = pd.read_csv(clinical_path)
    summary: dict = {
        "source": str(clinical_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "n_patients": int(df[PATIENT].nunique()) if PATIENT in df.columns else None,
        "n_unique_keys": int(df[KEY].nunique()) if KEY in df.columns else None,
    }

    # 标签分布（保留 NaN 计数）
    for task in TASKS:
        label = LABELS[task]
        if label in df.columns:
            counts = df[label].value_counts(dropna=False)
            summary[f"label_{task}"] = {
                str(k): int(v) for k, v in counts.items()
            }

    # 逐列报告
    col_report = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "n_missing": df.isna().sum().values,
        "missing_rate": df.isna().mean().round(4).values,
        "n_unique": df.nunique(dropna=False).values,
    }).sort_values("missing_rate", ascending=False)
    col_report.to_csv(AUDIT_DIR / "column_report.csv", index=False)

    # 重复胚胎键
    if KEY in df.columns:
        dup_keys = df[df.duplicated(subset=KEY, keep=False)].sort_values(KEY)
        dup_keys.to_csv(AUDIT_DIR / "duplicate_embryo_keys.csv", index=False)
        summary["duplicate_keys"] = int(dup_keys[KEY].nunique())

    pd.DataFrame([summary]).to_csv(AUDIT_DIR / "dataset_summary.csv", index=False)
    return summary


# ------------------------------------------------------------
# 视频审计
# ------------------------------------------------------------
def resolve_matches_path(arg: str | None) -> Path | None:
    """解析视频匹配表路径：优先命令行参数，其次常见默认文件名。"""
    if arg:
        return Path(arg)
    for cand in (VIDEO_MATCHES_CSV, UNIQUE_VIDEO_MATCHES_CSV):
        if cand.exists():
            return cand
    return None


def audit_video(matches_path: Path | None) -> dict | None:
    """审计视频目录：匹配表存在性 + 每胚胎帧数分布。"""
    if matches_path is None or not matches_path.exists():
        print(f"[跳过] 未找到视频匹配表（{VIDEO_MATCHES_CSV} 或 {UNIQUE_VIDEO_MATCHES_CSV}）")
        return None

    m = pd.read_csv(matches_path)
    # 兼容两种可能的目录列名
    dir_col = next((c for c in ("embryo_dir", "processed_f0", "video_dir") if c in m.columns), None)
    if dir_col is None:
        print(f"[警告] 匹配表缺少胚胎目录列（embryo_dir / processed_f0 / video_dir），仅做行数统计")
        print(f"        匹配表列：{list(m.columns)}")
        return {"n_matches": int(len(m))}

    # manifest 内的目录列是旧机器绝对路径，先映射到本机
    m["_dir"] = m[dir_col].map(remap_legacy_path)

    # 目录存在性校验
    missing_dirs = [str(p) for p in m["_dir"] if not Path(p).exists()]
    print(f"胚胎目录：{len(m)} 个，磁盘缺失 {len(missing_dirs)} 个")
    if missing_dirs and len(missing_dirs) == len(m):
        print(f"[警告] 所有胚胎目录都不存在：{missing_dirs[0]}")
        print(f"        提示：压缩包内 Timelapse_femi_processed 仅含 manifest/summary（symlink 模式），"
              f"帧图像本体在 Timelapse_1246/（包内未包含）。")
        return {"n_matches": int(len(m)), "n_missing_dirs": int(len(missing_dirs)), "all_dirs_missing": True}

    # 帧数统计（自然排序后统计图像文件数）
    records = []
    for row in m.itertuples(index=False):
        d = Path(row._dir)
        if not d.exists():
            records.append({"embryo_dir": str(d), "n_frames": 0})
            continue
        frames = sorted(
            [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: natural_key(p.name),
        )
        records.append({"embryo_dir": str(d), "n_frames": len(frames)})
    stats = pd.DataFrame(records)
    if stats.empty:
        print("[警告] 无任何胚胎目录可统计（目录缺失或为空）")
        return {"n_matches": int(len(m)), "n_missing_dirs": int(len(missing_dirs)), "no_frame_stats": True}
    stats.to_csv(AUDIT_DIR / "video_frame_stats.csv", index=False)

    summary = {
        "n_matches": int(len(m)),
        "n_missing_dirs": int(missing_dirs.__len__()),
        "n_embryos_no_frames": int((stats["n_frames"] == 0).sum()),
        "frame_stats": {
            "min": int(stats["n_frames"].min()),
            "median": float(stats["n_frames"].median()),
            "max": int(stats["n_frames"].max()),
            "mean": float(stats["n_frames"].mean()),
        },
    }
    return summary


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="双模态数据审计")
    parser.add_argument("--clinical", default=str(CLINICAL_CSV), help="临床表路径")
    parser.add_argument("--video-matches", default=None, help="视频匹配表路径（缺省自动探测）")
    args = parser.parse_args()

    ensure_dirs()
    clinical_path = Path(args.clinical)
    if not clinical_path.exists():
        raise FileNotFoundError(
            f"临床数据不存在：{clinical_path}\n"
            f"请将原始临床表放入 data/raw/ 后重试，或通过 --clinical 指定路径。"
        )

    print("=" * 60)
    print("审计临床数据")
    print("=" * 60)
    clin = audit_clinical(clinical_path)
    print(json.dumps(clin, ensure_ascii=False, indent=2))

    print("=" * 60)
    print("审计视频数据")
    print("=" * 60)
    matches_path = resolve_matches_path(args.video_matches)
    video = audit_video(matches_path)

    with open(AUDIT_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump({"clinical": clin, "video": video}, f, ensure_ascii=False, indent=2)
    print(f"\n审计报告已写入 {AUDIT_DIR}")


if __name__ == "__main__":
    main()
