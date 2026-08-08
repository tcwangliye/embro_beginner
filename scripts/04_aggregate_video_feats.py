"""04_aggregate_video_feats.py —— 帧级特征 -> 视频级特征聚合。

将不定长帧序列压缩为固定维度的胚胎级输入向量，供 05 融合建模使用。

聚合策略（对每个胚胎的所有帧）：
  mean   所有帧特征的均值          —— 视频整体"平均外观"
  std    所有帧特征的标准差        —— 时序变化幅度
  delta  后一半帧均值 - 前一半帧均值 —— 发育趋势
  last   最后一帧特征              —— 最终状态快照

输出：每个胚胎一行，列名 femi_mean_0000, femi_std_0000, ...（维度 x 4）

用法示例：
  python scripts/04_aggregate_video_feats.py
  python scripts/04_aggregate_video_feats.py --feats data/processed/frame_feats_ihlab_FEMI.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    FRAME_FEATS_CSV, KEY, PATIENT, PROCESSED_DIR, VIDEO_FEATS_CSV, ensure_dirs,
)

# 视频聚合特征的分组名（与 mean/std/delta/last 对应）
AGG_NAMES = ["mean", "std", "delta", "last"]


def load_frame_feats(path: Path | str) -> pd.DataFrame:
    """读取帧级特征 CSV，自动识别特征列（femi_xxxx 前缀）。"""
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("femi_")]
    if not feat_cols:
        raise ValueError(
            f"帧特征文件中未找到 femi_ 前缀的特征列。实际列：{list(df.columns)}"
        )
    return df, feat_cols


def aggregate_embryo(values: np.ndarray) -> np.ndarray:
    """将单个胚胎的帧特征矩阵 (n_frames, d) 聚合成 (4*d,)。

    mean / std / delta / last 四种聚合首尾相接。
    """
    values = np.asarray(values, dtype=np.float32)
    n, d = values.shape
    cut = max(1, n // 2)  # 保证前半至少 1 帧
    first_half = values[:cut].mean(axis=0)
    second_half = values[cut:].mean(axis=0)
    return np.concatenate([
        values.mean(axis=0),          # mean
        values.std(axis=0),           # std
        second_half - first_half,     # delta
        values[-1],                   # last
    ]).astype(np.float32)


def aggregate_video_feats(frame_feats: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """按胚胎分组聚合，返回视频级特征 DataFrame。"""
    records: list[dict] = []
    d = len(feat_cols)
    for key, grp in frame_feats.sort_values([KEY, "frame_index"]).groupby(KEY, sort=False):
        vals = grp[feat_cols].to_numpy(dtype=np.float32)
        agg = aggregate_embryo(vals)
        row: dict = {KEY: key, "n_frames": int(len(grp))}
        if PATIENT in grp.columns:
            row[PATIENT] = grp[PATIENT].iloc[0]
        for k, agg_name in enumerate(AGG_NAMES):
            for j in range(d):
                row[f"femi_{agg_name}_{j:04d}"] = float(agg[k * d + j])
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="帧级 -> 视频级特征聚合")
    parser.add_argument("--feats", default=str(FRAME_FEATS_CSV), help="帧级特征 CSV 路径")
    parser.add_argument("--out", default=str(VIDEO_FEATS_CSV), help="视频级特征输出路径")
    args = parser.parse_args()

    ensure_dirs()
    feats_path = Path(args.feats)
    if not feats_path.exists():
        # 默认路径缺失时，自动探测 03 输出过的历史帧特征（取最新）
        candidates = sorted(
            PROCESSED_DIR.glob("frame_feats_*.csv"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            feats_path = candidates[0]
            print(f"默认路径 {Path(args.feats)} 不存在，自动使用最新帧特征：{feats_path}")
    if not feats_path.exists():
        raise FileNotFoundError(
            f"帧级特征不存在：{feats_path}\n请先运行 03_extract_frame_feats.py，"
            f"或用 --feats 指定路径。"
        )

    frame_feats, feat_cols = load_frame_feats(feats_path)
    print(f"帧级特征：{len(frame_feats)} 帧 / {frame_feats[KEY].nunique()} 胚胎，特征维度 {len(feat_cols)}")

    video_feats = aggregate_video_feats(frame_feats, feat_cols)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_feats.to_csv(out_path, index=False)

    frame_counts = video_feats["n_frames"]
    print(f"视频级特征已保存：{out_path}（{len(video_feats)} 胚胎 x {video_feats.shape[1]} 列）")
    print(f"每胚胎帧数：min={frame_counts.min()}  median={frame_counts.median():.0f}  "
          f"max={frame_counts.max()}  无帧胚胎={int((frame_counts == 0).sum())}")


if __name__ == "__main__":
    main()
