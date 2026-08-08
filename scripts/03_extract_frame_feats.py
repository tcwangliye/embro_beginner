"""03_extract_frame_feats.py —— 用预训练 ViT 提取每帧的视觉嵌入。

流程：
  1. 若无帧清单则自动生成（扫描胚胎目录 + 自然排序 + stride 采样）
  2. 分批推理：图像 -> processor -> model.vit() -> 去 [CLS] 后对 patch 均值池化
  3. 输出帧级特征 CSV（data/processed/frame_feats_<tag>.csv）

视频特征提取管线（03 -> 04）的第一环，必须在 GPU 上运行。

用法示例：
  python scripts/03_extract_frame_feats.py                          # 默认 ihlab/FEMI，全量
  python scripts/03_extract_frame_feats.py --limit 20 --stride 10   # 小样本试跑
  python scripts/03_extract_frame_feats.py --model google/vit-base-patch16-224-in21k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import (
    FRAME_FEATS_CSV, FRAME_MANIFEST_CSV, IMAGE_EXTS, KEY, PATIENT,
    UNIQUE_VIDEO_MATCHES_CSV, VIDEO_MATCHES_CSV, ensure_dirs, natural_key,
)


# ------------------------------------------------------------
# 帧清单生成（幂等：已存在则复用，除非 --regenerate）
# ------------------------------------------------------------
def resolve_matches_path() -> Path | None:
    for cand in (VIDEO_MATCHES_CSV, UNIQUE_VIDEO_MATCHES_CSV):
        if cand.exists():
            return cand
    return None


def generate_frame_manifest(matches_path: Path, limit: int, stride: int) -> pd.DataFrame:
    """扫描每个胚胎目录，按自然排序 + stride 采样生成帧清单。"""
    m = pd.read_csv(matches_path)
    dir_col = next((c for c in ("embryo_dir", "processed_f0", "video_dir") if c in m.columns), None)
    if dir_col is None:
        raise ValueError(f"匹配表缺少胚胎目录列，实际列：{list(m.columns)}")

    keys = m[KEY].unique()[:limit] if limit > 0 else m[KEY].unique()
    rows: list[dict] = []
    for row in m.itertuples(index=False):
        if getattr(row, KEY) not in keys:
            continue
        d = Path(getattr(row, dir_col))
        if not d.exists():
            print(f"  [跳过] 目录不存在：{d}")
            continue
        frames = sorted(
            [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: natural_key(p.name),
        )
        sampled = frames[::stride]
        for idx, fp in enumerate(sampled):
            rows.append({
                KEY: getattr(row, KEY),
                PATIENT: getattr(row, "clean_patient_id", np.nan),
                "frame_index": idx,
                "frame_path": str(fp),
            })
    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values([KEY, "frame_index"]).reset_index(drop=True)
    manifest.to_csv(FRAME_MANIFEST_CSV, index=False)
    print(f"帧清单已生成：{len(manifest)} 帧 / {manifest[KEY].nunique()} 胚胎 -> {FRAME_MANIFEST_CSV}")
    return manifest


def load_manifest(limit: int, stride: int, regenerate: bool) -> pd.DataFrame:
    """加载或生成帧清单。"""
    if FRAME_MANIFEST_CSV.exists() and not regenerate:
        manifest = pd.read_csv(FRAME_MANIFEST_CSV)
        print(f"复用帧清单：{len(manifest)} 帧 / {manifest[KEY].nunique()} 胚胎")
        return manifest
    matches = resolve_matches_path()
    if matches is None:
        raise FileNotFoundError(
            f"未找到视频匹配表（{VIDEO_MATCHES_CSV} 或 {UNIQUE_VIDEO_MATCHES_CSV}），无法生成帧清单。"
        )
    return generate_frame_manifest(matches, limit, stride)


# ------------------------------------------------------------
# 帧特征提取
# ------------------------------------------------------------
def load_model(model_id: str, device: torch.device):
    """加载 FEMI 预训练模型（AutoModelForPreTraining + AutoImageProcessor）。"""
    from transformers import AutoImageProcessor, AutoModelForPreTraining

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForPreTraining.from_pretrained(model_id)
    # FEMI 基于 MAE 预训练：推理时关闭随机掩码
    if hasattr(model.config, "mask_ratio"):
        print(f"关闭 mask_ratio（{model.config.mask_ratio} -> 0.0）")
        model.config.mask_ratio = 0.0
    model.to(device).eval()
    return processor, model


def extract_embedding(model, pixel_values: torch.Tensor, device: torch.device) -> np.ndarray:
    """取 ViT 编码器输出：去 [CLS] token 后对 patch 做均值池化。

    优先 model.vit()（跳过预训练头）；若无 vit 属性则回退到
    model(..., output_hidden_states=True) 取最后一层。
    """
    with torch.inference_mode():
        if hasattr(model, "vit"):
            hidden = model.vit(pixel_values=pixel_values).last_hidden_state  # (B, N+1, D)
        else:
            hidden = model(pixel_values=pixel_values, output_hidden_states=True).hidden_states[-1]
        emb = hidden[:, 1:, :].mean(dim=1)  # 去 [CLS]，patch 均值池化 -> (B, D)
    return emb.float().cpu().numpy()


def extract_frame_feats(manifest: pd.DataFrame, model_id: str, batch_size: int, device: torch.device) -> pd.DataFrame:
    """分批推理，返回帧级特征 DataFrame（femi_0000, femi_0001, ...）。"""
    processor, model = load_model(model_id, device)
    n = len(manifest)
    out: list[dict] = []
    embed_dim: int | None = None

    for start in range(0, n, batch_size):
        batch = manifest.iloc[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch["frame_path"]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        embs = extract_embedding(model, inputs["pixel_values"], device)
        embed_dim = embs.shape[1]
        for (_, row), emb in zip(batch.iterrows(), embs):
            out.append({
                KEY: row[KEY],
                PATIENT: row[PATIENT],
                "frame_index": int(row["frame_index"]),
                "frame_path": row["frame_path"],
                **{f"femi_{i:04d}": float(v) for i, v in enumerate(emb)},
            })
        print(f"  已处理 {min(start + batch_size, n)}/{n} 帧", end="\r")
    print()

    feats = pd.DataFrame(out)
    assert embed_dim is not None
    feats = feats[  # 列序：元信息在前，特征在后
        [KEY, PATIENT, "frame_index", "frame_path"]
        + [f"femi_{i:04d}" for i in range(embed_dim)]
    ]
    return feats


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="FEMI 帧级特征提取")
    parser.add_argument("--model", default="ihlab/FEMI", help="HuggingFace 预训练模型 ID")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--limit", type=int, default=0, help="处理的胚胎数（0 = 全量）")
    parser.add_argument("--stride", type=int, default=20, help="帧采样间隔")
    parser.add_argument("--regenerate", action="store_true", help="强制重新生成帧清单")
    parser.add_argument("--out", default=str(FRAME_FEATS_CSV), help="帧级特征输出 CSV 路径")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"设备：{device}（GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else '无'}）")

    manifest = load_manifest(args.limit, args.stride, args.regenerate)
    if manifest.empty:
        raise SystemExit("帧清单为空，请检查匹配表与胚胎目录。")

    # 帧路径存在性校验（一次性发现磁盘问题）
    missing = [p for p in manifest["frame_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"清单中有 {len(missing)} 个帧文件在磁盘上不存在，例如：{missing[0]}")

    feats = extract_frame_feats(manifest, args.model, args.batch_size, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_path, index=False)
    print(f"帧级特征已保存：{out_path}（{len(feats)} 行 x {feats.shape[1]} 列）")


if __name__ == "__main__":
    main()
