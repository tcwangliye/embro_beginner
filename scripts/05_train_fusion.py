"""05_train_fusion.py —— 三路建模对比：clinical / video / fusion。

用同一套五折患者分组 CV 和 CatBoost 超参，公平回答
"视频特征能否提升纯临床基线的预测能力"：

  --mode clinical  纯临床特征（与 02 基线等价，便于交叉验证）
  --mode video     纯 FEMI 视频特征（PCA 降到 32 维）
  --mode fusion    临床 + 视频特征拼接

视频特征管线（video / fusion 模式）：
  原始 FEMI 特征 -> SimpleImputer(median) -> StandardScaler -> PCA
  —— 全部在每折训练集上 fit，验证集上 transform，避免数据泄漏。

产出（results/）：
  oof/{mode}_{task}_oof.csv                     OOF 预测
  models/{mode}_{task}_fold{i}.cbm              每折模型
  models/{mode}_{task}_video_pca_fold{i}.joblib PCA pipeline（video/fusion）
  metrics/{mode}_{task}_metrics.json            指标 + 匹配样本数 + 配置

用法示例：
  python scripts/05_train_fusion.py --mode clinical --task lb
  python scripts/05_train_fusion.py --mode video   --task lb
  python scripts/05_train_fusion.py --mode fusion  --task lb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CLINICAL_CSV, CLINICAL_FEATURES, KEY, LABELS, METRIC_DIR, MODEL_DIR,
    OOF_DIR, PATIENT, VIDEO_FEATS_CSV, assert_no_leakage, ensure_dirs,
    load_cohort, metric_dict, save_json, write_oof_csv,
)

MODES = ("clinical", "video", "fusion")


def load_video_feats(path: Path | str) -> pd.DataFrame:
    """读取视频级特征，返回特征列名列表。"""
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("femi_")]
    if not feat_cols:
        raise ValueError(f"视频特征中未找到 femi_ 前缀列：{list(df.columns)}")
    return df, feat_cols


def build_dataset(
    mode: str,
    task: str,
    clinical_path: Path,
    video_feats: pd.DataFrame | None,
    video_feat_cols: list[str],
    clinical_features: list[str],
):
    """按模式构建 (df, y, groups, meta)：与视频特征按 KEY 内连接。"""
    df, y, groups, audit = load_cohort(clinical_path, task, clinical_features)

    if mode == "clinical":
        return df, y, groups, audit, clinical_features

    assert video_feats is not None, "video/fusion 模式需要视频特征表"
    merged = df.merge(video_feats, on=KEY, how="inner", suffixes=("", "_video"))
    audit["n_clinical"] = int(len(df))
    audit["n_matched"] = int(len(merged))
    audit["n_missing_video"] = int(len(df) - len(merged))

    y = merged[LABELS[task]].to_numpy(dtype=float).astype(int)
    groups = merged[PATIENT].to_numpy()
    if mode == "video":
        return merged, y, groups, audit, video_feat_cols
    # fusion：临床 + 视频特征列名拼接
    return merged, y, groups, audit, clinical_features + video_feat_cols


def make_video_pipeline(n_components: int, n_train: int, n_raw: int):
    """构建视频特征降维管线。n_components 取三者最小值，防止样本不足报错。"""
    k = min(n_components, n_train - 1, n_raw)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=k, random_state=0)),
    ])


def run_fusion(
    mode: str,
    task: str,
    clinical_path: Path,
    video_feats_path: Path | None,
    pca_dim: int,
    seed: int,
    folds: int,
    iterations: int,
) -> dict:
    """运行指定模式的五折建模，返回指标摘要。"""
    clinical_features = list(CLINICAL_FEATURES)
    video_feats, video_feat_cols = (load_video_feats(video_feats_path)
                                    if mode != "clinical" else (None, []))

    df, y, groups, audit, feat_cols = build_dataset(
        mode, task, clinical_path, video_feats, video_feat_cols, clinical_features
    )

    video_cols = [c for c in feat_cols if c in video_feat_cols] if mode != "clinical" else []
    clin_cols = [c for c in feat_cols if c not in video_cols]

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(df), np.nan)
    fold_ids = np.full(len(df), -1)
    fold_metrics: list[dict] = []

    for fold, (tr, va) in enumerate(skf.split(df, y, groups)):
        assert_no_leakage(groups, tr, va)

        X_train, X_valid = df.loc[tr, feat_cols], df.loc[va, feat_cols]

        # video/fusion：视频列在折内 fit 降维管线（防泄漏）
        if mode != "clinical" and video_cols:
            pipe = make_video_pipeline(pca_dim, len(tr), len(video_cols))
            pipe.fit(X_train[video_cols])
            joblib.dump(pipe, MODEL_DIR / f"{mode}_{task}_video_pca_fold{fold}.joblib")
            pca_cols = [f"video_pca_{i:02d}" for i in range(pipe.named_steps["pca"].n_components_)]
            X_train = pd.concat([X_train[clin_cols].reset_index(drop=True),
                                 pd.DataFrame(pipe.transform(X_train[video_cols]), columns=pca_cols)], axis=1)
            X_valid = pd.concat([X_valid[clin_cols].reset_index(drop=True),
                                 pd.DataFrame(pipe.transform(X_valid[video_cols]), columns=pca_cols)], axis=1)

        model = CatBoostClassifier(
            iterations=iterations,
            depth=4,
            learning_rate=0.03,
            loss_function="Logloss",
            random_seed=seed + fold,
            allow_writing_files=False,
            verbose=0,
        )
        model.fit(X_train, y[tr])
        oof[va] = model.predict_proba(X_valid)[:, 1]
        fold_ids[va] = fold

        model.save_model(MODEL_DIR / f"{mode}_{task}_fold{fold}.cbm")
        m = metric_dict(y[va], oof[va])
        m.update({"fold": fold, "n_valid": int(len(va))})
        fold_metrics.append(m)
        print(f"  fold {fold}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  (n_valid={len(va)})")

    # 质量门禁
    assert np.isfinite(oof).all(), "OOF 预测存在 NaN / Inf！"
    assert (oof >= 0).all() and (oof <= 1).all(), "OOF 概率超出 [0, 1]！"
    per_patient_folds = pd.DataFrame({PATIENT: groups, "fold": fold_ids}).groupby(PATIENT)["fold"].nunique()
    assert per_patient_folds.max() == 1, "同一患者被分到多个折，存在泄漏！"

    tag = f"{mode}_{task}"
    write_oof_csv(df, y, oof, fold_ids, OOF_DIR / f"{tag}_oof.csv", LABELS[task])
    pd.DataFrame(fold_metrics).to_csv(METRIC_DIR / f"{tag}_fold_metrics.csv", index=False)

    overall = metric_dict(y, oof)
    summary = {
        "script": "05_train_fusion",
        "tag": tag,
        "mode": mode,
        "task": task,
        "n_samples": int(len(df)),
        "n_patients": int(df[PATIENT].nunique()),
        "n_positive": int(y.sum()),
        "n_video_features": len(video_feat_cols),
        "pca_dim": pca_dim,
        "folds": folds,
        "seed": seed,
        "iterations": iterations,
        "auroc": overall["auroc"],
        "auprc": overall["auprc"],
        "fold_metrics": fold_metrics,
        "audit": audit,
    }
    save_json(summary, METRIC_DIR / f"{tag}_metrics.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="临床/视频/融合三路对比")
    parser.add_argument("--mode", choices=MODES, default="fusion")
    parser.add_argument("--task", choices=["fh", "lb"], default="lb")
    parser.add_argument("--clinical", default=str(CLINICAL_CSV))
    parser.add_argument("--video-feats", default=str(VIDEO_FEATS_CSV))
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    ensure_dirs()
    video_path = Path(args.video_feats) if args.mode != "clinical" else None
    summary = run_fusion(
        mode=args.mode,
        task=args.task,
        clinical_path=Path(args.clinical),
        video_feats_path=video_path,
        pca_dim=args.pca_dim,
        seed=args.seed,
        folds=args.folds,
        iterations=args.iterations,
    )
    print("\n" + "=" * 60)
    print(f"[{args.mode}/{args.task}] AUROC={summary['auroc']:.4f}  AUPRC={summary['auprc']:.4f}  "
          f"(n={summary['n_samples']}, 匹配视频 {summary['audit'].get('n_matched', 'N/A')})")


if __name__ == "__main__":
    main()
