"""02_baseline_clinical.py —— 纯临床表格基线（CatBoost + 五折患者分组 CV）。

先立参考下限：只用移植前临床特征预测 LB（或 FH）。
后续所有融合方案都必须与这个基线公平对比。

产出（results/）：
  oof/baseline_clinical_{task}_oof.csv           OOF 预测（KEY/PATIENT/label/fold/prob）
  models/baseline_clinical_{task}_fold{i}.cbm    每折 CatBoost 模型
  metrics/baseline_clinical_{task}_metrics.json  整体指标 + 审计 + 配置
  metrics/baseline_clinical_{task}_fold_metrics.csv 每折指标
  metrics/baseline_clinical_{task}_feature_columns.txt

用法示例：
  python scripts/02_baseline_clinical.py --task lb
  python scripts/02_baseline_clinical.py --task fh --iterations 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from common import (
    CLINICAL_CSV, CLINICAL_FEATURES, KEY, LABELS, METRIC_DIR, MODEL_DIR,
    OOF_DIR, PATIENT, assert_no_leakage, ensure_dirs, load_cohort, metric_dict,
    save_json, write_oof_csv,
)


def run_baseline(
    task: str,
    clinical_path: Path,
    features: list[str],
    seed: int,
    folds: int,
    iterations: int,
) -> dict:
    """训练临床基线并返回指标摘要。"""
    label_col = LABELS[task]
    df, y, groups, audit = load_cohort(clinical_path, task, features)

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(df), np.nan)
    fold_ids = np.full(len(df), -1)
    fold_metrics: list[dict] = []

    for fold, (tr, va) in enumerate(skf.split(df, y, groups)):
        assert_no_leakage(groups, tr, va)

        model = CatBoostClassifier(
            iterations=iterations,
            depth=4,
            learning_rate=0.03,
            loss_function="Logloss",
            random_seed=seed + fold,
            allow_writing_files=False,   # 禁止生成临时分析文件
            verbose=0,
        )
        model.fit(df.loc[tr, features], y[tr])
        oof[va] = model.predict_proba(df.loc[va, features])[:, 1]
        fold_ids[va] = fold

        model.save_model(MODEL_DIR / f"baseline_clinical_{task}_fold{fold}.cbm")
        m = metric_dict(y[va], oof[va])
        m.update({"fold": fold, "n_valid": int(len(va))})
        fold_metrics.append(m)
        print(f"  fold {fold}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  (n_valid={len(va)})")

    # 质量门禁：OOF 必须完整、合法
    assert np.isfinite(oof).all(), "OOF 预测存在 NaN / Inf！"
    assert (oof >= 0).all() and (oof <= 1).all(), "OOF 概率超出 [0, 1]！"
    per_patient_folds = pd.DataFrame({PATIENT: groups, "fold": fold_ids}).groupby(PATIENT)["fold"].nunique()
    assert per_patient_folds.max() == 1, "同一患者被分到多个折，存在泄漏！"

    tag = f"baseline_clinical_{task}"
    write_oof_csv(df, y, oof, fold_ids, OOF_DIR / f"{tag}_oof.csv", label_col)
    pd.DataFrame(fold_metrics).to_csv(METRIC_DIR / f"{tag}_fold_metrics.csv", index=False)
    Path(METRIC_DIR / f"{tag}_feature_columns.txt").write_text("\n".join(features), encoding="utf-8")

    overall = metric_dict(y, oof)
    summary = {
        "script": "02_baseline_clinical",
        "tag": tag,
        "task": task,
        "n_samples": int(len(df)),
        "n_patients": int(df[PATIENT].nunique()),
        "n_positive": int(y.sum()),
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
    parser = argparse.ArgumentParser(description="纯临床表格基线")
    parser.add_argument("--task", choices=["fh", "lb"], default="lb", help="预测目标")
    parser.add_argument("--clinical", default=str(CLINICAL_CSV), help="临床表路径")
    parser.add_argument("--features", nargs="*", default=None, help="自定义特征列（缺省用 CLINICAL_FEATURES）")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    ensure_dirs()
    features = args.features or CLINICAL_FEATURES
    summary = run_baseline(
        task=args.task,
        clinical_path=Path(args.clinical),
        features=features,
        seed=args.seed,
        folds=args.folds,
        iterations=args.iterations,
    )
    print("\n" + "=" * 60)
    print(f"[{args.task}] 临床基线：AUROC={summary['auroc']:.4f}  AUPRC={summary['auprc']:.4f}  "
          f"(n={summary['n_samples']}, 患者 {summary['n_patients']}, 阳性 {summary['n_positive']})")


if __name__ == "__main__":
    main()
