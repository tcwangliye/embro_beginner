"""06_report.py —— 汇总所有实验指标，生成对比表并维护实验注册表。

  - 读取 results/metrics/*_metrics.json（02 / 05 产出的所有实验）
  - 打印对比表（控制台 + results/metrics/summary.csv）
  - 追加 / 更新 results/experiments.csv（同名实验覆盖，保证可复现可追溯）

用法示例：
  python scripts/06_report.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import METRIC_DIR, RESULTS_DIR, ensure_dirs, load_json

EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"


def collect_metrics(metric_dir: Path) -> pd.DataFrame:
    """扫描所有 *_metrics.json，展平成一行一条记录。"""
    records: list[dict] = []
    for f in sorted(metric_dir.glob("*_metrics.json")):
        d = load_json(f)
        records.append({
            "tag": d.get("tag", f.stem),
            "script": d.get("script", ""),
            "mode": d.get("mode", ""),
            "task": d.get("task", ""),
            "n_samples": d.get("n_samples"),
            "n_patients": d.get("n_patients"),
            "n_positive": d.get("n_positive"),
            "n_matched_video": (d.get("audit") or {}).get("n_matched"),
            "auroc": d.get("auroc"),
            "auprc": d.get("auprc"),
            "folds": d.get("folds"),
            "seed": d.get("seed"),
            "metric_file": f.name,
        })
    return pd.DataFrame(records)


def update_experiments(rows: pd.DataFrame) -> None:
    """更新实验注册表：按 tag 覆盖旧记录，其余追加。"""
    if EXPERIMENTS_CSV.exists():
        old = pd.read_csv(EXPERIMENTS_CSV)
    else:
        old = pd.DataFrame(columns=rows.columns)
    merged = pd.concat([old, rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["tag"], keep="last")
    merged = merged.sort_values("tag").reset_index(drop=True)
    merged.to_csv(EXPERIMENTS_CSV, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="实验指标汇总")
    parser.add_argument("--metric-dir", default=str(METRIC_DIR))
    args = parser.parse_args()

    ensure_dirs()
    rows = collect_metrics(Path(args.metric_dir))
    if rows.empty:
        print(f"未找到任何 *_metrics.json（{args.metric_dir}）\n"
              f"请先运行 02_baseline_clinical.py 或 05_train_fusion.py。")
        return

    # 打印对比表（AUROC 降序）
    view = rows[["tag", "mode", "task", "n_samples", "n_positive", "auroc", "auprc"]]
    print("=" * 88)
    print(view.to_string(index=False))
    print("=" * 88)

    summary_path = METRIC_DIR / "summary.csv"
    rows.to_csv(summary_path, index=False)
    update_experiments(rows)
    print(f"\n汇总已写入：{summary_path}")
    print(f"实验注册表已更新：{EXPERIMENTS_CSV}")


if __name__ == "__main__":
    main()
