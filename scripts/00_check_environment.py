"""00_check_environment.py —— 环境就绪检查（Phase 0 入口，必跑）。

逐项检查关键依赖包与 GPU 状态，一次报告全部缺失项（不中断），
缺包时给出安装建议；GPU 不可用时明确警告视频特征提取会不可行。

用法示例：
  python scripts/00_check_environment.py
  python scripts/00_check_environment.py --min-cuda  # GPU 缺失时返回非零退出码
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

# (显示名, import 模块名, 是否 GPU 相关)
PACKAGES = [
    ("pandas", "pandas", False),
    ("numpy", "numpy", False),
    ("scikit-learn", "sklearn", False),
    ("catboost", "catboost", False),
    ("joblib", "joblib", False),
    ("Pillow", "PIL", False),
    ("torch (PyTorch)", "torch", True),
    ("transformers (HF)", "transformers", True),
]

# 表格建模必需 / 视频建模必需
TABLE_ONLY = {"pandas", "numpy", "scikit-learn", "catboost", "joblib"}
VIDEO_REQUIRED = {"Pillow", "torch (PyTorch)", "transformers (HF)"}


def check_packages() -> list[dict]:
    """逐个 import 并获取版本，缺包时记录失败。"""
    results: list[dict] = []
    for name, module, is_gpu in PACKAGES:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "未知")
            results.append({"name": name, "ok": True, "version": version, "gpu": is_gpu})
        except ImportError:
            results.append({"name": name, "ok": False, "version": None, "gpu": is_gpu})
    return results


def check_gpu() -> dict:
    """检查 torch CUDA：可用性 / GPU 名称 / 显存 / cuDNN。"""
    torch = importlib.import_module("torch")
    info: dict = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": None,
        "total_memory_gb": None,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 1)
    return info


def check_data_ready() -> list[str]:
    """报告 data/ 目录就绪状态（信息性，不判失败）。"""
    notes = []
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not raw.exists():
        notes.append(f"data/raw/ 不存在（{raw}），请先创建并放入原始数据")
    else:
        files = sorted(p.name for p in raw.iterdir() if p.is_file())
        notes.append(f"data/raw/ 内有 {len(files)} 个文件：{', '.join(files) if files else '（空）'}")
    return notes


def format_result(results: list[dict]) -> str:
    """格式化为对齐的文本表格。"""
    width = max(len(r["name"]) for r in results)
    lines = [f"{'包':<{width}}  状态    版本"]
    lines.append("-" * (width + 30))
    for r in results:
        status = "OK " if r["ok"] else "缺失"
        ver = r["version"] or "-"
        lines.append(f"{r['name']:<{width}}  {status}  {ver}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="环境就绪检查")
    parser.add_argument("--min-cuda", action="store_true",
                        help="GPU 不可用时返回退出码 1（用于 CI 门禁）")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Python 版本：{sys.version.split()[0]}  ({sys.executable})")
    print(f"磁盘剩余：{shutil.disk_usage(Path.home()).free / (1024**3):.1f} GB（$HOME）")
    print("=" * 60)

    # 1) 依赖包
    results = check_packages()
    print("\n[1] 依赖包检查")
    print(format_result(results))

    missing = [r for r in results if not r["ok"]]
    if missing:
        print("\n安装建议：")
        for r in missing:
            print(f"  - {r['name']}：pip install {r['name'].split()[0].lower()}")
    print("\n缺包影响：")
    for r in results:
        if not r["ok"]:
            group = "视频建模必需" if r["name"] in VIDEO_REQUIRED else "表格建模必需"
            print(f"  - 缺 {r['name']} → 影响：{group}")

    # 2) GPU
    print("\n[2] GPU 检查")
    gpu_ok = False
    if all(r["ok"] for r in results if r["gpu"]):
        gpu = check_gpu()
        if gpu["cuda_available"]:
            gpu_ok = True
            print(f"  CUDA 可用：{gpu['device_name']}（显存 {gpu['total_memory_gb']} GB）")
            print(f"  cuDNN 版本：{gpu['cudnn_version']}")
        else:
            print("  CUDA 不可用：torch 已安装但未检测到 GPU")
            print("  → 表格建模（02/05 clinical）可运行；视频特征提取（03）将非常慢或不可行")
    else:
        print("  跳过（torch / transformers 未安装，无法检查）")

    # 3) 数据就绪（信息性）
    print("\n[3] 数据就绪（信息）")
    for note in check_data_ready():
        print(f"  - {note}")

    # 汇总
    print("\n" + "=" * 60)
    n_fail = len(missing)
    print(f"结果：依赖 {len(results) - n_fail}/{len(results)} 通过，"
          f"GPU {'可用' if gpu_ok else '不可用/未检查'}")
    if n_fail or (args.min_cuda and not gpu_ok):
        print("环境未就绪：请按上面建议安装缺失包后重跑本脚本。")
        return 1
    print("环境就绪，可以进入 Phase 1 数据审计。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
