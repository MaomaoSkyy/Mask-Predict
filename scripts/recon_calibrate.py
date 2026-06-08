"""逐变量 robust 校准 → 重定阈，验证“被埋掉的 AUC 优势能否捞成真 raw_f1”。

诊断(diagnose_regime)发现：recon 分在阈值无关的 AUC 上其实赢幅度基线，但 raw_f1 平庸。
头号嫌疑是 `max over 38 变量且无逐变量校准`——异常只在 1~2 变量，38 个异质误差通道直接
取 max，噪声通道把基线托起、各自尺度不一 → 单一 POT 阈对不准。

本脚本零重训，只在已存的 (T,D) 重构误差上做一件事：**聚合前先逐变量用 val 段的
中位/IQR 校准**，让每个通道变成“偏离自己正常多少个 IQR”，再走同一套 sweep。
对比 base(原始误差) vs calib(逐变量校准) 的最佳 raw_f1。

  calib > base 普遍   → 阈值层确实是瓶颈，AUC 优势捞回来了（廉价真提升）。
  calib ≈ base        → 不是聚合/尺度问题，瓶颈在别处（如 POT 本身/段级判定）。

用法：
  python scripts/recon_calibrate.py --config configs/smd.yaml --recon_dir runs_fuse
  python scripts/recon_calibrate.py --config configs/smd.yaml --recon_dir runs_fuse \
         --entities machine-1-7 machine-3-3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from static_fuse import best_1d  # noqa: E402  (复用：postprocess×agg×POT 扫最佳 raw_f1)
from diagnose_regime import auc, ALL28  # noqa: E402


def per_var_calibrate(v2d: np.ndarray, t2d: np.ndarray, eps: float = 1e-6):
    """逐变量 robust 标准化：用 **val** 段每个变量自己的中位/IQR 校准 val 和 test。
    cal[t,d] = (err[t,d] - median_d) / IQR_d  → “变量 d 偏离自己正常误差多少个 IQR”。
    """
    med = np.median(v2d, axis=0)                                  # (D,)
    q25, q75 = np.percentile(v2d, [25, 75], axis=0)
    iqr = np.maximum(q75 - q25, eps)
    return (v2d - med) / iqr, (t2d - med) / iqr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recon_dir", default="runs_fuse")
    ap.add_argument("--entities", nargs="+", default=ALL28)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = []
    n_ent = len(args.entities)
    for i, ent in enumerate(args.entities, 1):
        rd = Path(args.recon_dir) / ent
        if not (rd / "val_scores.npy").exists() or not (rd / "test_scores.npy").exists():
            print(f"[skip] {ent}: 缺 val/test_scores.npy（{rd}）", flush=True)
            continue
        cfg.data.entity = ent
        print(f"[{i}/{n_ent}] {ent} sweeping ...", flush=True)
        vr2 = np.load(rd / "val_scores.npy")
        tr2 = np.load(rd / "test_scores.npy")
        _, _, _, test_label, _ = build_smd_datasets(cfg)
        label = np.asarray(test_label, dtype=np.int64)

        base = best_1d(vr2, tr2, label)
        cv, ct = per_var_calibrate(vr2, tr2)
        calib = best_1d(cv, ct, label)
        if base is None or calib is None:
            print(f"[skip] {ent}: sweep 失败"); continue
        print(f"    base_f1={base['raw_f1']:.3f}  calib_f1={calib['raw_f1']:.3f}  "
              f"Δ={calib['raw_f1']-base['raw_f1']:+.3f}", flush=True)

        # AUC 对照（用各自最优 1D 序列，截到 label 长度）
        T = min(len(base["ts"]), len(label))
        base_auc = auc(base["ts"][:T], label[:T])
        T2 = min(len(calib["ts"]), len(label))
        calib_auc = auc(calib["ts"][:T2], label[:T2])

        rows.append({"ent": ent, "base": base["raw_f1"], "calib": calib["raw_f1"],
                     "base_auc": base_auc, "calib_auc": calib_auc})

    if not rows:
        print("无可评估 entity（先跑 run_static_fuse.sh 生成分数）。"); return

    print(f"\n{'entity':<14} {'base_f1':>8} {'calib_f1':>9} {'Δf1':>7} "
          f"{'base_AUC':>9} {'calib_AUC':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['ent']:<14} {r['base']:>8.3f} {r['calib']:>9.3f} "
              f"{r['calib']-r['base']:>+7.3f} {r['base_auc']:>9.3f} {r['calib_auc']:>10.3f}")
    base = np.array([r["base"] for r in rows]); cal = np.array([r["calib"] for r in rows])
    print("-" * 70)
    print(f"{'MEAN':<14} {base.mean():>8.3f} {cal.mean():>9.3f} {(cal-base).mean():>+7.3f}")
    n = len(rows)
    print(f"\n# calib > base (+0.005以上): {int((cal > base + 0.005).sum())}/{n}")
    print(f"# calib < base (-0.005以下): {int((cal < base - 0.005).sum())}/{n}")
    print(f"\n判读：calib 普遍 > base → 阈值/聚合层是瓶颈，逐变量校准把 AUC 优势捞成了真 F1；"
          f"\n      calib ≈ base → 瓶颈不在聚合尺度，得查 POT 本身或点级判定。")


if __name__ == "__main__":
    main()
