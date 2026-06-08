"""阈值天花板诊断：POT 选的 F1 vs 扫遍所有阈值的 oracle best-F1。

承接 recon_calibrate 的结论——逐变量校准早在 sweep 里(zscore 模式)，不是瓶颈。
剩下的缺口是「最佳序列 AUC 高、但 POT 的 raw_f1 平庸」。本脚本对每台**同一条最佳序列**：

  - pot_f1   : 当前 pipeline 用 POT(2 q × 4 level) 选阈得到的 raw_f1（= best_1d 的结果）
  - oracle_f1: 把阈值扫遍该序列所有取值，能拿到的最高 raw_f1（= 论文"best-F1"口径上界）
  - AUC      : 该序列的排序可分性（阈值无关）

判读：
  oracle_f1 ≫ pot_f1  → **POT 把 F1 漏了**，定阈层有救（找更好的 label-free 阈值规则）。
  oracle_f1 ≈ pot_f1  → 不是定阈问题，是逐点 F1 在类不平衡下的**固有天花板**
                        → 阈值层没救，该换评估指标（AUC-PR / VUS / affiliation / 段级）。
  oracle_f1 仍不高（如 <0.6）且 AUC 高 → 同上：高 AUC 不等于高逐点 F1，是指标的问题。

零重训，只读 runs_fuse/<ent>/{val,test}_scores.npy。
用法：
  python scripts/threshold_ceiling.py --config configs/smd.yaml --recon_dir runs_fuse
  python scripts/threshold_ceiling.py --config configs/smd.yaml --recon_dir runs_fuse --entities machine-1-1
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
from static_fuse import best_1d  # noqa: E402
from diagnose_regime import auc, ALL28  # noqa: E402


def oracle_best_f1(score: np.ndarray, label: np.ndarray) -> float:
    """扫遍所有阈值的最高逐点 F1（= 把分数从高到低逐个纳入为异常，取最优前缀）。"""
    n = min(len(score), len(label))
    score, label = score[:n], label[:n].astype(np.int64)
    P = int(label.sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(f1.max())


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
            print(f"[skip] {ent}: 缺分数（{rd}）", flush=True)
            continue
        cfg.data.entity = ent
        print(f"[{i}/{n_ent}] {ent} ...", flush=True)
        vr2 = np.load(rd / "val_scores.npy")
        tr2 = np.load(rd / "test_scores.npy")
        _, _, _, test_label, _ = build_smd_datasets(cfg)
        label = np.asarray(test_label, dtype=np.int64)

        base = best_1d(vr2, tr2, label)
        if base is None:
            print(f"[skip] {ent}: sweep 失败"); continue
        ts = base["ts"]
        pot_f1 = base["raw_f1"]
        orac = oracle_best_f1(ts, label)
        a = auc(ts[:min(len(ts), len(label))], label[:min(len(ts), len(label))])
        rows.append({"ent": ent, "pot": pot_f1, "oracle": orac, "auc": a})
        print(f"    pot_f1={pot_f1:.3f}  oracle_f1={orac:.3f}  gap={orac-pot_f1:+.3f}  AUC={a:.3f}",
              flush=True)

    if not rows:
        print("无可评估 entity。"); return

    print(f"\n{'entity':<14} {'pot_f1':>7} {'oracle_f1':>10} {'gap':>7} {'AUC':>7}")
    print("-" * 52)
    for r in rows:
        print(f"{r['ent']:<14} {r['pot']:>7.3f} {r['oracle']:>10.3f} "
              f"{r['oracle']-r['pot']:>+7.3f} {r['auc']:>7.3f}")
    pot = np.array([r["pot"] for r in rows]); orc = np.array([r["oracle"] for r in rows])
    print("-" * 52)
    print(f"{'MEAN':<14} {pot.mean():>7.3f} {orc.mean():>10.3f} {(orc-pot).mean():>+7.3f}")
    n = len(rows)
    print(f"\n# oracle 明显高于 pot (gap>+0.05): {int(((orc-pot) > 0.05).sum())}/{n}")
    print("判读：gap 普遍大 → POT 定阈漏了 F1，阈值层有救（找 label-free 的更优阈值规则）；")
    print("      gap 普遍小 → 是逐点 F1 的固有天花板，阈值层没救，该换指标（AUC-PR/VUS/affiliation）。")


if __name__ == "__main__":
    main()
