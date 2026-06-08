"""换指标评估：recon vs static 在**阈值无关**指标下的对比（AUC-PR + ROC-AUC）。

为什么换：threshold_ceiling 证明逐点 F1 是固有天花板（POT 已 = oracle），PA 又被 trivial
基线刷爆。模型的真实优势在**排序**（AUC），逐点 F1/PA 都不奖励它。AUC-PR（average
precision）阈值无关、对少数类（异常）敏感，是不平衡 AD 的标准指标——若 recon 的 AP 稳定
赢 static，这就是模型相对 trivial 幅度基线**第一个量化、可写的胜负**。

口径（对 recon 和 static **完全对称、label-free**，无 test 选配泄漏）：
  per-variable z-score(用 val 统计) → max over D → 1D 序列 → 算 AP / ROC-AUC。
随机基线的 AP = 异常率（anom%），故同时报 AP/anom 的提升倍数。

零重训，读 runs_fuse/<ent>/{val,test}_scores.npy + 数据现算 static=|x_norm|。
用法：
  python scripts/metric_compare.py --config configs/smd.yaml --recon_dir runs_fuse
  python scripts/metric_compare.py --config configs/smd.yaml --recon_dir runs_fuse --entities machine-1-1
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
from drift_sweep import postprocess, aggregate  # noqa: E402
from diagnose_regime import static_2d, auc, ALL28  # noqa: E402


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """AP = Σ (R_n − R_{n−1})·P_n（sklearn average_precision_score 的无插值定义）。"""
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
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(((rec - rec_prev) * prec).sum())


def pipe_1d(v2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """per-variable z-score(val 统计) → max over D，返回 test 的 1D 分数。"""
    _, tz = postprocess(v2, t2, "zscore")
    return aggregate(tz, "max")


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
        _, val_ds, test_ds, test_label, _ = build_smd_datasets(cfg)
        vs2, ts2 = static_2d(val_ds, cfg), static_2d(test_ds, cfg)

        rec1 = pipe_1d(vr2, tr2)
        sta1 = pipe_1d(vs2, ts2)
        T = min(len(rec1), len(sta1), len(test_label))
        lab = np.asarray(test_label[:T], dtype=np.int64)
        rec1, sta1 = rec1[:T], sta1[:T]
        rate = float(lab.mean())

        rows.append({
            "ent": ent, "rate": rate,
            "rec_ap": average_precision(rec1, lab), "sta_ap": average_precision(sta1, lab),
            "rec_auc": auc(rec1, lab), "sta_auc": auc(sta1, lab),
        })
        r = rows[-1]
        print(f"    recon_AP={r['rec_ap']:.3f}  static_AP={r['sta_ap']:.3f}  "
              f"Δ={r['rec_ap']-r['sta_ap']:+.3f}  (anom={rate*100:.1f}%)", flush=True)

    if not rows:
        print("无可评估 entity。"); return

    print(f"\n{'entity':<14} {'recon_AP':>9} {'static_AP':>10} {'ΔAP':>7} "
          f"{'rec_AUC':>8} {'sta_AUC':>8} {'anom%':>6}")
    print("-" * 70)
    for r in rows:
        print(f"{r['ent']:<14} {r['rec_ap']:>9.3f} {r['sta_ap']:>10.3f} "
              f"{r['rec_ap']-r['sta_ap']:>+7.3f} {r['rec_auc']:>8.3f} {r['sta_auc']:>8.3f} "
              f"{r['rate']*100:>5.1f}%")
    ra = np.array([r["rec_ap"] for r in rows]); sa = np.array([r["sta_ap"] for r in rows])
    print("-" * 70)
    print(f"{'MEAN':<14} {ra.mean():>9.3f} {sa.mean():>10.3f} {(ra-sa).mean():>+7.3f}")
    n = len(rows)
    print(f"\n# recon_AP 赢 static_AP (ΔAP>+0.01): {int(((ra-sa) > 0.01).sum())}/{n}")
    print(f"# recon_AP 输 static_AP (ΔAP<-0.01): {int(((ra-sa) < -0.01).sum())}/{n}")
    print("判读：recon_AP 均值与胜场稳定 > static → 模型在'正确的指标'下确实赢 trivial 基线，"
          "\n      这就是你方法值得写的核心结果（AUC-PR，不平衡 AD 的标准口径）。")


if __name__ == "__main__":
    main()
