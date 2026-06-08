"""[兼容保留] 早期的阈值扫描，仅做聚合 × POT 参数。

**推荐改用 scripts/drift_sweep.py**，它在此基础上额外扫描时序后处理
（z-score / smooth / runmax / combine），实测 SMD 上能把均值 raw_f1
从 0.37 → 0.50（详见 CLAUDE.md "drift 后处理"）。

本脚本仍保留 both-mode 融合策略，以便对 mode=both 的旧实验做对比。
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_scores  # noqa: E402
from src.inference.pot import pot_threshold  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def aggregate(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return scores.max(axis=1)
    if mode.startswith("topk"):
        k = int(mode[4:])
        return np.sort(scores, axis=1)[:, -k:].mean(axis=1)
    if mode == "mean":
        return scores.mean(axis=1)
    raise ValueError(mode)


def val_zscore(val_s: np.ndarray, test_s: np.ndarray) -> tuple:
    """用 val 统计量做 z-score（验证集近似纯净）。"""
    m = val_s.mean()
    s = val_s.std().clip(min=1e-8)
    return (val_s - m) / s, (test_s - m) / s


def run_single(val_s: np.ndarray, test_s: np.ndarray, label: np.ndarray, tag: str):
    """对一组已聚合到 (T,) 的分数扫 POT 参数。返回 best 行。"""
    qs = [0.95, 0.99]
    levels = [1e-3, 1e-4, 1e-5, 1e-6]
    best = None
    for q in qs:
        for lv in levels:
            try:
                thr = pot_threshold(val_s, q=q, level=lv)
                r = evaluate_scores(test_s, label, thr)
                row = (tag, q, lv, thr, r["raw_f1"], r["raw_p"], r["raw_r"], r["pa_f1"])
                print(f"{tag:>22} {q:5.2f} {lv:8.0e} {thr:9.4f} "
                      f"{r['raw_f1']:8.3f} {r['raw_p']:8.3f} {r['raw_r']:8.3f} {r['pa_f1']:8.3f}")
                if best is None or r["raw_f1"] > best[4]:
                    best = row
            except Exception as e:
                print(f"{tag:>22} {q:5.2f} {lv:8.0e}  err: {e}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--agg", default="topk3",
                        help="变量维聚合: max | topk1 | topk3 | topk5 | mean")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.entity:
        cfg.data.entity = args.entity

    run_dir = Path(cfg.train.save_dir) / cfg.data.entity
    label_path = Path(cfg.data.root) / "test_label" / f"{cfg.data.entity}.txt"
    label = np.loadtxt(label_path, dtype=np.int64)

    has_both = (run_dir / "val_scores_time.npy").exists() and (run_dir / "val_scores_var.npy").exists()

    header = (f"{'strategy':>22} {'q':>5} {'level':>8} {'thr':>9} "
              f"{'raw_f1':>8} {'raw_p':>8} {'raw_r':>8} {'pa_f1':>8}")
    print(header); print("-" * len(header))

    all_best = []

    if not has_both:
        # 旧路径：单路扫聚合 × POT
        val_scores = np.load(run_dir / "val_scores.npy")
        test_scores = np.load(run_dir / "test_scores.npy")
        print(f"# single-path  val_points={len(val_scores)}  D={val_scores.shape[1]}")
        for agg in ["max", "topk1", "topk3", "topk5", "mean"]:
            b = run_single(aggregate(val_scores, agg), aggregate(test_scores, agg), label, tag=agg)
            if b: all_best.append(b)
    else:
        val_t = np.load(run_dir / "val_scores_time.npy")
        val_v = np.load(run_dir / "val_scores_var.npy")
        test_t = np.load(run_dir / "test_scores_time.npy")
        test_v = np.load(run_dir / "test_scores_var.npy")
        agg = args.agg
        print(f"# both-path  val_points={len(val_t)}  D={val_t.shape[1]}  agg={agg}")

        v_t = aggregate(val_t, agg);  v_v = aggregate(val_v, agg)
        t_t = aggregate(test_t, agg); t_v = aggregate(test_v, agg)

        # 1) 单路基线
        all_best.append(run_single(v_t, t_t, label, tag="time_only"))
        all_best.append(run_single(v_v, t_v, label, tag="var_only"))

        # 2) val-stats z-score 后融合
        zv_t, zt_t = val_zscore(v_t, t_t)
        zv_v, zt_v = val_zscore(v_v, t_v)
        for w in [0.25, 0.5, 0.75]:
            zv_mix = w * zv_v + (1 - w) * zv_t
            zt_mix = w * zt_v + (1 - w) * zt_t
            all_best.append(run_single(zv_mix, zt_mix, label, tag=f"z_mean(w_v={w})"))
        # max 融合
        all_best.append(run_single(
            np.maximum(zv_t, zv_v), np.maximum(zt_t, zt_v), label, tag="z_max"))

        # 3) OR 双阈值（任一路超过自己的 POT → 异常）
        for q in [0.95, 0.99]:
            for lv in [1e-3, 1e-4, 1e-5]:
                try:
                    thr_t = pot_threshold(v_t, q=q, level=lv)
                    thr_v = pot_threshold(v_v, q=q, level=lv)
                    pred = ((t_t > thr_t) | (t_v > thr_v)).astype(np.int64)
                    from src.evaluation.metrics import prf, point_adjust
                    T = min(len(pred), len(label))
                    raw = prf(pred[:T], label[:T])
                    pa = prf(point_adjust(pred[:T], label[:T]), label[:T])
                    print(f"{'OR_thr':>22} {q:5.2f} {lv:8.0e} "
                          f"({thr_t:.3f},{thr_v:.3f})  "
                          f"{raw['f1']:8.3f} {raw['precision']:8.3f} {raw['recall']:8.3f} {pa['f1']:8.3f}")
                    row = ("OR_thr", q, lv, max(thr_t, thr_v), raw["f1"], raw["precision"], raw["recall"], pa["f1"])
                    all_best.append(row)
                except Exception as e:
                    print(f"OR_thr q={q} lv={lv}  err: {e}")

    all_best = [b for b in all_best if b]
    if all_best:
        best = max(all_best, key=lambda r: r[4])
        print("-" * len(header))
        print(f"BEST raw_f1: strategy={best[0]} q={best[1]} level={best[2]:.0e} "
              f"thr={best[3]:.4f} raw_f1={best[4]:.3f} pa_f1={best[7]:.3f}")


if __name__ == "__main__":
    main()
