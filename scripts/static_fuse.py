"""重构分 + 静态"绝对异常度"分 融合评估（north star = raw_f1）。

动机：重构误差是"上下文意外度"，只在变化点/相关断裂抬头，看不见**持续电平漂移**
（machine-3-2 那种长段异常被模型适应掉）。补一个**不依赖上下文**的静态分：
    s_static[t,d] = |x_norm[t,d]|   （RobustScaler 后，离正常中位多少个 IQR）
持续漂移 → 全段都高 → 正好补 body。两路 late-fusion（各自 z-score 后融合）。

每个 entity 报四个 raw_f1：recon-only / static-only / fused / 单路最强，并出汇总。
recon 分数从 --recon_dir/<entity>/{val,test}_scores.npy 读（需先 eval 保存）；
static 分数从数据现算（无需模型）。只读，需要 test_label。

用法：
  # 先在 machine-3-2 上试（用已存在的 recon 分数）
  python scripts/static_fuse.py --config configs/smd.yaml \
         --entities machine-3-2 --recon_dir runs_abl/nogate
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.evaluation.metrics import evaluate_scores  # noqa: E402
from src.inference.pot import pot_threshold  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from drift_sweep import aggregate, postprocess  # noqa: E402

MODES = ["raw", "zscore", "smooth_3", "smooth_5", "smooth_10",
         "runmax_3", "runmax_5", "runmax_10", "combine_smooth_5", "combine_smooth_10"]
AGGS = ["max", "topk3", "topk5", "mean"]
QS = [0.95, 0.99]
LEVELS = [1e-3, 1e-4, 1e-5, 1e-6]


def best_1d(val2d, test2d, label):
    """对 (T,D) 分数扫 后处理×聚合×POT，返回最佳 raw_f1 及最优 1-D 序列 (vs, ts)。"""
    best = None
    for m in MODES:
        try:
            vp, tp = postprocess(val2d, test2d, m)
        except Exception:
            continue
        for agg in AGGS:
            try:
                vs, ts = aggregate(vp, agg), aggregate(tp, agg)
            except Exception:
                continue
            for q in QS:
                for lv in LEVELS:
                    try:
                        thr = pot_threshold(vs, q=q, level=lv)
                        r = evaluate_scores(ts, label, thr)
                        if best is None or r["raw_f1"] > best["raw_f1"]:
                            best = {"raw_f1": r["raw_f1"], "pa_f1": r["pa_f1"],
                                    "vs": vs, "ts": ts, "mode": m, "agg": agg}
                    except Exception:
                        pass
    return best


def _z(v, t):
    mu, sd = v.mean(), max(v.std(), 1e-8)
    return (v - mu) / sd, (t - mu) / sd


def fuse_best(vr, tr, vs, ts, label):
    """两路 1-D late-fusion：各自 z-score 后 max / α-blend，扫 POT 取最佳。"""
    nv = min(len(vr), len(vs)); nt = min(len(tr), len(ts), len(label))
    vrz, trz = _z(vr[:nv], tr[:nt]); vsz, tsz = _z(vs[:nv], ts[:nt])
    lab = label[:nt]
    best = None
    cands = [("max", None)] + [("blend", a) for a in (0.25, 0.5, 0.75)]
    for kind, a in cands:
        if kind == "max":
            vf, tf = np.maximum(vrz, vsz), np.maximum(trz, tsz)
        else:
            vf, tf = a * vrz + (1 - a) * vsz, a * trz + (1 - a) * tsz
        for q in QS:
            for lv in LEVELS:
                try:
                    thr = pot_threshold(vf, q=q, level=lv)
                    r = evaluate_scores(tf, lab, thr)
                    if best is None or r["raw_f1"] > best["raw_f1"]:
                        best = {"raw_f1": r["raw_f1"], "pa_f1": r["pa_f1"],
                                "how": kind if a is None else f"blend{a}"}
                except Exception:
                    pass
    return best


def static_2d(dataset, cfg):
    """从数据集现算静态分 (T,D)=|x_norm|，按 score_series 同样的拼接顺序（shuffle=False）。"""
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    chunks = [b.abs().numpy() for b in loader]  # b: (B,T,D)
    a = np.concatenate(chunks, axis=0)
    return a.reshape(-1, a.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--entities", nargs="+", required=True)
    ap.add_argument("--recon_dir", required=True, help="recon 分数所在目录（<dir>/<entity>/*.npy）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = []
    for ent in args.entities:
        cfg.data.entity = ent
        rd = Path(args.recon_dir) / ent
        if not (rd / "val_scores.npy").exists():
            print(f"[skip] {ent}: 缺 recon 分数（{rd}）")
            continue
        vr2 = np.load(rd / "val_scores.npy"); tr2 = np.load(rd / "test_scores.npy")
        _, val_ds, test_ds, test_label, _ = build_smd_datasets(cfg)
        vs2, ts2 = static_2d(val_ds, cfg), static_2d(test_ds, cfg)
        label = np.asarray(test_label, dtype=np.int64)

        recon = best_1d(vr2, tr2, label)
        static = best_1d(vs2, ts2, label)
        if recon is None or static is None:
            print(f"[skip] {ent}: sweep 失败"); continue
        fused = fuse_best(recon["vs"], recon["ts"], static["vs"], static["ts"], label)
        rows.append({"entity": ent,
                     "recon": recon["raw_f1"], "static": static["raw_f1"], "fused": fused["raw_f1"],
                     "recon_pa": recon["pa_f1"], "static_pa": static["pa_f1"],
                     "fused_pa": fused["pa_f1"], "how": fused["how"]})

    if not rows:
        print("无可评估 entity。"); return

    print(f"\n{'entity':<16} {'recon':>7} {'static':>7} {'fused':>7} {'Δ(f-r)':>8} "
          f"{'best1':>7}  fuse")
    print("-" * 70)
    for r in rows:
        best1 = max(r["recon"], r["static"])
        print(f"{r['entity']:<16} {r['recon']:>7.3f} {r['static']:>7.3f} {r['fused']:>7.3f} "
              f"{r['fused']-r['recon']:>+8.3f} {best1:>7.3f}  {r['how']}")
    rec = np.array([r["recon"] for r in rows]); sta = np.array([r["static"] for r in rows])
    fus = np.array([r["fused"] for r in rows])
    print("-" * 70)
    print(f"{'MEAN':<16} {rec.mean():>7.3f} {sta.mean():>7.3f} {fus.mean():>7.3f} "
          f"{(fus-rec).mean():>+8.3f} {np.maximum(rec,sta).mean():>7.3f}")
    n = len(rows)
    print(f"\n# fused > recon 的 entity: {int((fus > rec + 0.005).sum())}/{n}")
    print(f"# static 单路就 >= recon 的 entity: {int((sta >= rec - 1e-9).sum())}/{n} "
          f"（多了要警惕：trivial 基线打平深度模型）")
    print(f"# fused 相比 max(recon,static) 还更高的: "
          f"{int((fus > np.maximum(rec, sta) + 0.005).sum())}/{n}（真互补才会 >）")

    # ---- PA-F1（point-adjust，与多数 SMD 论文同口径；会显著虚高，仅供对照标题数字）----
    print(f"\n# === PA-F1（point-adjust，论文标题数字多是这个口径，注意虚高）===")
    print(f"{'entity':<16} {'recon':>7} {'static':>7} {'fused':>7}")
    print("-" * 48)
    for r in rows:
        print(f"{r['entity']:<16} {r['recon_pa']:>7.3f} {r['static_pa']:>7.3f} {r['fused_pa']:>7.3f}")
    recp = np.array([r["recon_pa"] for r in rows]); stap = np.array([r["static_pa"] for r in rows])
    fusp = np.array([r["fused_pa"] for r in rows])
    print("-" * 48)
    print(f"{'MEAN':<16} {recp.mean():>7.3f} {stap.mean():>7.3f} {fusp.mean():>7.3f}")


if __name__ == "__main__":
    main()
