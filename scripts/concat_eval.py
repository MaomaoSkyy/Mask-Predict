"""把 per-entity 分数拼成"整条 SMD"评估——和 AnomalyTransformer 那种"28 台拼一条、
一个阈值"的协议对齐，给出可与之并排比较的 best-F1 / POT-F1 / AUC-PR / ROC-AUC。

关键：你是 per-entity 训练，28 台分数尺度各不同，**直接拼会被某一台的量纲主导**。
所以每台先 `per-variable z-score(val 统计) → max → 再用该台 val 的 1D 分布标准化`，
把每台都校准到"偏离自己正常多少"，再拼。这样单一全局阈值才有意义。

(best-F1 / AUC-PR / ROC-AUC 只取决于 (score,label) 集合，与拼接顺序无关。)

同时报 recon 与 static(|x_norm|) 两条，方便三方(你 / 幅度基线 / AT)同协议并排。
还会把拼好的整条 recon 分数+标签存成 npy，可直接喂 eval_external_scores.py 复核。

用法：
  python scripts/concat_eval.py --config configs/smd.yaml --recon_dir runs_fuse
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.evaluation.metrics import evaluate_scores  # noqa: E402
from src.inference.pot import pot_threshold  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from drift_sweep import postprocess, aggregate  # noqa: E402
from metric_compare import average_precision  # noqa: E402
from threshold_ceiling import oracle_best_f1  # noqa: E402
from diagnose_regime import auc, static_2d, ALL28  # noqa: E402


def pipe_val_test_1d(v2: np.ndarray, t2: np.ndarray):
    """per-var z-score(val) → max → 再按该台 val 的 1D 均值/方差标准化，返回 (val_1d, test_1d)。"""
    vz, tz = postprocess(v2, t2, "zscore")
    v1, t1 = aggregate(vz, "max"), aggregate(tz, "max")
    mu, sd = v1.mean(), max(v1.std(), 1e-8)
    return (v1 - mu) / sd, (t1 - mu) / sd


def pot_best(val_1d, test_1d, label):
    best = None
    for q in (0.99, 0.95):
        for lv in (1e-3, 1e-4, 1e-5, 1e-6):
            try:
                thr = pot_threshold(val_1d, q=q, level=lv)
                r = evaluate_scores(test_1d, label, thr)
                best = r["raw_f1"] if best is None else max(best, r["raw_f1"])
            except Exception:
                pass
    return best if best is not None else float("nan")


def report(name, val_1d, test_1d, label):
    print(f"\n=== {name}  整条 SMD ===  (T={len(test_1d)}, anom={label.mean()*100:.2f}%)")
    print(f"  best_f1 (oracle)    : {oracle_best_f1(test_1d, label):.3f}")
    print(f"  pot_f1  (label-free): {pot_best(val_1d, test_1d, label):.3f}")
    print(f"  AUC-PR  (avg prec)  : {average_precision(test_1d, label):.3f}   "
          f"(随机≈{label.mean():.3f})")
    print(f"  ROC-AUC             : {auc(test_1d, label):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recon_dir", default="runs_fuse")
    ap.add_argument("--entities", nargs="+", default=ALL28)
    ap.add_argument("--save_prefix", default="mp_whole", help="存整条 recon 分数/标签的前缀")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rec_v, rec_t, sta_v, sta_t, labs = [], [], [], [], []
    n_ent = len(args.entities)
    for i, ent in enumerate(args.entities, 1):
        rd = Path(args.recon_dir) / ent
        if not (rd / "val_scores.npy").exists() or not (rd / "test_scores.npy").exists():
            print(f"[skip] {ent}: 缺分数（{rd}）", flush=True)
            continue
        cfg.data.entity = ent
        print(f"[{i}/{n_ent}] {ent} ...", flush=True)
        vr2, tr2 = np.load(rd / "val_scores.npy"), np.load(rd / "test_scores.npy")
        _, val_ds, test_ds, test_label, _ = build_smd_datasets(cfg)
        vs2, ts2 = static_2d(val_ds, cfg), static_2d(test_ds, cfg)
        label = np.asarray(test_label, dtype=np.int64)

        rv, rt = pipe_val_test_1d(vr2, tr2)
        sv, st = pipe_val_test_1d(vs2, ts2)
        T = min(len(rt), len(st), len(label))
        rec_v.append(rv); rec_t.append(rt[:T])
        sta_v.append(sv); sta_t.append(st[:T])
        labs.append(label[:T])

    if not labs:
        print("无可评估 entity。"); return

    RV, RT = np.concatenate(rec_v), np.concatenate(rec_t)
    SV, ST = np.concatenate(sta_v), np.concatenate(sta_t)
    LAB = np.concatenate(labs)

    report("recon (mask-predict)", RV, RT, LAB)
    report("static (|x_norm| 基线)", SV, ST, LAB)

    np.save(f"{args.save_prefix}_scores.npy", RT)
    np.save(f"{args.save_prefix}_labels.npy", LAB)
    np.save(f"{args.save_prefix}_val.npy", RV)
    print(f"\n[saved] {args.save_prefix}_{{scores,labels,val}}.npy  "
          f"→ 可喂 eval_external_scores.py 复核，或与 AnomTrans 整条数并排")
    print("提示：把 AnomalyTransformer 的整条 raw best_f1/AUC-PR 跟上面 recon 那块直接比，就是同协议对决。")


if __name__ == "__main__":
    main()
