"""把**外部方法**(AnomalyTransformer / AE / LSTM…)dump 出的逐点异常分,放到本项目
**同一把尺子**上评估——和你自己模型完全同口径,做真·apples-to-apples 对比。

用法:外部方法只需存两个 npy：
  - scores.npy : 逐点异常分，(T,) 或 (T,D)(2D 会按 --agg 聚合，默认 max over 变量)
  - labels.npy : 逐点 0/1 标签，(T,)  (也支持 .txt)
然后：
  python scripts/eval_external_scores.py --scores at_energy.npy --labels at_labels.npy --name AnomTrans
  # 若该方法也有 val 段分数，给 --val_scores 可一并算 POT(label-free)F1，与你的口径一致
  python scripts/eval_external_scores.py --scores at_energy.npy --labels at_labels.npy \
         --val_scores at_val_energy.npy --name AnomTrans

输出四个数(与 metric_compare / threshold_ceiling 同实现)：
  best_f1   : 扫遍阈值的 oracle 逐点 F1（= 论文"best F1"口径，偷看 label 的上界）
  pot_f1    : POT-on-val 的 label-free 逐点 F1（需 --val_scores；否则跳过）
  AUC-PR    : average precision（阈值无关，不平衡 AD 标准指标）
  ROC-AUC   : 排序可分性
对照：你的 mask-predict 28 台均值 best_f1≈0.558 / pot_f1≈0.514 / AUC-PR≈0.362。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.inference.pot import pot_threshold  # noqa: E402
from src.evaluation.metrics import evaluate_scores  # noqa: E402
from metric_compare import average_precision  # noqa: E402
from threshold_ceiling import oracle_best_f1  # noqa: E402
from diagnose_regime import auc  # noqa: E402


def _load_1d_scores(path: str, agg: str) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 2:
        return a.max(axis=1) if agg == "max" else a.mean(axis=1)
    return a.ravel()


def _load_labels(path: str) -> np.ndarray:
    p = Path(path)
    a = np.loadtxt(p, dtype=np.int64) if p.suffix == ".txt" else np.load(p)
    return np.asarray(a, dtype=np.int64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="逐点异常分 npy (T,) 或 (T,D)")
    ap.add_argument("--labels", required=True, help="逐点 0/1 标签 npy/txt (T,)")
    ap.add_argument("--val_scores", default=None, help="val 段分数 npy，给了才算 POT-on-val F1")
    ap.add_argument("--agg", default="max", choices=["max", "mean"], help="2D 分数按变量维聚合方式")
    ap.add_argument("--name", default="external", help="方法名，仅用于打印")
    args = ap.parse_args()

    score = _load_1d_scores(args.scores, args.agg)
    label = _load_labels(args.labels)
    T = min(len(score), len(label))
    score, label = score[:T], label[:T]
    rate = float(label.mean())

    best = oracle_best_f1(score, label)
    ap_ = average_precision(score, label)
    roc = auc(score, label)

    pot_f1 = None
    if args.val_scores:
        val = _load_1d_scores(args.val_scores, args.agg)
        for q in (0.99, 0.95):
            for lv in (1e-3, 1e-4, 1e-5, 1e-6):
                try:
                    thr = pot_threshold(val, q=q, level=lv)
                    r = evaluate_scores(score, label, thr)
                    pot_f1 = r["raw_f1"] if pot_f1 is None else max(pot_f1, r["raw_f1"])
                except Exception:
                    pass

    print(f"\n=== {args.name} ===  (T={T}, anom={rate*100:.2f}%)")
    print(f"  best_f1 (oracle)   : {best:.3f}")
    if pot_f1 is not None:
        print(f"  pot_f1  (label-free): {pot_f1:.3f}")
    else:
        print(f"  pot_f1  (label-free): —（给 --val_scores 才算）")
    print(f"  AUC-PR  (avg prec) : {ap_:.3f}   (随机基线≈{rate:.3f})")
    print(f"  ROC-AUC            : {roc:.3f}")
    print(f"\n对照 mask-predict 28台均值: best_f1≈0.558  pot_f1≈0.514  AUC-PR≈0.362")
    print("注意：若该方法把 SMD 拼成整条评估、而你是 per-entity，协议不同，硬比前先对齐划分。")


if __name__ == "__main__":
    main()
