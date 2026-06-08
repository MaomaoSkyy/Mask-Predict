"""诊断"raw_f1≈0 但 pa_f1≈1"到底是哪种病。不画图、纯数值、远端可跑。

对某个 ckpt 在某条打分轴上的 1-D 异常分，给三个**阈值无关**的决定性指标：

  1) 点级 ROC-AUC / AP：分数到底分不分得开异常点（与阈值无关）。
     高(>0.8) 但 raw_f1≈0 → 是**阈值/POT 选得烂**，不是检测不到。
     ≈0.5            → 分数根本没分开，真·没检到。
  2) 最佳时间 lag：分数 vs label 做互相关，峰值 lag 偏离 0 → **窗口/stride 时间错位 bug**，
     raw 是"假 0"，对齐一修就跳。
  3) 段内 onset vs body：每段异常里分数是只在开头抬头、还是全段抬头。
     onset≫body → "只标变化点、持续段被模型适应掉"（重构范式固有局限）。

用法：
  python scripts/diagnose_scores.py --config configs/smd.yaml --entity machine-3-2 \
         --ckpt runs_staged/machine-3-2/last.pt --axis time_checkerboard
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False  # 与 train.py 一致：规避 depthwise conv 在某些 cuDNN 下 NOT_INITIALIZED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.inference.scorer import score_series  # noqa: E402
from src.models.mask_predict import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from drift_sweep import aggregate, postprocess  # noqa: E402


def _apply_override(cfg, dotted):
    key, _, val = dotted.partition("=")
    low = val.lower()
    v = (low == "true") if low in ("true", "false") else val
    if isinstance(v, str):
        for cast in (int, float):
            try:
                v = cast(val); break
            except ValueError:
                pass
    parts = key.split(".")
    ns = cfg
    for p in parts[:-1]:
        ns = getattr(ns, p)
    setattr(ns, parts[-1], v)
    d = cfg._raw
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = v


def roc_auc(score, label):
    """rank-based ROC-AUC（Mann-Whitney），无需 sklearn。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    n_pos = int(pos.sum()); n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def avg_precision(score, label):
    """点级 AP（PR 曲线下面积），对稀有异常比 AUC 更敏感。"""
    order = np.argsort(-score, kind="mergesort")
    y = label[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(label.sum()), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def corr_at_lag(s, label, lag):
    T = len(s)
    if lag >= 0:
        a, b = s[:T - lag], label[lag:]
    else:
        k = -lag; a, b = s[k:], label[:T - k]
    if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def segments(label):
    """返回 [(start, end_exclusive), ...]。"""
    segs = []
    in_seg = False
    for i, v in enumerate(label):
        if v == 1 and not in_seg:
            start = i; in_seg = True
        elif v == 0 and in_seg:
            segs.append((start, i)); in_seg = False
    if in_seg:
        segs.append((start, len(label)))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--axis", default="time_checkerboard",
                    choices=["time_checkerboard", "variable_rotation"])
    ap.add_argument("--agg", default="topk5", help="max|topk3|topk5|mean")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.data.entity = args.entity
    for ov in args.overrides:
        _apply_override(cfg, ov)
    if float(getattr(cfg.mask, "var_mask_prob", 0.0)) >= 1.0:
        cfg.mask.var_mask_prob = 0.5  # 兜底通过 scorer 守卫

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds, test_ds, test_label, _ = build_smd_datasets(cfg)
    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    cfg.inference.mode = args.axis
    val = score_series(model, val_ds, cfg, device)    # (Tv, D)
    test = score_series(model, test_ds, cfg, device)  # (Tt, D)
    vp, tp = postprocess(val, test, "zscore")         # per-variable z-score（用 val 统计）
    s = aggregate(tp, args.agg)                        # (Tt,)
    label = np.asarray(test_label, dtype=np.int64)
    T = min(len(s), len(label))
    s, label = s[:T], label[:T]

    print(f"# entity={args.entity}  axis={args.axis}  agg={args.agg}  T={T}  "
          f"异常占比={label.mean():.3f}")

    # 1) 阈值无关的可分性
    auc = roc_auc(s, label)
    ap_ = avg_precision(s, label)
    print(f"\n[1] 可分性（阈值无关）: ROC-AUC={auc:.3f}  AP={ap_:.3f}  "
          f"(随机 AUC=0.5, AP≈异常占比={label.mean():.3f})")

    # 2) 时间对齐
    lags = list(range(-40, 41))
    corrs = [corr_at_lag(s, label, l) for l in lags]
    bi = int(np.argmax(corrs))
    best_lag, best_corr = lags[bi], corrs[bi]
    corr0 = corr_at_lag(s, label, 0)
    print(f"\n[2] 时间对齐: corr@lag0={corr0:.3f}  best_lag={best_lag} (corr={best_corr:.3f})")

    # 3) 段内 onset vs body
    segs = segments(label)
    lens = np.array([e - st for st, e in segs]) if segs else np.array([0])
    s_z = (s - s.mean()) / (s.std() + 1e-8)
    onset_vals, body_vals, peak_fracs = [], [], []
    for st, e in segs:
        seg = s_z[st:e]
        L = len(seg)
        k = max(1, min(3, L))
        onset_vals.append(seg[:k].mean())
        if L > k:
            body_vals.append(seg[k:].mean())
        peak_fracs.append(int(np.argmax(seg)) / max(1, L - 1))
    out_mask = label == 0
    outside = float(s_z[out_mask].mean()) if out_mask.any() else float("nan")
    onset_m = float(np.mean(onset_vals)) if onset_vals else float("nan")
    body_m = float(np.mean(body_vals)) if body_vals else float("nan")
    peak_front = float(np.mean(np.array(peak_fracs) < 0.25)) if peak_fracs else float("nan")
    print(f"\n[3] 段结构: 段数={len(segs)}  段长 中位={int(np.median(lens))} "
          f"max={int(lens.max())}")
    print(f"    段内 z-score:  onset(前3点)={onset_m:.2f}  body(其余)={body_m:.2f}  "
          f"段外={outside:.2f}")
    print(f"    峰值落在段前 25% 的比例={peak_front:.2f}")

    # ---- 判定 ----
    print("\n=== 判定 ===")
    if abs(best_lag) >= 2 and best_corr > corr0 + 0.05:
        print(f"# B｜时间错位嫌疑：best_lag={best_lag}≠0 且相关明显更高 → 打分/label 错开了。")
        print("#   去查 score_series 的窗口→原始时间戳映射、test_stride、label 截断。raw 是假 0。")
    elif not np.isnan(auc) and auc > 0.8:
        print(f"# C｜阈值问题：AUC={auc:.3f} 其实分得很开，但 raw_f1≈0 → 是 POT/阈值选烂了。")
        print("#   分数本身能检到，问题在阈值选择（POT level/agg），不是模型也不是范式。")
    elif not np.isnan(onset_m) and not np.isnan(body_m) and onset_m > body_m + 0.8:
        print(f"# A｜onset-only：段内 onset({onset_m:.2f})≫body({body_m:.2f}) → 只标变化点、"
              f"持续段被模型适应掉。")
        print("#   这是'从上下文重构'范式对长持续/电平漂移异常的固有局限，换架构没用。")
        print("#   出路：变化点检测视角 / 显式建模'正常基线' / 或接受 pa 口径。")
    elif not np.isnan(auc) and auc < 0.6:
        print(f"# D｜真没检到：AUC={auc:.3f}≈随机 → 分数没分开。模型确实没学到这台的正常模式。")
    else:
        print(f"# 不典型：AUC={auc:.3f} lag={best_lag} onset={onset_m:.2f} body={body_m:.2f}。"
              f"贴回来一起看。")


if __name__ == "__main__":
    main()
