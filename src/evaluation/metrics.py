import numpy as np


def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """OmniAnomaly 提出的 point-adjust：若一段连续真实异常中有任意一点被检出，
    则该段全部记为检出。注意：单独看 PA-F1 会虚高，应同时报 raw F1。
    """
    pred = pred.copy().astype(np.int64)
    label = label.astype(np.int64)
    anomaly = False
    start = 0
    for i in range(len(label)):
        if label[i] == 1 and not anomaly:
            anomaly = True
            start = i
        elif label[i] == 0 and anomaly:
            if pred[start:i].any():
                pred[start:i] = 1
            anomaly = False
    if anomaly and pred[start:].any():
        pred[start:] = 1
    return pred


def prf(pred: np.ndarray, label: np.ndarray) -> dict:
    pred = pred.astype(np.int64)
    label = label.astype(np.int64)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_scores(scores_per_point: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    """scores_per_point: 1D (T,) 或 2D (T, D)。2D 时按变量维取 max 聚合。"""
    if scores_per_point.ndim == 2:
        s = scores_per_point.max(axis=1)
    else:
        s = scores_per_point
    T = min(len(s), len(label))
    s, label = s[:T], label[:T]
    pred = (s > threshold).astype(np.int64)
    raw = prf(pred, label)
    pred_pa = point_adjust(pred, label)
    pa = prf(pred_pa, label)
    return {
        "threshold": threshold,
        "raw_f1": raw["f1"], "raw_p": raw["precision"], "raw_r": raw["recall"],
        "pa_f1": pa["f1"], "pa_p": pa["precision"], "pa_r": pa["recall"],
    }
