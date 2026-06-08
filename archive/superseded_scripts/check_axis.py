"""轴诊断：同一个模型，分别用变量维打分（variable_rotation）和时间维打分
（time_checkerboard）跑完整 drift_sweep，比较哪条轴抓住了该 entity 的异常；
再给一个 max 融合预览。

用途：验证"模型学会时间 mask 后，时间维打分能否救起 variable_rotation≈0 的弱 entity"。

前提：ckpt 必须用 var_mask_prob<1（混合 mask）训练过，否则模型没学过时间 mask，
time 路打分无意义。本脚本内部把 cfg.mask.var_mask_prob 兜底成 0.5 以通过 scorer 的守卫。

注意一个 train/inference mask 口径差异：训练的时间 mask 是随机 (t,d) cell，
而 time_checkerboard 打分 mask 的是整个时间步（该 t 所有变量）。若 time 路表现不及预期，
下一步可加一个 cell 级打分模式（与训练口径一致）再试。

只读 ckpt，需要 test_label。
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
from drift_sweep import _zscore, postprocess, search_one  # noqa: E402

MODES = ["raw", "zscore", "smooth_3", "smooth_5", "smooth_10",
         "runmax_3", "runmax_5", "runmax_10", "combine_smooth_5", "combine_smooth_10"]


def full_sweep(val, test, label):
    """对 (T,D) 分数跑 后处理×聚合×POT 全搜索，返回最佳 dict（含 raw_f1/mode/agg/q/level）。"""
    best = None
    for m in MODES:
        try:
            vp, tp = postprocess(val, test, m)
            b = search_one(vp, tp, label)
        except Exception:
            continue
        if b and (best is None or b["raw_f1"] > best["raw_f1"]):
            best = {**b, "mode": m}
    return best


def _apply_override(cfg, dotted: str):
    """形如 model.encoder=dualaxis；同改 cfg 与 cfg._raw。"""
    key, _, val = dotted.partition("=")
    low = val.lower()
    if low in ("true", "false"):
        v = (low == "true")
    else:
        v = val
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="覆盖配置，如 --set model.encoder=dualaxis")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.data.entity = args.entity
    for ov in args.overrides:
        _apply_override(cfg, ov)
    if float(getattr(cfg.mask, "var_mask_prob", 0.0)) >= 1.0:
        print("[warn] cfg.mask.var_mask_prob>=1：假设该 ckpt 实际是用 <1 训练的；"
              "内部兜底成 0.5 以允许 time 打分。若 ckpt 没学过时间 mask，time 路结果无意义。")
        cfg.mask.var_mask_prob = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds, test_ds, test_label, _ = build_smd_datasets(cfg)
    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    print(f"# entity={args.entity}  val_windows={len(val_ds)}  test_windows={len(test_ds)}")

    streams = {}
    best = {}
    for mode in ["variable_rotation", "time_checkerboard"]:
        cfg.inference.mode = mode
        v = score_series(model, val_ds, cfg, device)   # (T, D)
        t = score_series(model, test_ds, cfg, device)
        streams[mode] = (v, t)
        best[mode] = full_sweep(v, t, test_label)

    # max 融合预览：各路 per-variable z-score 后逐 cell 取 max
    (vv, tv) = streams["variable_rotation"]
    (vt, tt) = streams["time_checkerboard"]
    vv_z, tv_z = _zscore(vv, tv)
    vt_z, tt_z = _zscore(vt, tt)
    bf = full_sweep(np.maximum(vv_z, vt_z), np.maximum(tv_z, tt_z), test_label)

    def line(name, b):
        if b is None:
            print(f"{name:<22} (no valid)")
        else:
            print(f"{name:<22} raw_f1={b['raw_f1']:.3f}  pa_f1={b['pa_f1']:.3f}  "
                  f"mode={b['mode']}  agg={b['agg']}  q={b['q']:.2f}  level={b['level']:.0e}")

    print(f"\n{'轴':<22} 最佳配置 + raw_f1")
    print("-" * 72)
    line("variable_rotation", best["variable_rotation"])
    line("time_checkerboard", best["time_checkerboard"])
    line("fuse(max)", bf)

    fv = best["variable_rotation"]["raw_f1"] if best["variable_rotation"] else 0.0
    ft = best["time_checkerboard"]["raw_f1"] if best["time_checkerboard"] else 0.0
    ff = bf["raw_f1"] if bf else 0.0
    print("\n# 解读：")
    if ft > fv + 0.05:
        print(f"# → 时间维打分 ({ft:.3f}) 明显高于变量维 ({fv:.3f})：该 entity 异常偏时序型，"
              f"variable_rotation 本来就瞎。融合/时间打分是对的方向。")
    elif fv > ft + 0.05:
        print(f"# → 变量维 ({fv:.3f}) 仍强于时间维 ({ft:.3f})：该 entity 异常偏跨变量关联型。")
    else:
        print(f"# → 两轴接近 (var={fv:.3f} time={ft:.3f})。")
    if ff > max(fv, ft) + 0.02:
        print(f"# → 融合 ({ff:.3f}) 高于任一单轴：两轴互补，融合有真实增益。")
    else:
        print(f"# → 融合 ({ff:.3f}) 未超过单轴最好：互补性有限，取单轴最强即可。")


if __name__ == "__main__":
    main()
