"""推理 + 阈值 + 指标。单卡运行。"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False  # 与 train.py 一致：规避 depthwise conv 在某些 cuDNN 下 NOT_INITIALIZED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.evaluation.metrics import evaluate_scores  # noqa: E402
from src.inference.pot import pot_threshold  # noqa: E402
from src.inference.scorer import score_series  # noqa: E402
from src.models.mask_predict import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="覆盖配置，如 --set model.encoder=dualaxis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.entity:
        cfg.data.entity = args.entity
    for ov in args.overrides:
        _apply_override(cfg, ov)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds, test_ds, test_label, _ = build_smd_datasets(cfg)

    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd["model"])

    print(f"[eval] entity={cfg.data.entity}  val_windows={len(val_ds)}  test_windows={len(test_ds)}")

    val_out = score_series(model, val_ds, cfg, device)
    test_out = score_series(model, test_ds, cfg, device)

    out_dir = Path(cfg.train.save_dir) / cfg.data.entity
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(val_out, dict):
        # both 模式：保存两路，融合交给 sweep
        np.save(out_dir / "val_scores_time.npy", val_out["time"])
        np.save(out_dir / "val_scores_var.npy", val_out["var"])
        np.save(out_dir / "test_scores_time.npy", test_out["time"])
        np.save(out_dir / "test_scores_var.npy", test_out["var"])
        # 主报告用 var 路（当前表现最强）
        val_scores, test_scores = val_out["var"], test_out["var"]
        np.save(out_dir / "val_scores.npy", val_scores)
        np.save(out_dir / "test_scores.npy", test_scores)
        print("[eval] mode=both: saved time/var scores separately; report uses var path")
    else:
        val_scores, test_scores = val_out, test_out
        np.save(out_dir / "val_scores.npy", val_scores)
        np.save(out_dir / "test_scores.npy", test_scores)

    val_agg = val_scores.max(axis=1)
    thr = pot_threshold(val_agg, q=cfg.pot.q, level=cfg.pot.level)
    print(f"[eval] POT threshold = {thr:.6f}")
    result = evaluate_scores(test_scores, test_label, thr)
    print("[eval] metrics:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(out_dir / "eval.txt", "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    print(f"[eval] saved to {out_dir}")


if __name__ == "__main__":
    main()
