"""导出训练好的源侧变量门 g_d，并（可选）与 LOO contribution 对照。

用途
----
门训练完后回答两个问题：
  1. 门有没有"分化"？如果 38 个门都 ≈ 0.88（初值附近没动）→ 模型想用全部变量，
     与 LOO 负面结论一致，门没找到可关的变量。
  2. 门关掉的变量，跟 LOO 里"删了 F1 不降"的 noise 候选重合吗？
     重合 → 门无监督地复现了 oracle 特征选择（有意义的 contribution）。
     不重合 → 门关的是源侧无用变量，与目标侧 LOO 本就是两根轴（也可能各有道理）。

只读 ckpt，不重训、不需要 label。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.mask_predict import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--closed", type=float, default=0.5,
                        help="门 < 此值视为'关闭'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.entity:
        cfg.data.entity = args.entity
    if not bool(getattr(cfg.model, "var_gate", False)):
        print("[warn] cfg.model.var_gate=false：该 ckpt 可能没有门参数。")

    model = build_model(cfg)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"])
    g = model.gate_values().detach().cpu().numpy()  # (D,)
    D = g.shape[0]

    order = np.argsort(g)  # 升序：最该关的在前
    closed = [int(i) for i in order if g[i] < args.closed]

    print(f"# entity={cfg.data.entity}  D={D}  "
          f"gate mean={g.mean():.3f} min={g.min():.3f} max={g.max():.3f}")
    print(f"# 门是否分化：std={g.std():.3f}  "
          f"({'已分化' if g.std() > 0.05 else '几乎没动→模型想用全部变量'})")
    print(f"# 关闭(<{args.closed})的变量: {closed}  (共 {len(closed)}/{D})\n")

    print(f"{'rank':>4} {'col':>4} {'gate':>7}  bar")
    print("-" * 40)
    for rank, i in enumerate(order):
        i = int(i)
        bar = "█" * int(round(g[i] * 20))
        tag = "  closed" if g[i] < args.closed else ""
        print(f"{rank:>4} {i:>4} {g[i]:>7.3f}  {bar}{tag}")


if __name__ == "__main__":
    main()
