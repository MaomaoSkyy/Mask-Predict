"""门对照实验：验证源侧门是否"非退化"——把某变量作为源关掉，重建 val_loss 是否真的变差。

背景
----
门是乘在变量维 attention 的 value 上的**仿射插值** v_eff = g·v + (1-g)·v_absent。
若实现退化（旧的纯乘性门 v_eff = g·v），均匀关门会被下游**共享**线性层（×1/c）精确补偿
→ 重建对门完全不敏感 → L1 把所有门零代价压到 0，门测不出任何信息。

本脚本用**固定 mask** 做 A/B，直接在重建 loss 层面证伪/确认修复：

  baseline   : 用训练得到的门
  close d    : 把变量 d 的门强制设 0（其余维保持训练值），其它不变
  delta[d]   = loss(close d) - loss(baseline)
               > 0 越大 → 变量 d 作为"源"越关键（别人重建时依赖它）
               ≈ 0     → 关不关都一样 → d 作为源无信息（或门已经把它关了）

两个 sanity：
  all-open  (门全 1)   ：参考上界
  all-close (门全 0)   ：**关键判据**。修复成功时重建应大幅变差（all_close/base ≫ 1）；
                        旧退化实现里几乎不变（≈1）——这是区分"门有意义/门是摆设"的分水岭。

所有配置共用同一批 mask（mask 由 (seed, batch_idx) 决定），消除 mask 随机性，delta 才可比。
只读 ckpt，不重训，不需要 label。

用法
----
  python scripts/gate_ablation.py --config configs/smd.yaml \
         --entity machine-1-1 --ckpt runs/machine-1-1/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.masking import mixed_mask  # noqa: E402
from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.models.mask_predict import build_model  # noqa: E402
from src.training.losses import masked_l2_loss  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@torch.no_grad()
def eval_val_loss(model, loader, cfg, device, seed=0):
    """跑一遍 val，返回平均 masked 重建 loss。
    每个 batch 的 mask 由 (seed, batch_idx) 决定 → 同一 seed 下跨不同门配置完全可复现，
    这样不同门设置之间的 loss 差才只来自门、而非 mask 随机性。"""
    model.eval()
    total, n = 0.0, 0
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        torch.manual_seed(seed * 100003 + bi)  # 固定该 batch 的 mask
        mask = mixed_mask(
            batch.shape, cfg.mask.ratio, cfg.mask.span_prob,
            cfg.mask.span_min, cfg.mask.span_max,
            var_mask_prob=float(getattr(cfg.mask, "var_mask_prob", 0.0)),
            var_k_min=int(getattr(cfg.mask, "var_k_min", 1)),
            var_k_max=int(getattr(cfg.mask, "var_k_max", 3)),
            var_time_span=int(getattr(cfg.mask, "var_time_span", 0)),
            device=device,
        )
        pred = model(batch, mask)
        loss = masked_l2_loss(pred, batch, mask)
        total += loss.item() * batch.size(0)
        n += batch.size(0)
    return total / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0, help="固定 mask 的随机种子")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.entity:
        cfg.data.entity = args.entity
    if not bool(getattr(cfg.model, "var_gate", False)):
        print("[err] cfg.model.var_gate=false：该 ckpt 没有门，无法做对照。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds, _, _, _ = build_smd_datasets(cfg)
    bs = args.batch_size or cfg.train.batch_size
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing:
        # 老 ckpt（修复前的纯乘性门）没有 v_absent。置 0 → 仿射插值 v=g·v+(1-g)·0 = g·v
        # 精确退化回旧模型，可作为"修复前"对照（应看到 all_close/base≈1、deltas≈0）。
        if all("v_absent" in k for k in missing):
            zeroed = 0
            for name, p in model.named_parameters():
                if name in set(missing):
                    torch.nn.init.zeros_(p)
                    zeroed += 1
            print(f"[warn] ckpt 缺 {zeroed} 个 v_absent 键（修复前的退化门）；已置 0 还原旧模型。")
            print(f"[warn] 这是'修复前'对照；要验证修复效果，请用新代码重训后的 ckpt。\n")
        else:
            raise RuntimeError(f"非预期的缺失键: {missing}")
    if unexpected:
        print(f"[warn] 忽略 ckpt 中多余的键: {unexpected}")

    logit = model.var_gate.logit                      # (D,) 可学习门 logit
    g_trained = torch.sigmoid(logit).detach().cpu().numpy()
    D = logit.numel()
    orig = logit.data.clone()                         # 备份，每次评估后还原

    def with_logit(new_logit_data):
        logit.data.copy_(new_logit_data)
        l = eval_val_loss(model, val_loader, cfg, device, seed=args.seed)
        logit.data.copy_(orig)
        return l

    print(f"# entity={cfg.data.entity}  D={D}  val_windows={len(val_ds)}  seed={args.seed}")
    print(f"# 训练门统计: mean={g_trained.mean():.3f} min={g_trained.min():.3f} "
          f"max={g_trained.max():.3f} std={g_trained.std():.3f} "
          f"({'已分化' if g_trained.std() > 0.05 else '几乎没动'})")

    base = eval_val_loss(model, val_loader, cfg, device, seed=args.seed)  # 训练门
    all_open = with_logit(torch.full_like(orig, 30.0))    # 门全 1
    all_close = with_logit(torch.full_like(orig, -30.0))  # 门全 0
    ratio = all_close / max(base, 1e-12)

    print(f"\n[baseline 训练门] val_loss = {base:.5f}")
    print(f"[all-open 门全1 ] val_loss = {all_open:.5f}  (Δ={all_open-base:+.5f})")
    print(f"[all-close门全0 ] val_loss = {all_close:.5f}  (Δ={all_close-base:+.5f}, "
          f"all_close/base = {ratio:.2f}×)")
    print(f"\n# 关键判据：all-close/base 应 ≫ 1（门全关→无源信息→重建大幅变差）。")
    print(f"#   ≈1 → 退化没破除，门仍是摆设，回去查 v_absent 是否生效 / 是否被共享层补偿。")

    # ---- 逐变量：把 d 作为源关掉（其余维保持训练门），看 val_loss 涨多少 ----
    delta = np.full(D, np.nan)
    for d in range(D):
        ld = orig.clone()
        ld[d] = -30.0                                 # 仅把变量 d 的门压到 0
        delta[d] = with_logit(ld) - base

    order = np.argsort(delta)[::-1]                   # 降序：关掉后涨最多（最有用的源）在前
    print(f"\n# 逐变量源侧消融（固定 mask）：delta = loss(close d) - baseline")
    print(f"# delta 越大 → 变量 d 作为'源'越关键；delta≈0 → 作为源无信息 / 门已关")
    print(f"{'rank':>4} {'col':>4} {'gate':>7} {'delta':>11}  bar")
    print("-" * 48)
    dmax = max(1e-9, float(np.nanmax(np.abs(delta))))
    for rank, d in enumerate(order):
        d = int(d)
        bar = "█" * int(round(abs(delta[d]) / dmax * 20))
        print(f"{rank:>4} {d:>4} {g_trained[d]:>7.3f} {delta[d]:>+11.5f}  {bar}")

    # ---- 综合判据 ----
    useful = int((delta > 0.01 * base).sum())         # 关掉后 loss 涨 >1% 视为有用源
    print(f"\n# 关掉后 val_loss 涨 >1% 的变量数（有用源）: {useful}/{D}")
    if ratio < 1.05:
        print("# → ✗ 退化未破除：门全关几乎不影响重建。检查 v_absent 是否真的生效。")
    elif useful == 0:
        print("# → 门非退化（all-close 显著变差），但没有单个变量是关键源：信息高度冗余，"
              "关一个别人能补。看 delta 相对排序，或把 var_k_max 调大让 mask 更激进。")
    else:
        print(f"# → ✓ 门非退化且有分化：{useful} 个变量关掉显著伤重建，是真正被依赖的源。")


if __name__ == "__main__":
    main()
