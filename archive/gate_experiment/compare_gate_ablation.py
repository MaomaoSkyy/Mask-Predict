"""门控消融对比 + 判定。

对每个 entity，在已保存的 OFF / GATE 分数上跑**同一套** drift_sweep 搜索取最优 raw_f1，
并在 OFF 分数上算 ORACLE 天花板（用 label 的目标侧通道选择上界，复用 loo_ablation 逻辑）：

  OFF    : var_gate=false 基线（不做任何源选择）
  GATE   : var_gate=true  你的机制（无监督学"该关哪些源"）
  ORACLE : 在 OFF 分数上、用 label 贪心删通道能达到的 F1 上界（"有没有空间可选"的参照）

判定（预先登记，跨 entity 配对）：
  - ORACLE 相比 OFF 几乎无提升        → 通道选择本就没空间 → **关闭机制**，进下一个实验
  - ORACLE≫OFF 但 GATE≈OFF           → 有空间但门没抓住（重建目标≠检测目标）→ **当前形态不保留/需重做**
  - GATE 显著 > OFF（配对一致）       → **保留机制**

裁判是 raw_f1（检测），不是重建 loss。eval 须用 last.pt（见 run_gate_ablation.sh）。
只读已保存分数，不重训、需要 test_label。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # 便于 import 同目录的 drift_sweep / loo_ablation

from src.utils.config import load_config  # noqa: E402
from loo_ablation import full_sweep_subset, eval_fixed  # noqa: E402

# 判定阈值（F1 绝对值）
HEADROOM_EPS = 0.01   # ORACLE-OFF 小于此 → 视为"没空间"
WIN_EPS = 0.005       # |ΔF1| 小于此 → 视为平局
KEEP_MARGIN = 0.01    # 中位 ΔF1 大于此且配对一致 → 保留机制


def oracle_ceiling(val, test, label, all_cols):
    """在给定（OFF）分数上的目标侧通道选择上界：固定 baseline 最佳配置做单变量 LOO，
    删掉所有 contribution<=0 的通道后对子集重新完整 sweep。返回 (off_f1, oracle_f1, n_drop)。"""
    base = full_sweep_subset(val, test, label, all_cols)
    if base is None:
        return None, None, 0
    off_f1 = base["raw_f1"]
    mode, agg, q, level = base["mode"], base["agg"], base["q"], base["level"]
    contrib = np.full(len(all_cols), np.nan)
    for d in all_cols:
        keep = [c for c in all_cols if c != d]
        f1_wo = eval_fixed(val, test, label, keep, mode, agg, q, level)
        if f1_wo is not None:
            contrib[d] = off_f1 - f1_wo
    noise = [d for d in all_cols if not np.isnan(contrib[d]) and contrib[d] <= 0]
    info = [d for d in all_cols if d not in noise]
    if 0 < len(noise) < len(all_cols):
        pruned = full_sweep_subset(val, test, label, info)
        oracle_f1 = max(off_f1, pruned["raw_f1"]) if pruned else off_f1
    else:
        oracle_f1 = off_f1
    return off_f1, oracle_f1, len(noise)


def load_scores(save_dir, entity):
    d = Path(save_dir) / entity
    v, t = d / "val_scores.npy", d / "test_scores.npy"
    if not (v.exists() and t.exists()):
        return None, None
    return np.load(v), np.load(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", nargs="+", required=True)
    ap.add_argument("--gate_config", default="configs/smd_abl_gate.yaml")
    ap.add_argument("--nogate_config", default="configs/smd_abl_nogate.yaml")
    args = ap.parse_args()

    cfg_g = load_config(args.gate_config)
    cfg_n = load_config(args.nogate_config)
    dir_g, dir_n = cfg_g.train.save_dir, cfg_n.train.save_dir
    label_root = Path(cfg_n.data.root) / "test_label"

    rows = []
    for ent in args.entities:
        vg, tg = load_scores(dir_g, ent)
        vn, tn = load_scores(dir_n, ent)
        if vn is None or vg is None:
            print(f"[skip] {ent}: 缺分数（先跑 run_gate_ablation.sh）")
            continue
        label = np.loadtxt(label_root / f"{ent}.txt", dtype=np.int64)
        D = vn.shape[1]
        all_cols = list(range(D))

        off_f1, oracle_f1, n_drop = oracle_ceiling(vn, tn, label, all_cols)
        gate_best = full_sweep_subset(vg, tg, label, all_cols)
        gate_f1 = gate_best["raw_f1"] if gate_best else float("nan")
        rows.append({
            "entity": ent, "off": off_f1, "gate": gate_f1, "oracle": oracle_f1,
            "delta": gate_f1 - off_f1, "headroom": oracle_f1 - off_f1, "n_drop": n_drop,
        })

    if not rows:
        print("没有可对比的 entity。")
        return

    # ---- 明细表 ----
    print(f"\n{'entity':<16} {'OFF':>7} {'GATE':>7} {'ΔF1':>8} {'ORACLE':>7} "
          f"{'headroom':>9} {'#drop':>6}")
    print("-" * 64)
    for r in rows:
        flag = "  ✓" if r["delta"] > WIN_EPS else ("  ✗" if r["delta"] < -WIN_EPS else "")
        print(f"{r['entity']:<16} {r['off']:>7.3f} {r['gate']:>7.3f} "
              f"{r['delta']:>+8.3f} {r['oracle']:>7.3f} {r['headroom']:>+9.3f} "
              f"{r['n_drop']:>6}{flag}")

    deltas = np.array([r["delta"] for r in rows])
    heads = np.array([r["headroom"] for r in rows])
    n = len(rows)
    wins = int((deltas > WIN_EPS).sum())
    losses = int((deltas < -WIN_EPS).sum())
    ties = n - wins - losses
    print("-" * 64)
    print(f"{'MEAN':<16} {np.mean([r['off'] for r in rows]):>7.3f} "
          f"{np.mean([r['gate'] for r in rows]):>7.3f} {deltas.mean():>+8.3f} "
          f"{np.mean([r['oracle'] for r in rows]):>7.3f} {heads.mean():>+9.3f}")
    print(f"\nΔF1: median={np.median(deltas):+.3f} mean={deltas.mean():+.3f}  "
          f"wins/losses/ties = {wins}/{losses}/{ties}  (|Δ|>{WIN_EPS} 算胜负)")
    print(f"headroom(ORACLE-OFF): median={np.median(heads):+.3f} mean={heads.mean():+.3f}")

    # 配对检验（有 scipy 用 Wilcoxon，否则退化为符号检验）
    pval = None
    try:
        from scipy.stats import wilcoxon
        nz = deltas[np.abs(deltas) > 1e-9]
        if len(nz) >= 1 and not np.allclose(nz, nz[0]):
            pval = wilcoxon(nz).pvalue
            print(f"Wilcoxon signed-rank p = {pval:.4f}  (n={len(nz)})")
    except Exception:
        from math import comb
        k = wins
        m = wins + losses
        if m > 0:
            pval = sum(comb(m, i) for i in range(k, m + 1)) / 2 ** m  # 单边符号检验
            print(f"符号检验 p(单边) = {pval:.4f}  (wins={wins}/{m})  [无 scipy，退化方案]")

    # ---- 预先登记的判定 ----
    print("\n=== 判定 ===")
    med_head = float(np.median(heads))
    med_delta = float(np.median(deltas))
    sig = (pval is not None and pval < 0.1)
    if med_head < HEADROOM_EPS:
        print(f"# ORACLE 中位仅比 OFF 高 {med_head:+.3f}(<{HEADROOM_EPS})：通道选择本就没空间。")
        print("# → 关闭门控机制，进下一个实验。这不是门的错——这数据没得选。")
    elif med_delta > KEEP_MARGIN and wins > losses and sig:
        print(f"# GATE 中位比 OFF 高 {med_delta:+.3f}(>{KEEP_MARGIN})，{wins}/{n} 胜，配对显著。")
        print("# → 保留门控机制。可附带说明：门无监督地逼近了 oracle 上界。")
    elif med_head >= HEADROOM_EPS and abs(med_delta) <= KEEP_MARGIN:
        print(f"# 有空间(headroom={med_head:+.3f})但 GATE≈OFF(Δ={med_delta:+.3f})：")
        print("# → 门没抓住可选信号（重建目标≠检测目标）。当前形态不保留；要么重做门的训练目标，")
        print("#   要么改走目标侧打分选择（loo 那根轴）。")
    else:
        print(f"# 结论不明确：headroom={med_head:+.3f} ΔF1={med_delta:+.3f} "
              f"wins/losses={wins}/{losses} p={pval}。")
        print("# → 建议扩大 entity 数 / 多 seed 复跑，或检查个别 entity 的异常形态。")


if __name__ == "__main__":
    main()
