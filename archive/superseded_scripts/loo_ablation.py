"""Leave-One-Variable-Out 消融：用因果方式检验"哪些变量无关紧要"。

动机
----
项目里一直把 38 个变量当成对称、等权的通道。但有两个先验：
  (1) 变量间存在（非线性）关系；
  (2) 有些变量是常数 / 纯噪声 / 死端口，对异常检测无贡献甚至有害。
"读 attention 判断变量没用"不可靠（softmax 相对化 + attention≠重要性）。
本脚本不靠启发式，直接做**可证伪**的消融：把变量 d 从聚合里剔除，看 F1 怎么变。

方法（零重训，只在已保存的 val/test_scores.npy 上做）
----
1. baseline：用全 38 个变量跑 drift_sweep 的完整搜索（mode×agg×q×level），
   拿到 baseline_f1 和最佳配置 (mode*, agg*, q*, level*)。
2. 单变量 LOO：**固定** baseline 的最佳配置，逐个把列 d 从聚合中删掉，
   重新拟合 POT、评估，得到 f1_without_d。
   contribution[d] = baseline_f1 - f1_without_d
     - contribution > 0  ：删了 F1 下降  → 该变量**有信息**
     - contribution <= 0 ：删了 F1 不降反升 → 该变量**无关/有害**（noise 候选）
   固定配置是为了"隔离单个变量的影响"，避免删列后换个配置夺冠造成的混淆。
3. 批量剔除：把所有 noise 候选一起删掉，对剩余变量子集**重新完整 sweep**
   （给子集它自己的最佳配置，公平比较天花板），看 F1 是否进一步上升。

判定
----
  - 批量剔除后 F1 明显上升 → 先验成立，且我们拿到了具体哪些列是无关的。
  - 几乎所有 contribution > 0 / 批量剔除后不升反降 → SMD 上变量都各有边际贡献，
    "无关变量"先验弱，gate / 稀疏 attention 不值得做。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_scores  # noqa: E402
from src.inference.pot import pot_threshold  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# 直接复用 drift_sweep 的后处理 / 聚合 / 搜索逻辑，保持口径一致
from drift_sweep import postprocess, aggregate, search_one  # noqa: E402


MODES = [
    "raw", "zscore",
    "smooth_3", "smooth_5", "smooth_10",
    "runmax_3", "runmax_5", "runmax_10",
    "combine_smooth_5", "combine_smooth_10",
]
QS = [0.95, 0.99]
LEVELS = [1e-3, 1e-4, 1e-5, 1e-6]


def full_sweep_subset(val: np.ndarray, test: np.ndarray, label: np.ndarray,
                      keep_cols: list) -> dict:
    """对给定变量子集跑完整 mode×agg×q×level 搜索，返回最佳配置 + f1。"""
    v_sub = val[:, keep_cols]
    t_sub = test[:, keep_cols]
    best = None
    for mode in MODES:
        try:
            v_pp, t_pp = postprocess(v_sub, t_sub, mode)
        except Exception:
            continue
        b = search_one(v_pp, t_pp, label)  # 内部扫 agg×q×level
        if b is None:
            continue
        if best is None or b["raw_f1"] > best["raw_f1"]:
            best = {**b, "mode": mode}
    return best


def eval_fixed(val: np.ndarray, test: np.ndarray, label: np.ndarray,
               keep_cols: list, mode: str, agg: str, q: float, level: float):
    """固定后处理配置，只对 keep_cols 聚合 → 拟合 POT → 评估。返回 raw_f1（失败给 None）。"""
    try:
        v_pp, t_pp = postprocess(val[:, keep_cols], test[:, keep_cols], mode)
        v_s = aggregate(v_pp, agg)
        t_s = aggregate(t_pp, agg)
        thr = pot_threshold(v_s, q=q, level=level)
        r = evaluate_scores(t_s, label, thr)
        return r["raw_f1"]
    except Exception:
        return None


def run_entity(cfg, entity: str, drop_thresh: float):
    run_dir = Path(cfg.train.save_dir) / entity
    val = np.load(run_dir / "val_scores.npy")     # (T_val, D)
    test = np.load(run_dir / "test_scores.npy")   # (T_test, D)
    label = np.loadtxt(Path(cfg.data.root) / "test_label" / f"{entity}.txt",
                       dtype=np.int64)
    T, D = val.shape
    all_cols = list(range(D))

    print(f"\n{'='*72}")
    print(f"# entity={entity}  D={D}  val_T={T}  test_T={test.shape[0]}")
    print(f"{'='*72}")

    # ---- 1) baseline：全变量完整 sweep ----
    base = full_sweep_subset(val, test, label, all_cols)
    base_f1 = base["raw_f1"]
    mode, agg, q, level = base["mode"], base["agg"], base["q"], base["level"]
    print(f"[baseline] raw_f1={base_f1:.4f}  mode={mode}  agg={agg}  "
          f"q={q}  level={level:.0e}")

    # 一致性自检：固定配置全变量应复现 baseline
    chk = eval_fixed(val, test, label, all_cols, mode, agg, q, level)
    if chk is not None and abs(chk - base_f1) > 1e-6:
        print(f"  [warn] fixed-config full-set f1={chk:.4f} != baseline "
              f"(agg/POT 在子集上行为略有差异，属正常)")

    # ---- 2) 单变量 LOO（固定 baseline 配置）----
    contrib = np.full(D, np.nan)
    for d in range(D):
        keep = [c for c in all_cols if c != d]
        f1_wo = eval_fixed(val, test, label, keep, mode, agg, q, level)
        if f1_wo is not None:
            contrib[d] = base_f1 - f1_wo

    order = np.argsort(contrib)  # 升序：最负（删了最有益）在前
    print(f"\n# 单变量 LOO（固定 mode={mode} agg={agg} q={q} level={level:.0e}）")
    print(f"# contribution = baseline_f1 - f1_without_d  "
          f"(>0 有信息, <=0 noise候选)")
    print(f"{'rank':>4} {'col':>4} {'f1_without':>11} {'contribution':>13}  tag")
    print("-" * 50)
    for rank, d in enumerate(order):
        if np.isnan(contrib[d]):
            print(f"{rank:>4} {d:>4} {'(POT fail)':>11} {'—':>13}  skip")
            continue
        f1_wo = base_f1 - contrib[d]
        tag = "noise" if contrib[d] <= drop_thresh else "info"
        star = "  ←" if contrib[d] < -1e-6 else ""
        print(f"{rank:>4} {d:>4} {f1_wo:>11.4f} {contrib[d]:>+13.4f}  {tag}{star}")

    # ---- 3) 批量剔除 noise 候选 → 子集重新完整 sweep ----
    noise_cols = [d for d in all_cols
                  if not np.isnan(contrib[d]) and contrib[d] <= drop_thresh]
    info_cols = [d for d in all_cols if d not in noise_cols]
    print(f"\n# noise 候选 (contribution <= {drop_thresh:+.3f}): "
          f"{noise_cols}  (共 {len(noise_cols)}/{D})")

    if 0 < len(noise_cols) < D:
        pruned = full_sweep_subset(val, test, label, info_cols)
        delta = pruned["raw_f1"] - base_f1
        verdict = "先验成立 ✓" if delta > 1e-3 else (
            "无明显收益" if abs(delta) <= 1e-3 else "剔除反而变差 ✗")
        print(f"[pruned]   raw_f1={pruned['raw_f1']:.4f}  "
              f"mode={pruned['mode']}  agg={pruned['agg']}  "
              f"q={pruned['q']}  level={pruned['level']:.0e}")
        print(f"[verdict]  Δf1 = {delta:+.4f}   →   {verdict}")
        return {"entity": entity, "base_f1": base_f1,
                "pruned_f1": pruned["raw_f1"], "delta": delta,
                "n_noise": len(noise_cols), "noise_cols": noise_cols}
    else:
        print("[verdict]  无 noise 候选（或全部判为 noise），先验弱 / 无法剔除")
        return {"entity": entity, "base_f1": base_f1,
                "pruned_f1": base_f1, "delta": 0.0,
                "n_noise": len(noise_cols), "noise_cols": noise_cols}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", required=True,
                        help="单个 entity，或逗号分隔多个，如 machine-1-1,machine-1-2")
    parser.add_argument("--drop_thresh", type=float, default=0.0,
                        help="contribution <= 此值视为 noise 候选（默认 0，可设 -0.005 更保守）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    entities = [e.strip() for e in args.entity.split(",") if e.strip()]

    summary = []
    for ent in entities:
        try:
            summary.append(run_entity(cfg, ent, args.drop_thresh))
        except FileNotFoundError as e:
            print(f"\n[skip] {ent}: {e}")

    if len(summary) > 1:
        print(f"\n{'='*72}")
        print("# 汇总")
        print(f"{'entity':<16} {'base_f1':>8} {'pruned_f1':>10} "
              f"{'Δf1':>8} {'#noise':>7}")
        print("-" * 56)
        for s in summary:
            print(f"{s['entity']:<16} {s['base_f1']:>8.4f} "
                  f"{s['pruned_f1']:>10.4f} {s['delta']:>+8.4f} {s['n_noise']:>7}")
        deltas = np.array([s["delta"] for s in summary])
        print("-" * 56)
        print(f"{'MEAN':<16} {np.mean([s['base_f1'] for s in summary]):>8.4f} "
              f"{np.mean([s['pruned_f1'] for s in summary]):>10.4f} "
              f"{deltas.mean():>+8.4f}")
        n_pos = int((deltas > 1e-3).sum())
        print(f"\n# {n_pos}/{len(summary)} 个 entity 剔除 noise 后 F1 上升 "
              f"(>+0.001)")

        # ---- 跨 entity noise 列一致性（决定性判据）----
        # SMD 每台机器是同样 38 个指标，列号语义跨 entity 通用。
        # noise 列若稳定复现 → 真无用指标；若各 entity 各异 → F1 曲面噪声，先验死。
        from collections import Counter
        D = int(cfg.data.n_features)
        n_ent = len(summary)
        cnt = Counter()
        for s in summary:
            cnt.update(s["noise_cols"])
        robust = [c for c in range(D) if cnt.get(c, 0) >= max(2, (n_ent + 1) // 2)]
        never = [c for c in range(D) if cnt.get(c, 0) == 0]
        once = [c for c in range(D) if cnt.get(c, 0) == 1]

        print(f"\n# 跨 entity noise 列一致性（列号 : 在几个/{n_ent} entity 中被判 noise）")
        for col, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            bar = "█" * c
            flag = "  ★robust" if c >= max(2, (n_ent + 1) // 2) else ""
            print(f"  col {col:>2}: {c}/{n_ent}  {bar}{flag}")
        print(f"\n# robust noise（≥半数 entity）: {robust}  (共 {len(robust)})")
        print(f"# 从不是 noise（始终有信息）: {len(never)} 列")
        print(f"# 只在单个 entity 是 noise（疑似曲面噪声）: {len(once)} 列")

        # ---- 关键：跟"随机指派"零模型对照 ----
        # 各 entity 声明的 noise 数量差异巨大（坏模型会乱标一大堆），绝对数会骗人。
        # 真正的问题：这种"一致性"是否只是随机指派同样数量 noise 的产物？
        thresh = max(2, (n_ent + 1) // 2)
        counts_per_entity = [len(s["noise_cols"]) for s in summary]
        rng = np.random.default_rng(0)
        N = 3000
        nr = np.zeros(N); nn = np.zeros(N)
        for i in range(N):
            tally = np.zeros(D, dtype=int)
            for k in counts_per_entity:
                tally[rng.choice(D, size=min(k, D), replace=False)] += 1
            nr[i] = int((tally >= thresh).sum())
            nn[i] = int((tally == 0).sum())
        exp_robust, exp_never = nr.mean(), nn.mean()
        excess = len(robust) - exp_robust
        print(f"\n# 零模型对照（随机指派同样数量 noise，{N} 次）：")
        print(f"#   robust(≥{thresh})  observed={len(robust)}  null期望={exp_robust:.1f}"
              f"  →  超出零模型 {excess:+.1f} 列")
        print(f"#   never-noise      observed={len(never)}  null期望={exp_never:.1f}")

        # ---- 综合判定（基于"超出零模型"，不再看绝对数）----
        deltas_no_outlier = sorted(deltas)[:-1]
        mean_robust = float(np.mean(deltas_no_outlier)) if deltas_no_outlier else 0.0
        print(f"\n# 均值 Δf1={deltas.mean():+.4f}  去掉最大 outlier 后={mean_robust:+.4f}"
              f"  (oracle 上界，用了 test label)")
        if excess > 0.15 * D and mean_robust > 0.005:
            print("# → 先验成立：noise 列显著超出随机，存在稳定无用指标，值得上 gate / 剪枝")
        elif excess > 0.05 * D:
            print("# → 先验弱：略超随机，剪枝收益有限，gate 性价比存疑")
        else:
            print("# → 先验不成立：'一致性'与随机指派无异（超出≈0）。"
                  "变量无用性不是稳定属性，不要上 gate；问题在模型本身（看低 baseline entity）")


if __name__ == "__main__":
    main()
