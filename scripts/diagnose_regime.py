"""Regime 诊断 + diffusion 前提验证（纯分数级，无需重跑模型 / 无需 GPU）。

对每个 entity：读已存的 recon 分数 (runs_fuse/<ent>/test_scores.npy, (T,D) 重构误差)
与 test_label，现算 static=|x_norm| (T,D)。两者各 max over 变量维 → 1D，算两条**阈值无关**指标：

  - AUC(anomaly vs normal)：该分单独的可分性（0.5=瞎猜，1=完美）。
  - lift：异常点中位分比正常点高几个 σ（直觉用）。

判读（核心）：
  recon_AUC >> static_AUC               → 关系型异常：重构看得见、幅度看不见 → **你方法的卖点**
  static_AUC >> recon_AUC 且 recon_AUC≈0.5 → 模型把异常也重构好了(#3 太会补)，只有幅度管用
                                          → **diffusion(密度模型)前提成立，值得上**
  两者都≈0.5                            → 该分都抓不到（持续段/隐蔽）→ 换 diffusion 也难救

同时对 --plot 指定的 entity 出 PNG：recon_1d / static_1d（各自 z-score）叠 label 红色阴影，
直观看“异常段里幅度平平、重构误差爆没爆”。

用法：
  python scripts/diagnose_regime.py --config configs/smd.yaml --recon_dir runs_fuse
  python scripts/diagnose_regime.py --config configs/smd.yaml --recon_dir runs_fuse \
         --entities machine-2-1 machine-1-3 --plot machine-2-1 machine-1-3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.smd_dataset import build_smd_datasets  # noqa: E402
from src.utils.config import load_config  # noqa: E402

ALL28 = [f"machine-1-{i}" for i in range(1, 9)] + \
        [f"machine-2-{i}" for i in range(1, 10)] + \
        [f"machine-3-{i}" for i in range(1, 12)]
# 默认重点画：两台 recon 赢(卖点) + 两台 static 赢(#3 测试)
DEFAULT_PLOT = ["machine-2-1", "machine-3-7", "machine-1-3", "machine-1-4"]


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列），用于无偏 AUC。"""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc(score: np.ndarray, lab: np.ndarray) -> float:
    pos = lab == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _avg_ranks(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def lift(score: np.ndarray, lab: np.ndarray) -> float:
    pos, neg = score[lab == 1], score[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((np.median(pos) - np.median(neg)) / max(neg.std(), 1e-8))


def static_2d(dataset, cfg) -> np.ndarray:
    """从数据现算 |x_norm| (T,D)，与 score_series 同样的拼接顺序(shuffle=False)。"""
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    a = np.concatenate([b.abs().numpy() for b in loader], axis=0)
    return a.reshape(-1, a.shape[-1])


def verdict(rec_auc: float, sta_auc: float) -> str:
    if np.isnan(rec_auc) or np.isnan(sta_auc):
        return "无异常/无法判定"
    if rec_auc - sta_auc > 0.03:
        return "关系型 → 你方法卖点"
    if sta_auc - rec_auc > 0.03 and rec_auc < 0.60:
        return "#3 重构抹平 → diffusion 前提成立"
    if max(rec_auc, sta_auc) < 0.60:
        return "都抓不到(持续/隐蔽) → diffusion 也难救"
    return "两者接近"


def make_plot(ent, recon, static, lab, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot-skip] {ent}: matplotlib 不可用 ({e})")
        return
    def z(v):
        return (v - v.mean()) / max(v.std(), 1e-8)
    rz, sz = z(recon), z(static)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 5))
    for ax, s, name in [(axes[0], rz, "recon error (z, max over var)"),
                        (axes[1], sz, "|x_norm| (z, max over var)")]:
        ax.plot(s, lw=0.6, color="steelblue")
        ax.fill_between(np.arange(len(lab)), s.min(), s.max(),
                        where=lab == 1, color="red", alpha=0.15, step="mid")
        ax.set_ylabel(name, fontsize=9)
        ax.margins(x=0)
    axes[0].set_title(f"{ent}   recon_AUC={auc(recon, lab):.3f}  static_AUC={auc(static, lab):.3f}",
                      fontsize=11)
    axes[1].set_xlabel("time")
    fig.tight_layout()
    p = out_dir / f"{ent}_regime.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"[plot] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recon_dir", default="runs_fuse", help="recon 分数目录 <dir>/<ent>/test_scores.npy")
    ap.add_argument("--entities", nargs="+", default=ALL28)
    ap.add_argument("--plot", nargs="+", default=DEFAULT_PLOT, help="对这些 entity 出 PNG；'none' 关闭")
    ap.add_argument("--out", default="regime_diag", help="PNG 输出目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_set = set() if args.plot == ["none"] else set(args.plot)

    rows = []
    for ent in args.entities:
        rd = Path(args.recon_dir) / ent
        if not (rd / "test_scores.npy").exists():
            print(f"[skip] {ent}: 缺 {rd/'test_scores.npy'}")
            continue
        cfg.data.entity = ent
        rec2 = np.load(rd / "test_scores.npy")                 # (T,D) 重构误差
        _, _, test_ds, test_label, _ = build_smd_datasets(cfg)
        sta2 = static_2d(test_ds, cfg)                          # (T,D) |x_norm|
        T = min(len(rec2), len(sta2), len(test_label))
        recon = rec2[:T].max(1)
        static = sta2[:T].max(1)
        lab = np.asarray(test_label[:T], dtype=np.int64)

        r_auc, s_auc = auc(recon, lab), auc(static, lab)
        rows.append({"ent": ent, "r_auc": r_auc, "s_auc": s_auc,
                     "r_lift": lift(recon, lab), "s_lift": lift(static, lab),
                     "anom%": 100.0 * lab.mean(), "verdict": verdict(r_auc, s_auc)})
        if ent in plot_set:
            make_plot(ent, recon, static, lab, out_dir)

    if not rows:
        print("无可评估 entity（先跑 run_static_fuse.sh 生成 test_scores.npy）。")
        return

    print(f"\n{'entity':<14} {'recon_AUC':>9} {'static_AUC':>10} {'ΔAUC':>7} "
          f"{'r_lift':>7} {'s_lift':>7} {'anom%':>6}  判读")
    print("-" * 92)
    for r in rows:
        print(f"{r['ent']:<14} {r['r_auc']:>9.3f} {r['s_auc']:>10.3f} "
              f"{r['r_auc']-r['s_auc']:>+7.3f} {r['r_lift']:>7.2f} {r['s_lift']:>7.2f} "
              f"{r['anom%']:>5.1f}%  {r['verdict']}")

    # 汇总计数
    sell = sum(1 for r in rows if "卖点" in r["verdict"])
    diff = sum(1 for r in rows if "diffusion 前提成立" in r["verdict"])
    dead = sum(1 for r in rows if "难救" in r["verdict"])
    n = len(rows)
    print("-" * 92)
    print(f"\n# 关系型(你方法卖点)        : {sell}/{n}")
    print(f"# #3 重构抹平(diffusion 值得上): {diff}/{n}")
    print(f"# 都抓不到(diffusion 也难救)  : {dead}/{n}")
    print(f"\n判读 diffusion 该不该上：看上面第 2 行。如果 diffusion-前提-成立的台数多，"
          f"说明‘模型把异常补掉了’是真普遍现象 → 值得砸 diffusion；"
          f"\n如果多数是‘卖点’或‘难救’，diffusion 救不了主要矛盾，先别上。")


if __name__ == "__main__":
    main()
