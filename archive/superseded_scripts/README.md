# 归档脚本（superseded / 已得结论 / 死探索）

这些脚本已完成它们的使命或被取代，从 `scripts/` 移出以降低复杂度。结论都已沉淀进
`CLAUDE.md` 的「关键发现」与项目记忆，代码留作可追溯的参考。

> ⚠️ 重跑须知：这些脚本里多数 `from drift_sweep import ...`，依赖 `scripts/` 在
> sys.path 上。要再跑，把对应文件**拷回 `scripts/`** 即可（别在本目录直接跑）。

| 脚本 | 干什么 | 为什么归档 |
|---|---|---|
| `analyze_groups.py` | GDN 风格逐变量 top-K 相关邻居 → `data/groups/*.json` | 相关性先验在 SMD 无效（|corr|<0.1），产物从未接进模型 |
| `relation_drift.py` | per-timestep 误差向量的 Mahalanobis 关系漂移分 | 负面结论：与 drift 融合 α 恒为 0/1，SMD 是 magnitude 主导 |
| `fuse_sweep.py` | drift + relation 双路 late-fusion 扫描 | 同上；28 台复跑也证明 late-fusion 冗余（仅 5/28 真互补） |
| `loo_ablation.py` | Leave-One-Variable-Out + 随机零模型对照 | 负面结论：变量"无用性"不跨 entity 稳定，按列剪枝不值得 |
| `check_axis.py` | variable_rotation vs time_checkerboard 两轴打分对比 | 已定论用 variable_rotation；时间轴打分救不起弱 entity |
| `diagnose_scores.py` | 阈值无关诊断（ROC-AUC/AP、互相关 lag、onset/body） | 被 `scripts/diagnose_regime.py` 取代并扩展 |
| `threshold_sweep.py` | 早期阈值扫描（仅聚合 × POT） | 被 `scripts/drift_sweep.py` 取代（后者多扫时序后处理） |

## 仍可能复活的

- `relation_drift.py` 的"联合分布偏离"思路，和当前「打分层是瓶颈」的方向其实同源——
  若 `recon_calibrate.py`（逐变量校准）证明聚合层有救，可回头试 Mahalanobis 这类**联合**校准。
