# Mask-Predict 时序异常检测项目

## 项目目标

基于 **Mask-then-Predict** 范式，针对多变量时序数据（首要目标数据集：**SMD, Server Machine Dataset**）做无监督异常检测。

核心思想：训练时随机 mask 掉窗口内的部分 `(时间, 变量)` 位置（BERT 风格 MLM），让 Transformer encoder 仅从上下文重构这些位置；推理时用"棋盘式 mask"覆盖整窗，每个点的重构误差作为异常分数。

相比传统的整窗重构（AE / VAE），该范式能消除 self-attention 的 identity shortcut，对**点异常和短 spike** 更敏感。

## 运行环境

- **训练硬件**：服务器 2 × RTX 3090（24G × 2）
- **训练方式**：默认用 `torchrun --nproc_per_node=2` 启 DDP；同时启用 AMP（fp16）
- **本地**：Windows，仅做代码编写与调试，**不在本地跑训练**

## 项目结构

```
Mask-Predict/
├── CLAUDE.md                 # 本文件
├── README.md
├── requirements.txt
├── configs/smd.yaml          # 主配置（窗口长度、模型尺寸、训练超参）
├── data/                     # 原始数据存放处（SMD 需自行下载）
├── src/
│   ├── data/
│   │   ├── smd_dataset.py    # SMD 加载 + 归一化 + 滑窗
│   │   └── masking.py        # 训练 mask（point + span 混合）/ 推理棋盘 mask
│   ├── models/
│   │   ├── embedding.py      # 每变量独立嵌入 + 位置编码 + [MASK] token（VariableGate 已弃用，见下）
│   │   ├── transformer.py    # StagedEncoder（因果TCN→跨变量attn，当前默认）+ DualAxisEncoder（旧双路，留作 baseline）
│   │   └── mask_predict.py   # 主模型：Embedding → Encoder → 预测头
│   ├── training/
│   │   ├── trainer.py        # 训练循环、DDP、AMP、checkpoint
│   │   └── losses.py         # 仅对 mask 位置算 L2
│   ├── inference/
│   │   ├── scorer.py         # 棋盘 mask 推理 → 逐点分数
│   │   └── pot.py            # POT (Peaks-Over-Threshold) 自动阈值
│   ├── evaluation/
│   │   └── metrics.py        # F1, precision, recall, point-adjust
│   └── utils/{config.py, logger.py}
└── scripts/                 # 仅保留核心链路；一次性/已得结论的实验脚本见 archive/
    ├── train.py              # 单机训练入口（也兼容 DDP）；支持 --set k=v 覆盖配置
    ├── eval.py               # 推理 → 存 val/test_scores.npy → 单配置 POT 指标
    ├── train_ddp.sh          # 2×3090 启动脚本
    ├── drift_sweep.py        # 主评估：时序后处理×聚合×POT 扫最佳 raw_f1（共享库，被多脚本 import）
    ├── static_fuse.py        # 重构分 vs 静态|x_norm|幅度分 三路对比（raw + PA-F1）
    ├── run_static_fuse.sh    # 上者的批量 runner（默认 staged，全 28 台）
    ├── diagnose_regime.py    # 阈值无关诊断：recon_AUC vs static_AUC，判 regime / diffusion 前提
    └── recon_calibrate.py    # 逐变量 robust 校准 → 重定阈，验证“打分层是否瓶颈”
```

> **归档**：`archive/gate_experiment/`（源侧门，判定无收益）、`archive/superseded_scripts/`
> （relation/fusion/LOO/轴诊断/旧阈值扫描等，结论已沉淀，见各自 README）。要复用须搬回 `scripts/`。

## 数据约定

SMD 原始数据：<https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset>

放置方式：
```
data/SMD/
├── train/machine-1-1.txt        # 训练集（仅正常），38 列逗号分隔
├── test/machine-1-1.txt         # 测试集
└── test_label/machine-1-1.txt   # 测试集逐点标签（0/1）
```

每个 entity（machine-i-j）独立训练或联合训练，目前默认 **per-entity**，可通过 config 切换为 joint（加 entity embedding）。

## 关键设计决策

| 决策点 | 当前选择 | 备注 |
|---|---|---|
| 主干 | **StagedEncoder**：Stage1 per-variable 因果膨胀 TCN → Stage2 per-timestep 跨变量 attention | `encoder: staged`；旧的 DualAxisEncoder（双路交替）留作 baseline |
| Mask 策略 | **时间 cell mask 与变量整列 mask 各半**（`var_mask_prob: 0.5`）| 两种都学过，variable_rotation 与 time 打分都成立 |
| 推理 | **variable_rotation**：D 次前向，每次只 mask 一个变量 | 详见 `inference/scorer.py` |
| Loss | 仅对 mask 位置算 L2 | 不算未 mask 位置，避免 identity shortcut |
| **后处理** | **drift 后处理（per-variable z-score + smooth/runmax）** | **SMD 上均值 +0.125 raw_f1**；scripts/drift_sweep.py |
| 阈值 | POT (GPD 拟合) + 病态值过滤 + finite 检查 | 详见 `inference/pot.py` |
| 评估 | Raw F1（north star）+ point-adjust F1 都报 | PA 在 SMD 已失效：trivial 幅度基线 PA≈0.87≈SOTA，只对照标题用 |
| 归一化 | **RobustScaler**（1%/99% 分位 + min_range 守卫 + clip[-4,5]）| SMD 有近常数列，朴素 min-max 会被 val 微小波动除爆 |
| 窗口 | T=100, train stride=1，val stride=20，test stride=100 | |
| ~~源侧变量门~~ | **已弃用**（`var_gate: false`）| 严格消融判无收益，代码见 archive/gate_experiment；详见下 |
| early stop | val 重构 loss 连 `early_stop_patience` epoch 不降则停 | DDP 安全（val_loss all_reduce 同步） |

## 关键发现 / 经验教训

- **变量 mask 优于时间 mask**（反直觉但实验明确）：variable_rotation 用其他变量预测被 mask 的那个，对 SMD 这种"软相关"系统指标更有效
- **SMD 变量间几乎没有线性相关**：Spearman 邻居 |corr| 普遍 < 0.1。说明 Transformer 抓到的是非线性 / 时延 / 条件依赖。**这宣判了"基于相关性的先验分组"路线在 SMD 上无效**
- **POT 的脆弱性**：GPD 在分布病态时给出天文数字阈值；必须加 finite/范围检查 + 病态值过滤
- **drift 后处理是单步最大收益**（+0.125 均值）：模型早就抓到了信号，只是被点级噪音掩盖；不同 entity 偏好不同后处理（spike → runmax_3，持续 → smooth_10）
- **【负面结论】Mahalanobis 关系漂移无额外信号**：对 per-timestep 误差向量算 Mahalanobis 距离（`archive/superseded_scripts/relation_drift.py`），与 drift 融合时 α 永远落在 0 或 1、从不取中间 → SMD 异常是 **magnitude 主导**，不是"变量组合异常"主导。纯后处理路线到此撞墙
- **【负面结论】变量"无用性"不是稳定属性**：LOO 消融（`archive/superseded_scripts/loo_ablation.py`）逐个删变量看 F1。单 entity 内变量极不对称（贡献 +0.30 ~ -0.01），但跨 8 entity 的 noise 列**与随机指派零模型无异**。→ **按列剪枝的特征选择不值得做**
- **诊断脚本的零模型对照**：任何"跨 entity 一致性"统计都必须跟随机指派零模型比，否则绝对数会骗人（已归档的 loo_ablation 内置此对照）
- **【全 28 台总判定，2026-06】**：见项目记忆 `mask-predict-smd-full-verdict`。要点：
  - **结构不是杠杆**（dualaxis≈staged≈staged+容量，raw_f1 一条平线，已 4 次确认）；
  - **diffusion 也 NO-GO**：`diagnose_regime.py` 判 “#3 模型把异常重构掉” = **0/28**（异常点重构误差确实抬高）→ 没这个病，密度模型治不对症；
  - **真瓶颈在打分层**：阈值无关的 recon_AUC **明显赢** static_AUC 10/28、只输 2/28，但 raw_f1 被 `max over 38 变量无逐变量校准` 埋平（static 因被 RobustScaler 校准过、分布稳、阈值好定才"追平"）。→ 下一步打 `recon_calibrate.py`（逐变量校准重定阈），不是改模型

## 训练 / 评估流程

**训练**（服务器，2×3090）：
```bash
bash scripts/train_ddp.sh configs/smd.yaml machine-1-1
```

**评估**（单卡即可，两步）：
```bash
# 1) 跑推理生成 val_scores.npy / test_scores.npy
python scripts/eval.py --config configs/smd.yaml --entity machine-1-1 --ckpt runs/machine-1-1/best.pt

# 2) drift 后处理 + POT 扫描，输出 BEST raw_f1（推荐用这个看真实性能）
python scripts/drift_sweep.py --config configs/smd.yaml --entity machine-1-1
```

`drift_sweep.py` 是当前的主评估脚本（时序后处理×聚合×POT 扫最佳 raw_f1）。批量重构-vs-幅度对比用
`run_static_fuse.sh`；诊断 regime / 阈值层用 `diagnose_regime.py` 与 `recon_calibrate.py`。

## 开发规范（给 Claude 自己看的）

- **不要在本地跑训练**。本地只做静态检查、单元逻辑、形状/dtype 验证（可用极小 batch 跑一次前向）。
- 修改模型 / 数据接口后，务必检查 `scripts/train.py` 和 `scripts/eval.py` 仍能跑通（dry-run）。
- 新增超参一律走 `configs/*.yaml`，不要硬编码。
- DDP 相关：使用 `torchrun`；`DistributedSampler`；只在 rank 0 写 checkpoint 和日志。
- AMP：用 `torch.amp.autocast('cuda', dtype=torch.float16)` + `GradScaler`。
- 日志：训练用 stdout + 简单的 csv；不引入 wandb / tensorboard 依赖，保持轻量。
- 随机种子在配置里固定；DDP 下每个 rank 用 `seed + rank`。

## 源侧变量门（var_gate）— 已弃用

每变量一个可学习 sigmoid 门，控制"重构别人时该不该把这个变量当输入"，配 L1 稀疏惩罚。
踩过两个坑（乘性门的规范退化 → 改 v_absent 仿射插值；sigmoid 饱和冻结 → logit clamp），
但**最终严格消融判定在 SMD 无收益**（源侧门与目标侧 LOO 同样的负面结论）。

→ 现状 `var_gate: false`，相关代码仍在 `src/models/{embedding.py,transformer.py}`（gated 分支，
默认关），实验脚本与说明在 `archive/gate_experiment/`。**别再投入这条线**（见记忆
`mask-predict-channel-selection-null`）。配置里 `train.gate_*` 一组超参在门关闭时全为 no-op。

### 仍待验证的候选（按新颖性排序）

- **ELECTRA 式变量替换检测**：随机把某些 (t,d) 替换成同期其他变量/同变量其他时刻的值，加判别头预测"是否被替换"。TSAD 没人迁移过，判别头本身就是关系一致性分数。改模型 → 对所有 entity 有益
- **变量维 attention 分布漂移**：把 AnomalyTransformer 的 association discrepancy 从时间轴搬到变量轴，KL(测试时 attention ‖ val 正常 attention) 作关系漂移分数

## 其他待办

- [ ] 联合多 entity 训练 + entity embedding
- [ ] 预测头（next-step forecasting）与重构头融合
- [ ] 合成异常注入（AnomalyBERT 风格）
- [ ] 对其他数据集（MSL, SMAP, SWaT）的适配
- [x] ~~Graph attention 分支~~ → 已确认相关性先验在 SMD 上无效，放弃
