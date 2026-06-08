# 源侧变量门控实验（已归档 · 结论：无效）

时间：2026-06

## 假设
并非所有变量维度都对异常检测有贡献；用一个可学习的源侧门（每变量一个 gate，作用在
变量维 attention 的 value 上）无监督地关掉无用通道，应能增强异常检测。

## 过程踩的坑（都已解决，但暴露了机制本身难驯）
1. 纯乘性门 `v=g·v` → 均匀塌缩到 0（规范退化：均匀缩放被共享 out/qkv 线性层补偿）。
   修复：仿射插值 `v=g·v+(1-g)·v_absent`（v_absent=学习的“缺席”常量）。
2. 修复后门又饱和顶到 1.0 冻住（sigmoid 两端梯度消失 + L1 hold 期给了重建无约束窗口）。
   修复：logit 投影到 [-4,4] 防饱和 + L1 从头施压。
3. 始终难以稳定停在“部分开/部分关”的窄带，需要盲扫 L1。

## 判定（决定性）
用 `loo_ablation.py`（目标侧逐通道删 + 零模型对照）跨 8 台 entity：
- oracle（用 test label 的上界）均值 ΔF1 仅 +0.022，去最大 outlier 后 +0.015。
- 零模型对照：robust noise 通道 observed=23 vs 随机期望 23.3，**超出≈0** → “哪些通道无用”
  跨 entity 与随机指派无异；每台标 16~31/38 通道为 noise、从不是 noise 的 0 列。
- **结论：不存在稳定可泛化的“无用通道”结构，门控无效，弃。** 这也解释了门为何难驯——
  没有稳定目标可学。

## 文件（归档快照，原在 configs/ 与 scripts/；如需重跑需修内部相对路径）
- smd_abl_gate.yaml / smd_abl_nogate.yaml：两臂配置（var_gate 开/关）
- gate_ablation.py：固定 mask 下逐变量源侧消融（重建层面，验证门是否非退化）
- run_gate_ablation.sh / compare_gate_ablation.py：两臂 train→eval→对比+判定
- run_channel_verdict.sh：gate-off 基线 + loo 判定（产出上面的结论）

## 真正的 headroom（下一步方向）
base_f1 跨 entity 从 0.018(machine-3-2) 到 0.978(machine-2-8)，差 50×。
力气应放在“为什么模型在某些 entity 上几乎检测不到异常”，疑似 variable_rotation（跨变量
重构）对时序型/单变量型异常盲 → 需要融合时间维打分（见主仓 fuse / both 模式）。
