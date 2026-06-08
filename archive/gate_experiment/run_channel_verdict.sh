#!/usr/bin/env bash
# 确定性"通道选择是否提升异常检测"判定。
#
#   不训练学习门、零稀疏超参。流程：
#     1) 训练 gate-off 基线（var_gate=false）+ eval，对每个 entity 产出 (T,D) 重构误差分数
#     2) loo_ablation 跨 entity 判定：固定最佳配置逐通道从异常分里删，看 F1 怎么变，
#        并用零模型（随机指派同样数量 noise）对照——超出随机才算"通道选择真有用"
#
# 用法:
#   bash scripts/run_channel_verdict.sh                       # 默认 8 台代表性 entity
#   EPOCHS=40 bash scripts/run_channel_verdict.sh            # 想要最终数字就多训
#   FORCE=1   bash scripts/run_channel_verdict.sh            # 强制重训（默认已存在 last.pt 则跳过）
#   bash scripts/run_channel_verdict.sh machine-1-1 machine-2-3   # 指定 entity
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_ENTITIES=(machine-1-1 machine-1-5 machine-2-1 machine-2-4 machine-2-8 machine-3-2 machine-3-7 machine-3-11)
if [ "$#" -gt 0 ]; then ENTITIES=("$@"); else ENTITIES=("${DEFAULT_ENTITIES[@]}"); fi

CFG=configs/smd_abl_nogate.yaml     # gate-off 基线（干净，无门混淆）
DIR=runs_abl/nogate
NPROC=${NPROC:-2}
EPOCHS=${EPOCHS:-5}                 # 判定 relative 通道贡献，对训练长度不敏感；想要终值用 40
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "=== 通道选择判定：gate-off 基线 ==="
echo "entities (${#ENTITIES[@]}): ${ENTITIES[*]}   epochs=$EPOCHS   FORCE=${FORCE:-0}"
echo

for ent in "${ENTITIES[@]}"; do
  ckpt="${DIR}/${ent}/last.pt"
  echo "---- baseline entity=$ent ----"
  if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip-train] 已存在 $ckpt（FORCE=1 可强制重训）"
  else
    torchrun --standalone --nproc_per_node="${NPROC}" \
      scripts/train.py --config "$CFG" --entity "$ent" --set train.epochs="$EPOCHS"
  fi
  python scripts/eval.py --config "$CFG" --entity "$ent" --ckpt "$ckpt"
  echo
done

ENT_CSV=$(IFS=,; echo "${ENTITIES[*]}")
echo "=== loo_ablation 判定（目标侧通道选择 + 零模型对照）==="
python scripts/loo_ablation.py --config "$CFG" --entity "$ENT_CSV"
