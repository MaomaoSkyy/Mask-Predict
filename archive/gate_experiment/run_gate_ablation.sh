#!/usr/bin/env bash
# 门控消融 orchestrator：对每个 entity 跑两臂（GATE / NOGATE）的 train → eval，
# 产出各臂的 val_scores/test_scores 到独立 save_dir，供 compare_gate_ablation.py 对比。
#
# 用法:
#   bash scripts/run_gate_ablation.sh                      # 默认 8 台代表性 entity
#   bash scripts/run_gate_ablation.sh machine-1-1 machine-2-3   # 指定 entity
#   FORCE=1 bash scripts/run_gate_ablation.sh              # 强制重训（默认已存在 last.pt 则跳过 train）
#
# 关键设计:
#   - 两臂唯一差别是 model.var_gate（配置见 configs/smd_abl_{gate,nogate}.yaml），其余逐字一致、同 seed。
#   - eval 用 last.pt（门训练最充分的时刻；best.pt 按重建选 = 门最开，会喂瘪门臂）。
set -euo pipefail
cd "$(dirname "$0")/.."

# 默认 8 台代表性 entity，横跨 machine-1/2/3 三组
DEFAULT_ENTITIES=(machine-1-1 machine-1-5 machine-2-1 machine-2-4 machine-2-8 machine-3-2 machine-3-7 machine-3-11)
if [ "$#" -gt 0 ]; then
  ENTITIES=("$@")
else
  ENTITIES=("${DEFAULT_ENTITIES[@]}")
fi

# 臂: 名称 -> (config, save_dir)
ARM_NAMES=(nogate gate)
declare -A ARM_CFG=( [nogate]=configs/smd_abl_nogate.yaml [gate]=configs/smd_abl_gate.yaml )
declare -A ARM_DIR=( [nogate]=runs_abl/nogate            [gate]=runs_abl/gate )

NPROC=${NPROC:-2}     # GPU 数（与原 train_ddp.sh 一致，默认 2）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "=== 门控消融 ==="
echo "entities (${#ENTITIES[@]}): ${ENTITIES[*]}"
echo "arms: ${ARM_NAMES[*]}   nproc=${NPROC}   FORCE=${FORCE:-0}"
echo

for ent in "${ENTITIES[@]}"; do
  for arm in "${ARM_NAMES[@]}"; do
    cfg=${ARM_CFG[$arm]}
    dir=${ARM_DIR[$arm]}
    ckpt="${dir}/${ent}/last.pt"
    echo "---- [$arm] entity=$ent ----"

    if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "[skip-train] 已存在 $ckpt（FORCE=1 可强制重训）"
    else
      torchrun --standalone --nproc_per_node="${NPROC}" \
        scripts/train.py --config "$cfg" --entity "$ent"
    fi

    # eval：用 last.pt 产出分数
    python scripts/eval.py --config "$cfg" --entity "$ent" --ckpt "$ckpt"
    echo
  done
done

echo "=== 训练+推理完成。下一步对比： ==="
echo "python scripts/compare_gate_ablation.py --entities ${ENTITIES[*]}"
