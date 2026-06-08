#!/usr/bin/env bash
# 重构分 + 静态幅度分 融合实验：machine-1-1 ~ 1-8。
#   每台：训 recon 基线 → eval 存 variable_rotation 分数 → 最后 static_fuse 三路对比（raw + PA F1）。
#
# 用法:
#   bash scripts/run_static_fuse.sh                  # 默认 1-1 ~ 1-8，encoder=staged
#   EPOCHS=60 bash scripts/run_static_fuse.sh        # epochs 是上限，early-stop 会提前停
#   ENCODER=dualaxis bash scripts/run_static_fuse.sh # 复现你之前贴的那组 dualaxis 数
#   FORCE=1   bash scripts/run_static_fuse.sh        # 强制重训
set -euo pipefail
cd "$(dirname "$0")/.."

# SMD 全部 28 台（1组8 + 2组9 + 3组11）
DEFAULT=(
  machine-1-1 machine-1-2 machine-1-3 machine-1-4 machine-1-5 machine-1-6 machine-1-7 machine-1-8
  machine-2-1 machine-2-2 machine-2-3 machine-2-4 machine-2-5 machine-2-6 machine-2-7 machine-2-8 machine-2-9
  machine-3-1 machine-3-2 machine-3-3 machine-3-4 machine-3-5 machine-3-6 machine-3-7 machine-3-8 machine-3-9 machine-3-10 machine-3-11
)
if [ "$#" -gt 0 ]; then ENTITIES=("$@"); else ENTITIES=("${DEFAULT[@]}"); fi

CFG=configs/smd.yaml
DIR=runs_fuse
NPROC=${NPROC:-2}
EPOCHS=${EPOCHS:-40}              # 上限；配 early_stop_patience（config 里=3）自动提前停
ENCODER=${ENCODER:-staged}       # 默认你当前的方法 staged；ENCODER=dualaxis 复现旧基线
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "=== recon 基线 (${ENCODER}) for static-fuse 实验 ==="
echo "entities (${#ENTITIES[@]}): ${ENTITIES[*]}   epochs<=$EPOCHS  encoder=$ENCODER"
echo

for ent in "${ENTITIES[@]}"; do
  ckpt="${DIR}/${ent}/last.pt"
  echo "---- $ent ----"
  if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip-train] $ckpt"
  else
    torchrun --standalone --nproc_per_node="${NPROC}" \
      scripts/train.py --config "$CFG" --entity "$ent" \
      --set model.encoder="$ENCODER" --set train.epochs="$EPOCHS" --set train.save_dir="$DIR"
  fi
  python scripts/eval.py --config "$CFG" --entity "$ent" --ckpt "$ckpt" \
    --set model.encoder="$ENCODER" --set train.save_dir="$DIR"
  echo
done

echo "=== 三路融合对比 ==="
python scripts/static_fuse.py --config "$CFG" --recon_dir "$DIR" --entities "${ENTITIES[@]}"
