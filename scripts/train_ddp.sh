#!/usr/bin/env bash
# 用法: bash scripts/train_ddp.sh <config> <entity>
# 示例: bash scripts/train_ddp.sh configs/smd.yaml machine-1-1
set -euo pipefail

CONFIG=${1:-configs/smd.yaml}
ENTITY=${2:-machine-1-1}

export OMP_NUM_THREADS=4
torchrun \
  --standalone \
  --nproc_per_node=2 \
  scripts/train.py \
  --config "$CONFIG" \
  --entity "$ENTITY"
