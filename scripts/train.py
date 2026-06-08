"""单卡或 DDP 训练入口。DDP 用 `torchrun --nproc_per_node=2 scripts/train.py ...`。"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.trainer import train  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _parse_val(s: str):
    """把 --set 的字符串值解析成 bool/int/float/str。"""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _apply_override(cfg, dotted: str):
    """形如 train.epochs=40，同时改 cfg(SimpleNamespace) 与 cfg._raw(dict，存入 ckpt)。"""
    key, _, val = dotted.partition("=")
    v = _parse_val(val)
    parts = key.split(".")
    ns = cfg
    for p in parts[:-1]:
        ns = getattr(ns, p)
    setattr(ns, parts[-1], v)
    d = cfg._raw
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", default=None, help="覆盖 config 中的 data.entity")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="覆盖任意配置项，点号路径，可多次。如 --set train.epochs=40")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.entity:
        cfg.data.entity = args.entity
    for ov in args.overrides:
        _apply_override(cfg, ov)

    # 全局禁用 cuDNN：避开 depthwise conv + fp16 在某些 cuDNN 版本上
    # backward pass 时 CUDNN_STATUS_NOT_INITIALIZED 的兼容问题。
    # Transformer 主干用 cuBLAS（不受影响），影响的只有 LocalConv 那 640 参数的小卷积。
    torch.backends.cudnn.enabled = False

    # torchrun 会设置这些环境变量
    if "RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    train(cfg)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
