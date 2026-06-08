import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..data.masking import mixed_mask
from ..data.smd_dataset import build_smd_datasets
from ..models.mask_predict import build_model
from ..utils.logger import Logger
from .losses import masked_l2_loss


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def train(cfg):
    rank = _rank()
    world = _world()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")) if _is_dist() else 0)
    torch.manual_seed(cfg.seed + rank)

    save_dir = Path(cfg.train.save_dir) / cfg.data.entity
    logger = Logger(str(save_dir), rank=rank)
    logger.info(f"world={world} device={device} entity={cfg.data.entity}")

    train_ds, val_ds, _, _, _ = build_smd_datasets(cfg)
    logger.info(f"train_windows={len(train_ds)} val_windows={len(val_ds)}")

    if _is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True,
    )

    model = build_model(cfg).to(device)
    if _is_dist():
        model = DDP(model, device_ids=[device.index])

    groups = [{"params": list(model.parameters()), "lr": cfg.train.lr,
               "base_lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}]
    optim = torch.optim.AdamW(groups)
    scaler = GradScaler("cuda", enabled=cfg.train.amp)

    total_steps = cfg.train.epochs * len(train_loader)
    best_val = float("inf")
    no_improve = 0
    es_patience = int(getattr(cfg.train, "early_stop_patience", 0))     # 0=关闭
    es_min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0))
    global_step = 0

    for epoch in range(cfg.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)  # (B, T, D)
            mask = mixed_mask(
                batch.shape,
                ratio=cfg.mask.ratio,
                span_prob=cfg.mask.span_prob,
                span_min=cfg.mask.span_min,
                span_max=cfg.mask.span_max,
                var_mask_prob=float(getattr(cfg.mask, "var_mask_prob", 0.0)),
                var_k_min=int(getattr(cfg.mask, "var_k_min", 1)),
                var_k_max=int(getattr(cfg.mask, "var_k_max", 3)),
                var_time_span=int(getattr(cfg.mask, "var_time_span", 0)),
                device=device,
            )

            lr_scale = _warmup_cosine(global_step, cfg.train.warmup_steps, total_steps)
            for g in optim.param_groups:
                g["lr"] = g["base_lr"] * lr_scale

            optim.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp):
                pred = model(batch, mask)
                loss = masked_l2_loss(pred, batch, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optim)
            scaler.update()

            if global_step % cfg.train.log_every == 0:
                logger.info(
                    f"epoch={epoch} step={global_step} lr={cfg.train.lr*lr_scale:.2e} "
                    f"loss={loss.item():.4f}"
                )
                logger.log_metrics(global_step, {
                    "epoch": epoch, "loss": loss.item(), "lr": cfg.train.lr * lr_scale,
                })
            global_step += 1

        # ---- 验证 ----
        val_loss = evaluate(model, val_loader, cfg, device)
        if _is_dist():
            t = torch.tensor([val_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            val_loss = t.item()
        logger.info(f"epoch={epoch} val_loss={val_loss:.4f}")
        logger.log_metrics(global_step, {"epoch": epoch, "val_loss": val_loss})

        # early stopping：improved 由所有 rank 上一致的（已同步）val_loss 判定 →
        # no_improve 在各 rank 同步演进 → 同一 epoch 一起 break，不会 DDP 死锁。
        improved = val_loss < best_val - es_min_delta
        if improved:
            best_val = val_loss
            no_improve = 0
            if rank == 0:
                state = model.module.state_dict() if _is_dist() else model.state_dict()
                torch.save({"model": state, "cfg": cfg._raw}, save_dir / "best.pt")
                logger.info(f"[ckpt] saved best (val={val_loss:.4f}) to {save_dir/'best.pt'}")
        else:
            no_improve += 1

        if es_patience > 0 and no_improve >= es_patience:
            logger.info(f"[early-stop] val_loss 连续 {es_patience} epoch 无提升 "
                        f"(best={best_val:.4f})，在 epoch={epoch} 提前停止")
            break

    if rank == 0:
        state = model.module.state_dict() if _is_dist() else model.state_dict()
        torch.save({"model": state, "cfg": cfg._raw}, save_dir / "last.pt")


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        mask = mixed_mask(
            batch.shape, cfg.mask.ratio, cfg.mask.span_prob,
            cfg.mask.span_min, cfg.mask.span_max,
            var_mask_prob=float(getattr(cfg.mask, "var_mask_prob", 0.0)),
            var_k_min=int(getattr(cfg.mask, "var_k_min", 1)),
            var_k_max=int(getattr(cfg.mask, "var_k_max", 3)),
            var_time_span=int(getattr(cfg.mask, "var_time_span", 0)),
            device=device,
        )
        with autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp):
            pred = model(batch, mask)
            loss = masked_l2_loss(pred, batch, mask)
        total += loss.item() * batch.size(0)
        n += batch.size(0)
    return total / max(1, n)
