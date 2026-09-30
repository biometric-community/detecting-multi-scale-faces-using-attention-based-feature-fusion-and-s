"""# SANet training entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sanet.data import WiderFaceDataset, WiderTrainTransform, resolve_wider_paths, wider_collate
from sanet.engine.config import load_config, resolve_data_root
from sanet.losses import SANetLoss
from sanet.models import SANet


def train(cfg: dict, project_root: Path) -> None:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_root = resolve_data_root(cfg, project_root)
    paths = resolve_wider_paths(data_root)
    max_samples = cfg["data"].get("max_train_samples")
    ds = WiderFaceDataset(
        paths["train_images"],
        paths["train_gt"],
        transforms=WiderTrainTransform(cfg["model"]["input_size"]),
        max_samples=max_samples,
    )
    # Paper batch is 14; use micro-batch + grad accumulation when GPU memory is tight.
    target_bs = int(cfg["train"]["batch_size"])
    micro_bs = int(cfg["train"].get("micro_batch_size") or target_bs)
    accum = max(1, target_bs // micro_bs)
    if micro_bs * accum != target_bs:
        # e.g. target 14, micro 2 → accum 7
        accum = max(1, (target_bs + micro_bs - 1) // micro_bs)
    loader = DataLoader(
        ds,
        batch_size=micro_bs,
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=wider_collate,
        pin_memory=True,
        drop_last=True,
    )
    model = SANet(
        attention_mode=cfg["model"].get("attention_mode", "sa"),
        pretrained_backbone=cfg["model"].get("pretrained_backbone", True),
    ).to(device)
    criterion = SANetLoss(iou_thresh=cfg["train"].get("iou_thresh", 0.35))
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"].get("momentum", 0.9),
        weight_decay=cfg["train"].get("weight_decay", 5e-4),
    )
    max_iters = int(cfg["train"]["max_iters"])
    milestones = set(cfg["train"].get("lr_milestones", [80000, 100000]))
    log_every = int(cfg["train"].get("log_every", 50))
    ckpt_dir = project_root / cfg["paths"]["checkpoint_dir"]
    log_dir = project_root / cfg["paths"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    it = 0
    epoch = 0
    history = []
    t0 = time.time()
    data_iter = iter(loader)
    print(
        f"Train device={device} target_batch={target_bs} micro_batch={micro_bs} accum={accum}",
        flush=True,
    )
    while it < max_iters:
        if it in milestones:
            for g in optim.param_groups:
                g["lr"] *= cfg["train"].get("lr_gamma", 0.1)
        optim.zero_grad(set_to_none=True)
        loss_meter = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "n_matched": 0}
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(loader)
                batch = next(data_iter)
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            out = model(images)
            losses = criterion(out, targets, model.max_in, model.max_out)
            (losses["loss"] / accum).backward()
            loss_meter["loss"] += float(losses["loss"].detach()) / accum
            loss_meter["loss_cls"] += float(losses["loss_cls"].detach()) / accum
            loss_meter["loss_reg"] += float(losses["loss_reg"].detach()) / accum
            loss_meter["n_matched"] += int(losses["n_matched"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        it += 1
        if it % log_every == 0 or it == 1:
            row = {
                "iter": it,
                "epoch": epoch,
                "loss": loss_meter["loss"],
                "loss_cls": loss_meter["loss_cls"],
                "loss_reg": loss_meter["loss_reg"],
                "lr": optim.param_groups[0]["lr"],
                "n_matched": loss_meter["n_matched"],
                "time_s": time.time() - t0,
                "micro_batch": micro_bs,
                "accum": accum,
            }
            history.append(row)
            print(
                f"iter {it}/{max_iters} loss={row['loss']:.4f} "
                f"cls={row['loss_cls']:.4f} reg={row['loss_reg']:.4f} lr={row['lr']:.6f}",
                flush=True,
            )
            with open(log_dir / "train_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        if it % int(cfg["train"].get("ckpt_every", 5000)) == 0 or it == max_iters:
            ckpt = {
                "iter": it,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / f"sanet_iter{it:06d}.pt")
            torch.save(ckpt, ckpt_dir / "latest.pt")

    summary = {
        "max_iters": max_iters,
        "final_iter": it,
        "final_loss": history[-1]["loss"] if history else None,
        "history_len": len(history),
        "seconds": time.time() - t0,
        "num_train_images": len(ds),
        "attention_mode": cfg["model"].get("attention_mode", "sa"),
        "batch_size": target_bs,
        "micro_batch_size": micro_bs,
        "grad_accum_steps": accum,
    }
    with open(log_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("Wrote", log_dir / "train_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SANet on WIDER FACE")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config if not Path(args.config).is_absolute() else args.config)
    train(cfg, project_root)


if __name__ == "__main__":
    main()
