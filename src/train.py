# src/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import decollate_batch
from monai.losses import DiceLoss
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

from src.data.msd import get_loaders
from src.infer import infer_logits
from src.metrics import dice_metric, post_label, post_pred
from src.models.unet3d import build_model
from src.utils import (
    CSVHistory,
    get_device,
    load_yaml,
    make_run_dir,
    save_json,
    save_yaml,
    set_determinism,
)


def _save_example_png(run_dir: Path, idx: int, image: torch.Tensor, label: torch.Tensor, pred: torch.Tensor) -> None:
    """
    image: (1, D, H, W), label/pred: (D, H, W) int
    Saves a mid-slice overlay-style panel.
    """
    img = image.squeeze(0).cpu().numpy()
    lab = label.cpu().numpy()
    prd = pred.cpu().numpy()

    z = img.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(img[z], cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(img[z], cmap="gray")
    ax[1].imshow(lab[z], alpha=0.5)
    ax[1].set_title("GT")
    ax[2].imshow(img[z], cmap="gray")
    ax[2].imshow(prd[z], alpha=0.5)
    ax[2].set_title("Pred")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = run_dir / "examples" / f"example_{idx:02d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_determinism(int(cfg["split"]["seed"]))

    run_dir = make_run_dir(cfg["paths"]["output_root"])
    save_yaml(cfg, run_dir / "config_resolved.yaml")

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, dataset_summary = get_loaders(cfg)
    save_json(dataset_summary, run_dir / "dataset_summary.json")

    model = build_model(cfg).to(device)

    # Loss: Dice + weighted CE (foreground heavier)
    dice = DiceLoss(to_onehot_y=True, softmax=True)
    ce_weight = torch.tensor([0.2, 0.8], device=device)  # [bg, fg]
    ce = nn.CrossEntropyLoss(weight=ce_weight)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))

    dmetric = dice_metric()
    ppred = post_pred()   # argmax + onehot(2)
    plabel = post_label() # onehot(2)

    history = CSVHistory(
        path=run_dir / "history.csv",
        fieldnames=("epoch", "train_loss", "val_loss", "val_dice"),
    )

    best_dice = -1.0
    best_epoch = -1
    best_path = run_dir / "checkpoints" / "best.pt"

    max_epochs = int(cfg["train"]["max_epochs"])
    val_interval = int(cfg["train"]["val_interval"])
    num_examples = int(cfg["report"]["num_examples"])

    for epoch in range(1, max_epochs + 1):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["image"].to(device)               # (B,1,D,H,W)
            y = batch["label"].to(device).long()        # enforce int labels

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg["train"]["amp"])):
                logits = model(x)                       # (B,2,D,H,W)
                # DiceLoss expects (B,2,...) pred and (B,1,...) label (int OK)
                loss_d = dice(logits, y)
                # CrossEntropyLoss expects target as (B,D,H,W) class indices
                loss_ce = ce(logits, y.squeeze(1))
                loss = loss_d + loss_ce

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # -------------------------
        # Validate
        # -------------------------
        val_loss = float("nan")
        val_dice = float("nan")

        if epoch % val_interval == 0:
            model.eval()
            vlosses = []
            dmetric.reset()
            example_count = 0
            printed_sanity = False

            with torch.no_grad():
                for vb in val_loader:
                    vx = vb["image"].to(device)
                    vy = vb["label"].to(device).long()

                    with autocast(enabled=bool(cfg["train"]["amp"])):
                        vlogits = infer_logits(model, vx, cfg)
                        vloss_d = dice(vlogits, vy)
                        vloss_ce = ce(vlogits, vy.squeeze(1))
                        vloss = vloss_d + vloss_ce
                    vlosses.append(vloss.item())

                    # DiceMetric best practice: decollate per item
                    val_outputs = [ppred(i) for i in decollate_batch(vlogits)]
                    val_labels = [plabel(i) for i in decollate_batch(vy)]
                    dmetric(y_pred=val_outputs, y=val_labels)

                    # Optional sanity print for first validation batch of epoch 1
                    if epoch == 1 and not printed_sanity:
                        pred_lbl = torch.argmax(vlogits, dim=1).squeeze(0)         # (D,H,W)
                        gt_lbl = vy.squeeze(0).squeeze(0)                           # (D,H,W)
                        print(
                            "GT unique:", torch.unique(gt_lbl).tolist(),
                            "FG voxels:", int((gt_lbl == 1).sum())
                        )
                        print(
                            "PR unique:", torch.unique(pred_lbl).tolist(),
                            "FG voxels:", int((pred_lbl == 1).sum())
                        )
                        printed_sanity = True

                    # Save a few qualitative examples
                    if example_count < num_examples:
                        pred_lbl = torch.argmax(vlogits, dim=1).squeeze(0)          # (D,H,W)
                        gt_lbl = vy.squeeze(0).squeeze(0)                            # (D,H,W)
                        _save_example_png(run_dir, example_count, vx.squeeze(0), gt_lbl, pred_lbl)
                        example_count += 1

            val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            val_dice = float(dmetric.aggregate().item())

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_dice": best_dice},
                    best_path,
                )

        # Log
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice}
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | best={best_dice:.4f}"
        )

    # Final metrics summary
    metrics_summary = {
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "device": str(device),
        "best_checkpoint": str(best_path),
    }
    save_json(metrics_summary, run_dir / "metrics_summary.json")

    print(
        "Training complete.\n"
        f"Run dir: {run_dir}\n"
        f"Best checkpoint: {best_path}\n"
        f"Best epoch: {best_epoch}\n"
        f"Best val_dice: {best_dice:.4f}"
    )


if __name__ == "__main__":
    main()
