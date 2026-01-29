# src/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.msd import get_loaders
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

# -------------------------
# Helpers: deep supervision
# -------------------------
from typing import Sequence, Union

DSOut = Union[torch.Tensor, Sequence[torch.Tensor]]

def _is_deep_supervision_output(x: DSOut) -> bool:
    # MONAI DynUNet DS may return:
    #  - list/tuple of tensors: [ (B,C,D,H,W), ... ]
    #  - a stacked tensor: (B,N,C,D,H,W)
    return isinstance(x, (list, tuple)) or (isinstance(x, torch.Tensor) and x.ndim == 6)

def _iter_ds_outputs(x: DSOut) -> list[torch.Tensor]:
    # Always return a list of (B,C,D,H,W) tensors
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, torch.Tensor) and x.ndim == 6:
        # x: (B,N,C,D,H,W) -> list of N tensors (B,C,D,H,W)
        return [x[:, i] for i in range(x.shape[1])]
    return [x]  # not DS

def _main_output(x: DSOut) -> torch.Tensor:
    # Highest-resolution output is the first one
    return _iter_ds_outputs(x)[0]

def _ds_weights(n: int) -> List[float]:
    """
    nnU-Net-style weights: 1.0, 0.5, 0.25, ...
    normalized later.
    """
    if n <= 1:
        return [1.0]
    return [1.0] + [0.5**i for i in range(1, n)]

def _resize_label_to_logits(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    y:     (B, 1, D, H, W) int/long
    logits:(B, C, d, h, w)
    Returns y resized to logits spatial shape using nearest neighbor.
    """
    if logits.shape[2:] == y.shape[2:]:
        return y
    y_ds = torch.nn.functional.interpolate(
        y.float(),
        size=logits.shape[2:],
        mode="nearest",
    ).long()
    return y_ds


# -------------------------
# Visualization helper
# -------------------------
def _save_example_png(run_dir: Path, idx: int, image: torch.Tensor, label: torch.Tensor, pred: torch.Tensor) -> None:
    """
    image: (1, D, H, W)
    label: (D, H, W) int
    pred:  (D, H, W) int
    Saves a mid-slice panel: Image | GT | Pred
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -------------------------
# Inference helper (DS-safe)
# -------------------------
def infer_logits(model: nn.Module, x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Sliding-window inference compatible with deep supervision models.
    Uses only the highest-resolution output from predictor.
    """
    icfg = cfg.get("infer", {})
    roi_size = tuple(icfg.get("roi_size", [96, 96, 96]))
    sw_batch_size = int(icfg.get("sw_batch_size", 1))
    overlap = float(icfg.get("overlap", 0.5))

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        out = model(patch)
        return _main_output(out)

    return sliding_window_inference(
        inputs=x,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
    )


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
    ppred = post_pred()    # argmax + onehot(2)
    plabel = post_label()  # onehot(2)

    history = CSVHistory(
        path=run_dir / "history.csv",
        fieldnames=("epoch", "train_loss", "val_loss", "val_dice"),
    )

    best_dice = -1.0
    best_epoch = -1
    best_path = run_dir / "checkpoints" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    max_epochs = int(cfg["train"]["max_epochs"])
    val_interval = int(cfg["train"]["val_interval"])
    num_examples = int(cfg["report"]["num_examples"])

    ds_enabled = bool(cfg.get("model", {}).get("deep_supervision", False))

    for epoch in range(1, max_epochs + 1):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_losses: List[float] = []

        train_pbar = tqdm(train_loader, desc=f"Train | Epoch {epoch:03d}", leave=False)
        for batch in train_pbar:
            x = batch["image"].to(device)               # (B,1,D,H,W)
            y = batch["label"].to(device).long()        # enforce int labels

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=bool(cfg["train"]["amp"])):
                out = model(x)

                if _is_deep_supervision_output(out):
                    outs = _iter_ds_outputs(out)
                    weights = _ds_weights(len(outs))
                    wsum = float(sum(weights))

                    loss_total = 0.0
                    for w, logits in zip(weights, outs):
                        y_ds = _resize_label_to_logits(y, logits)
                        ld = dice(logits, y_ds)
                        lce = ce(logits, y_ds.squeeze(1))
                        loss_total = loss_total + (w / wsum) * (ld + lce)
                    loss = loss_total
                else:
                    logits = out
                    loss = dice(logits, y) + ce(logits, y.squeeze(1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_val = float(loss.item())
            train_losses.append(loss_val)

            # Update batch progress bar with running loss
            train_pbar.set_postfix(loss=f"{loss_val:.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # -------------------------
        # Validate
        # -------------------------
        val_loss = float("nan")
        val_dice = float("nan")

        if epoch % val_interval == 0:
            model.eval()
            vlosses: List[float] = []
            dmetric.reset()

            example_count = 0
            printed_sanity = False

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Val   | Epoch {epoch:03d}", leave=False)
                for vb in val_pbar:
                    vx = vb["image"].to(device)
                    vy = vb["label"].to(device).long()

                    with autocast(enabled=bool(cfg["train"]["amp"])):
                        vlogits = infer_logits(model, vx, cfg)  # (B,2,D,H,W)
                        vloss = dice(vlogits, vy) + ce(vlogits, vy.squeeze(1))

                    vloss_val = float(vloss.item())
                    vlosses.append(vloss_val)
                    val_pbar.set_postfix(loss=f"{vloss_val:.4f}")

                    # Dice metric (decollate best practice)
                    val_outputs = [ppred(i) for i in decollate_batch(vlogits)]
                    val_labels = [plabel(i) for i in decollate_batch(vy)]
                    dmetric(y_pred=val_outputs, y=val_labels)

                    # One-time sanity print: epoch 1, first val batch
                    if epoch == 1 and not printed_sanity:
                        pred_lbl = torch.argmax(vlogits, dim=1).squeeze(0)      # (D,H,W)
                        gt_lbl = vy.squeeze(0).squeeze(0)                        # (D,H,W)
                        print("GT unique:", torch.unique(gt_lbl).tolist(), "FG voxels:", int((gt_lbl == 1).sum()))
                        print("PR unique:", torch.unique(pred_lbl).tolist(), "FG voxels:", int((pred_lbl == 1).sum()))
                        printed_sanity = True

                    # Save qualitative examples (use main output)
                    if example_count < num_examples:
                        pred_lbl = torch.argmax(vlogits, dim=1).squeeze(0)       # (D,H,W)
                        gt_lbl = vy.squeeze(0).squeeze(0)                         # (D,H,W)
                        _save_example_png(run_dir, example_count, vx.squeeze(0), gt_lbl, pred_lbl)
                        example_count += 1

            val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            val_dice = float(dmetric.aggregate().item())

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": best_dice,
                        "model_name": str(cfg.get("model", {}).get("name", "")),
                        "deep_supervision": ds_enabled,
                    },
                    best_path,
                )

        # Log epoch
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice})

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
        "model_name": str(cfg.get("model", {}).get("name", "")),
        "deep_supervision": bool(cfg.get("model", {}).get("deep_supervision", False)),
        "deep_supr_num": int(cfg.get("model", {}).get("deep_supr_num", 0) or 0),
    }
    save_json(metrics_summary, run_dir / "metrics_summary.json")

    print(
        "Training complete.\n"
        f"Run dir: {run_dir}\n"
        f"Best checkpoint: {best_path}\n"
        f"Best epoch: {best_epoch}\n"
        f"Best val_dice: {best_dice:.4f}\n"
        f"Deep supervision: {metrics_summary['deep_supervision']}"
    )


if __name__ == "__main__":
    main()
