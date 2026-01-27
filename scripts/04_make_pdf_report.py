from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # ensure columns exist even if early/partial runs
    for col in ["epoch", "train_loss", "val_loss", "val_dice"]:
        if col not in df.columns:
            df[col] = None
    return df


def _plot_curves(history: pd.DataFrame, out_png: Path) -> None:
    epochs = history["epoch"].values
    train_loss = history["train_loss"].values
    val_loss = history["val_loss"].values
    val_dice = history["val_dice"].values

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dice, label="val_dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title("Validation Dice (label=1)")
    plt.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def _list_example_images(examples_dir: Path, max_n: int = 12) -> List[Path]:
    if not examples_dir.exists():
        return []
    imgs = sorted(examples_dir.glob("*.png"))
    return imgs[:max_n]


def _draw_heading(c: Canvas, x: float, y: float, text: str, size: int = 16) -> None:
    c.setFont("Helvetica-Bold", size)
    c.drawString(x, y, text)


def _draw_kv_block(
    c: Canvas,
    x: float,
    y: float,
    items: List[Tuple[str, str]],
    key_width: float = 1.8 * inch,
    line_height: float = 0.22 * inch,
    font_size: int = 10,
) -> float:
    c.setFont("Helvetica", font_size)
    yy = y
    for k, v in items:
        c.setFont("Helvetica-Bold", font_size)
        c.drawString(x, yy, k)
        c.setFont("Helvetica", font_size)
        c.drawString(x + key_width, yy, v)
        yy -= line_height
    return yy


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _best_row(history: pd.DataFrame) -> Optional[pd.Series]:
    if history.empty:
        return None
    # best by val_dice; ignore NaNs
    h = history.copy()
    h["val_dice"] = pd.to_numeric(h["val_dice"], errors="coerce")
    if h["val_dice"].notna().sum() == 0:
        return None
    idx = h["val_dice"].idxmax()
    return h.loc[idx]


def _draw_image_fit(
    c: Canvas,
    img_path: Path,
    x: float,
    y_top: float,
    box_w: float,
    box_h: float,
) -> float:
    """
    Draw image fitted to (box_w, box_h) with top-left anchor at (x, y_top).
    Returns y after drawing (bottom y - small gap).
    """
    if not img_path.exists():
        return y_top

    # reportlab drawImage uses bottom-left anchor; convert:
    y_bottom = y_top - box_h
    c.drawImage(str(img_path), x, y_bottom, width=box_w, height=box_h, preserveAspectRatio=True, anchor="c")
    return y_bottom - 0.12 * inch


def make_pdf(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    history_path = run_dir / "history.csv"
    cfg_path = run_dir / "config_resolved.yaml"
    ds_path = run_dir / "dataset_summary.json"
    metrics_path = run_dir / "metrics_summary.json"
    examples_dir = run_dir / "examples"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.csv: {history_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml: {cfg_path}")
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset_summary.json: {ds_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics_summary.json: {metrics_path}")

    history = _load_history(history_path)
    cfg = _read_yaml(cfg_path)
    ds = _read_json(ds_path)
    metrics = _read_json(metrics_path)

    title = "Left Atrium Segmentation — Baseline 3D U-Net"
    subtitle = f"Run: {run_dir.name}"

    pdf_path = run_dir / "report.pdf"
    c = Canvas(str(pdf_path), pagesize=LETTER)
    W, H = LETTER

    # Common margins
    x0 = 0.75 * inch
    y = H - 0.9 * inch

    # ---- Page 1: Summary + Curves ----
    _draw_heading(c, x0, y, title, size=18)
    y -= 0.32 * inch
    c.setFont("Helvetica", 11)
    c.drawString(x0, y, subtitle)
    y -= 0.35 * inch

    best = _best_row(history)
    best_epoch = _safe_str(best["epoch"]) if best is not None else "N/A"
    best_dice_hist = _safe_str(best["val_dice"]) if best is not None else "N/A"
    best_dice_ckpt = _safe_str(metrics.get("best_val_dice", "N/A"))

    summary_items = [
        ("Device", _safe_str(metrics.get("device", "auto"))),
        ("Dataset", _safe_str(ds.get("dataset_json", "dataset.json"))),
        ("Data root", _safe_str(ds.get("data_root", ""))),
        ("Train / Val", f"{ds.get('num_train', 'N/A')} / {ds.get('num_val', 'N/A')}"),
        ("Labels", "0=background, 1=left atrium"),
        ("Best epoch (history)", best_epoch),
        ("Best val_dice (history)", best_dice_hist),
        ("Best val_dice (checkpoint)", best_dice_ckpt),
        ("Best checkpoint", _safe_str(metrics.get("best_checkpoint", ""))),
    ]
    y = _draw_kv_block(c, x0, y, summary_items, key_width=2.1 * inch, font_size=10)
    y -= 0.15 * inch

    curves_png = run_dir / "curves.png"
    _plot_curves(history, curves_png)

    # Draw curves image
    y_top = y
    box_w = W - 2 * x0
    box_h = 3.2 * inch
    y = _draw_image_fit(c, curves_png, x0, y_top, box_w, box_h)

    c.showPage()

    # ---- Page 2: Qualitative examples ----
    y = H - 0.9 * inch
    _draw_heading(c, x0, y, "Qualitative Examples", size=16)
    y -= 0.35 * inch
    c.setFont("Helvetica", 10)
    c.drawString(x0, y, "Panels: Image | Ground Truth | Prediction (mid-slice)")
    y -= 0.3 * inch

    example_paths = _list_example_images(examples_dir, max_n=6)

    if not example_paths:
        c.setFont("Helvetica", 11)
        c.drawString(x0, y, "No example images found in run_dir/examples/.")
        c.showPage()
    else:
        # 1-column layout for larger tiles
        box_w = W - 2 * x0
        box_h = 2.9 * inch  # adjust as desired

        for i, p in enumerate(example_paths):
            if y - box_h < 0.8 * inch:
                c.showPage()
                y = H - 0.9 * inch
                _draw_heading(c, x0, y, "Qualitative Examples (cont.)", size=16)
                y -= 0.4 * inch

            y = _draw_image_fit(c, p, x0, y, box_w, box_h)

    c.showPage()


    # ---- Page 3: Config snapshot ----
    y = H - 0.9 * inch
    _draw_heading(c, x0, y, "Configuration Snapshot", size=16)
    y -= 0.3 * inch

    c.setFont("Helvetica", 8)
    cfg_text = (run_dir / "config_resolved.yaml").read_text(encoding="utf-8").splitlines()
    max_lines = 120  # keep it readable; full config is still saved separately
    for line in cfg_text[:max_lines]:
        if y < 0.75 * inch:
            c.showPage()
            y = H - 0.9 * inch
            c.setFont("Helvetica", 8)
        c.drawString(x0, y, line[:120])
        y -= 0.14 * inch

    c.save()
    return pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str, help="Path to outputs/runs/run_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    pdf_path = make_pdf(Path(args.run_dir))
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
