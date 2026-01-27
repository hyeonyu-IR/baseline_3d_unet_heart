# Usage Guide

This document provides a step-by-step guide for running training, evaluation, and report generation using the `baseline_3d_unet_heart` pipeline.

---

## Prerequisites

- Conda environment with required packages (PyTorch, MONAI, and other dependencies listed in your project).
- GPU recommended for training.
- Dataset prepared following the Medical Segmentation Decathlon (MSD) layout.

---

## 1. Environment Activation

Activate your conda environment:

Windows / POSIX (same command inside conda-enabled shell):
```powershell
conda activate medimg
```

Verify that PyTorch and MONAI are installed and visible to Python:
```bash
python -c "import torch, monai; print(torch.__version__, monai.__version__)"
```

---

## 2. Dataset Directory Structure

The pipeline expects an MSD-style layout. Example structure:

```
heart_segmentation/
├── dataset.json
├── imagesTr/
├── labelsTr/
└── imagesTs/
```

Configure the dataset root path in:
- `configs/baseline.yaml`

Example config snippet (Windows path shown; use POSIX paths on Linux/macOS):
```yaml
paths:
  data_root: "C:\\Users\\hyeon\\Documents\\miniconda_medimg_env\\data\\heart_segmentation"
```

---

## 3. Sanity Check Run (Recommended)

Before launching a long experiment, do a short run to confirm pipeline correctness.

1. Edit `configs/baseline.yaml` and set a short number of epochs:
```yaml
train:
  max_epochs: 10
```

2. Run the quick test:
Windows:
```powershell
python -m src.train --config configs\baseline.yaml
```
POSIX:
```bash
python -m src.train --config configs/baseline.yaml
```

What this confirms:
- Dataset loading and preprocessing
- GPU/CPU execution
- Training loop and logging
- Example image generation

---

## 4. Full Training Run

When ready for a full experiment, set:
```yaml
train:
  max_epochs: 200
```

Then run:
Windows:
```powershell
python -m src.train --config configs\baseline.yaml
```
POSIX:
```bash
python -m src.train --config configs/baseline.yaml
```

Each run creates a timestamped directory:
```
outputs/runs/run_YYYYMMDD_HHMMSS/
```
Contents include:
- `config_resolved.yaml` — resolved config used for the run
- `dataset_summary.json` — dataset metadata
- `history.csv` — per-epoch training history
- `metrics_summary.json` — best metrics / summary
- `checkpoints/best.pt` — best model checkpoint
- `examples/` — qualitative example images

---

## 5. Generate PDF Report

After training completes, create the automated PDF report:

Windows:
```powershell
python scripts\04_make_pdf_report.py --run_dir outputs\runs\run_YYYYMMDD_HHMMSS
```
POSIX:
```bash
python scripts/04_make_pdf_report.py --run_dir outputs/runs/run_YYYYMMDD_HHMMSS
```

Output:
```
outputs/runs/run_YYYYMMDD_HHMMSS/report.pdf
```

Report contents:
- Page 1: Run summary, dataset metadata, device info, best epoch & Dice, loss & Dice curves
- Page 2: Qualitative mid-slice panels (Image | Ground Truth | Prediction)
- Page 3+: Full YAML config snapshot for reproducibility

---

## 6. Inspecting Outputs

View training history:
Windows:
```powershell
type outputs\runs\run_YYYYMMDD_HHMMSS\history.csv
```
POSIX:
```bash
cat outputs/runs/run_YYYYMMDD_HHMMSS/history.csv
```

View summary metrics:
Windows:
```powershell
type outputs\runs\run_YYYYMMDD_HHMMSS\metrics_summary.json
```
POSIX:
```bash
cat outputs/runs/run_YYYYMMDD_HHMMSS/metrics_summary.json
```

List example images:
Windows:
```powershell
dir outputs\runs\run_YYYYMMDD_HHMMSS\examples
```
POSIX:
```bash
ls outputs/runs/run_YYYYMMDD_HHMMSS/examples
```

---

## 7. Recommended Workflow

1. Short sanity run (10 epochs)
2. Full training (200–300 epochs depending on dataset and compute)
3. Generate PDF report
4. Commit results, tag baseline, and archive run directory

---

## 8. Reproducibility Notes

Each run saves:
- Full configuration snapshot (`config_resolved.yaml`)
- Deterministic seeds (if enabled in config)
- Training history and metrics
- Best model checkpoint

Use the saved `config_resolved.yaml` with the checkpoint and dataset to reproduce results exactly.

---

## 9. Common Issues & Troubleshooting

CUDA not detected:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, check:
- NVIDIA drivers are installed
- CUDA toolkit or compatible PyTorch binary is used
- GPU is visible (`nvidia-smi`)

Out-of-memory (OOM):
- Reduce patch size:
```yaml
patch:
  patch_size: [80, 80, 80]
```
- Or reduce batch size:
```yaml
train:
  batch_size: 1
```

Data loading errors:
- Ensure `dataset.json` paths are correct and files exist
- Confirm image/label spacing and orientation are consistent

---

## 10. Quick Command Reference

- Train:
  - Windows: `python -m src.train --config configs\baseline.yaml`
  - POSIX: `python -m src.train --config configs/baseline.yaml`
- Generate PDF:
  - `python scripts/04_make_pdf_report.py --run_dir outputs/runs/run_YYYYMMDD_HHMMSS`
- View logs:
  - `cat outputs/runs/run_YYYYMMDD_HHMMSS/history.csv` (POSIX)  
  - `type outputs\runs\run_YYYYMMDD_HHMMSS\history.csv` (Windows)

---

## 11. Contact & Issues

For questions, feature requests, or bug reports, please open an issue in the repository.

Thank you for using baseline_3d_unet_heart — the pipeline is designed for reproducible, publication-ready 3D segmentation experiments.