# baseline_3d_unet_heart

Baseline 3D U-Net Pipeline for Left Atrium Segmentation (MRI)

Built with PyTorch + MONAI, including fully automated PDF experiment reporting.

## Overview

This repository provides a reproducible, end-to-end deep learning pipeline for 3D medical image segmentation, demonstrated using the Medical Segmentation Decathlon (MSD) Task02 — Left Atrium MRI dataset.

Key features:
- Dataset ingestion using `dataset.json`
- Robust preprocessing and augmentation
- Patch-based 3D U-Net training (MONAI)
- Sliding-window inference
- Automated logging and run artifacts
- Standardized PDF experiment reports (learning curves, metrics, qualitative examples, config snapshot)

The template is dataset-agnostic so you can adapt it quickly to other 3D segmentation tasks.

## Repository Structure

```
baseline_3d_unet_heart/
├── configs/
│   └── baseline.yaml
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── metrics.py
│   ├── utils.py
│   ├── data/
│   │   └── msd.py
│   └── models/
│       └── unet3d.py
├── scripts/
│   └── 04_make_pdf_report.py
├── outputs/
│   └── runs/
├── docs/
│   ├── USAGE.md
│   └── CHANGELOG.md
├── README.md
└── .gitignore
```

## Dataset Preparation

This pipeline follows the Medical Segmentation Decathlon (MSD) folder layout. Your dataset directory must contain:

```
heart_segmentation/
├── dataset.json
├── imagesTr/
├── labelsTr/
└── imagesTs/
```

Example (Windows):

```
C:\Users\hyeon\Documents\miniconda_medimg_env\data\heart_segmentation
```

Point your config to this dataset path in:
- `configs/baseline.yaml`

## Environment Setup

Activate your conda environment:

```bash
conda activate medimg
```

Verify key packages:

```bash
python -c "import torch, monai; print(torch.__version__, monai.__version__)"
```

(Ensure you have a compatible PyTorch and MONAI version for your CUDA and GPU drivers.)

## Training

### Quick Sanity Run (Recommended First Step)

Before launching a full experiment, run a short test to confirm data loading, GPU execution, training loop and logging.

1. Edit `configs/baseline.yaml` and set:

```yaml
train:
  max_epochs: 10
```

2. Run:

```bash
python -m src.train --config configs\baseline.yaml
```

This run will exercise the pipeline with a short training schedule and produce example outputs.

### Full Training Run

When ready, run a full experiment by setting:

```yaml
train:
  max_epochs: 200
```

Then run:

```bash
python -m src.train --config configs\baseline.yaml # baseline 3d unet
python -m src.train --config configs\dynunet.yaml # nn-UNet
python -m src.train --config configs\dynunet_ds.yaml # nn-UNet with deep supervision
```

Each run automatically generates a timestamped run directory:

```
outputs\runs\run_YYYYMMDD_HHMMSS\
├── config_resolved.yaml
├── dataset_summary.json
├── history.csv
├── metrics_summary.json
├── checkpoints\best.pt
└── examples\
```

## Generating the PDF Report

After training finishes, generate the experiment report:

```bash
python scripts\04_make_pdf_report.py --run_dir outputs\runs\run_YYYYMMDD_HHMMSS
```

This produces:

```
outputs\runs\run_YYYYMMDD_HHMMSS\report.pdf
```

### Report Contents

- Page 1 — Run Summary + Learning Curves
  - Dataset metadata
  - Training configuration
  - Device (GPU/CPU) information
  - Best epoch and Dice score
  - Loss & Dice curves

- Page 2 — Qualitative Examples
  - Mid-slice panels: Image | Ground Truth | Prediction

- Page 3+ — Configuration Snapshot
  - Full YAML config used for reproducibility

The PDF provides a concise, publication-ready summary of each experiment.

## Common Commands

- List all runs:
  - Windows: `dir outputs\runs`
  - POSIX: `ls outputs/runs`
- Inspect training logs:
  - `type outputs\runs\run_YYYYMMDD_HHMMSS\history.csv` (Windows)
  - `cat outputs/runs/run_YYYYMMDD_HHMMSS/history.csv` (POSIX)
- View best checkpoint metrics:
  - `type outputs\runs\run_YYYYMMDD_HHMMSS\metrics_summary.json`

## Git Workflow (Recommended)

Initial commit:

```bash
git init
git add .
git commit -m "Baseline 3D U-Net heart segmentation pipeline with PDF reporting"
```

Tag stable baseline:

```bash
git tag v0.1.0
git push origin main --tags
```

## Methodological Notes

- Model: 3D U-Net (MONAI implementation)
- Loss: Dice + weighted Cross Entropy
- Metric: Dice (foreground only)
- Inference: Sliding window (96×96×96)
- Normalization: Z-score (non-zero voxels)
- Patch Sampling: Foreground-biased

These choices are intended to provide a robust baseline for volumetric cardiac segmentation.

## Citation & Dataset

If you use this pipeline in academic work, please cite:

- Simpson AL, et al. Medical Segmentation Decathlon: A Large-Scale Learning Benchmark for Medical Image Segmentation. arXiv:1902.09063
- Project MONAI Consortium. MONAI: An Open-Source Framework for Deep Learning in Healthcare Imaging.

## Contact

For questions or issues, open an issue on the repository or contact the maintainer.
