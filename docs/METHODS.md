
---

# `docs/METHODS.md` (Manuscript-Ready)

```markdown
# Methods

## Dataset

We used the **Medical Segmentation Decathlon (MSD) Task02: Left Atrium MRI dataset**, which consists of 20 training and 10 testing three-dimensional cardiac MRI volumes with expert-annotated left atrium segmentations. Images were provided in NIfTI format and accompanied by a standardized `dataset.json` metadata file.

The training set was randomly split into **80% training and 20% validation subsets** using a fixed random seed to ensure reproducibility.

---

## Preprocessing

All volumes underwent the following preprocessing steps:

1. Reorientation to standard RAS coordinate space.
2. Resampling to isotropic voxel spacing of **1.25 mm × 1.25 mm × 1.25 mm**.
3. Intensity normalization using **z-score normalization applied to non-zero voxels**, appropriate for MRI data.
4. Foreground-aware patch sampling using spatial crops of **96 × 96 × 96 voxels**.

---

## Data Augmentation

To improve generalization, the following augmentations were applied during training:

- Random flipping along all spatial axes (probability = 0.5)
- Random 90-degree rotations (probability = 0.2)

---

## Model Architecture

We implemented a **3D U-Net architecture** using the MONAI framework, consisting of:

- Five resolution levels
- Channel configuration: [16, 32, 64, 128, 256]
- Residual convolutional blocks (two per level)
- Instance normalization

The network accepts single-channel volumetric inputs and outputs two-class probability maps (background and left atrium).

---

## Training Strategy

Training was performed using:

- **Loss function:** Combined Dice loss and weighted cross-entropy loss  
- **Optimizer:** AdamW (learning rate = 2×10⁻⁴, weight decay = 1×10⁻⁵)  
- **Mixed precision training:** Enabled using PyTorch AMP  
- **Batch size:** 1 (patch-based)  

Models were trained for up to **200 epochs**, with validation performed after every epoch.

---

## Inference

During validation, predictions were generated using **sliding window inference** with:

- Window size: 96 × 96 × 96
- Overlap: 50%

---

## Evaluation Metric

Segmentation performance was evaluated using the **Dice similarity coefficient (DSC)** computed on the **foreground (left atrium) class only**.

---

## Reproducibility

Each experiment automatically logs:

- Training configuration snapshot
- Random seed
- Training history (CSV)
- Best-performing model checkpoint
- Qualitative segmentation examples
- Structured PDF experiment report

This ensures **full experimental reproducibility**.

---

## Implementation

All experiments were implemented using **PyTorch** and **MONAI**. Automated PDF reporting was generated using **Matplotlib** and **ReportLab**.

---

## References

1. Simpson AL, et al. *Medical Segmentation Decathlon: A Large-Scale Learning Benchmark for Medical Image Segmentation.* arXiv:1902.09063.  
2. Project MONAI Consortium. *MONAI: An Open-Source Framework for Deep Learning in Healthcare Imaging.*

---
