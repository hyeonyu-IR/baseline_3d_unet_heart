# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.0] - 2026-01-26

### Added
- Baseline 3D U-Net pipeline for left atrium segmentation (MRI)
- Generic MSD dataset ingestion using dataset.json
- Patch-based training with MONAI
- Sliding-window inference
- Automated PDF experiment reporting
- Reproducible run directory structure
- Full academic documentation set

### Fixed
- Dice metric computation using decollate_batch
- Foreground collapse caused by label dtype mismatch
- CE class weighting for class imbalance

