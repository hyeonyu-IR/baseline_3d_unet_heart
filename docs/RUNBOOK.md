# Runbook — Experiment Management & Reproducibility

This runbook defines best practices for experiment execution, tracking, and reproducibility using the `baseline_3d_unet_heart` pipeline.

---

## 1. Purpose

This document ensures experiments are:

- Traceable and auditable
- Reproducible end-to-end
- Paired with clear dataset provenance
- Documented for manuscript and reporting needs
- Consistently archived for future reuse

---

## 2. Run Directory Contract

Each experiment produces a self-contained run folder that contains everything required to reproduce and audit the experiment:

```
outputs/runs/run_YYYYMMDD_HHMMSS/
├── history.csv
├── metrics_summary.json
├── config_resolved.yaml
├── dataset_summary.json
├── checkpoints/
│   └── best.pt
├── examples/
└── report.pdf
```

The run directory MUST contain the full configuration snapshot, training history, dataset metadata, best checkpoint, qualitative examples, and the generated PDF report.

---

## 3. Naming Convention

Run directories follow the timestamped pattern:

```
run_YYYYMMDD_HHMMSS
```

Example:

```
run_20260126_215152
```

This naming convention guarantees chronological sorting and easy cross-referencing between runs and version control tags.

---

## 4. Standard Operating Procedure (SOP)

Follow this SOP for consistent experiments.

### Step 1 — Configure experiment
Edit the canonical config:

- `configs/baseline.yaml`

Tune the following as required:
- patch size (under `patch`)
- augmentations
- learning rate
- batch size
- `train.max_epochs`

Save changes and (optionally) add a short human-readable note in your local experiment log before running.

### Step 2 — Run training

Windows:
```powershell
python -m src.train --config configs\baseline.yaml
```

POSIX:
```bash
python -m src.train --config configs/baseline.yaml
```

A new run directory will be created under `outputs/runs/`.

### Step 3 — Generate report

Windows:
```powershell
python scripts\04_make_pdf_report.py --run_dir outputs\runs\run_YYYYMMDD_HHMMSS
```

POSIX:
```bash
python scripts/04_make_pdf_report.py --run_dir outputs/runs/run_YYYYMMDD_HHMMSS
```

Review `report.pdf` for run summary, curves, qualitative examples, and the full config snapshot.

### Step 4 — Archive results

- Review `report.pdf`.
- Record the run ID and key metrics in your experiment log.
- Commit code/config changes and tag the commit if the run represents a stable baseline.

---

## 5. Git Versioning Strategy

Suggested tagging strategy for major milestones:

- Initial baseline: `v0.1.0`
- Model improvements (small): `v0.2.0`, `v0.3.0`, ...
- Architecture / API changes: `v1.0.0`

Tagging example:
```bash
git tag v0.1.0
git push origin main --tags
```

---

## 6. Experiment Log Template

Maintain a simple experiment log (CSV / Markdown / spreadsheet / Notion). Example columns:

| Run ID | Epochs | Patch | LR | Dice | Notes |
|--------|--------:|:-----:|----:|----:|------|
| run_20260126_215152 | 10 | 96³ | 2e-4 | 0.076 | Sanity run |

Use the `Run ID` to locate the full run folder for detailed artifacts.

---

## 7. Reproducibility Checklist

Before publishing results, confirm:

- [ ] `config_resolved.yaml` is saved in the run folder
- [ ] `dataset_summary.json` is present and correct
- [ ] `history.csv` contains all training epochs
- [ ] `metrics_summary.json` reflects the reported numbers
- [ ] `checkpoints/best.pt` is present and validated
- [ ] `report.pdf` generated and archived
- [ ] Code at the time of the run is committed and tagged in Git

Keep copies of the run folder in long-term storage if results are critical.

---

## 8. Clinical AI Governance Considerations

For translational research and potential deployment, ensure:

- Immutable experiment logs (timestamped, write-once where possible)
- Archive the training dataset snapshot used for the experiment
- Record all preprocessing and augmentation parameters
- Preserve final trained checkpoints and config snapshots
- Maintain institutional audit trails and data access logs
- Document model limitations, intended use cases, and failure modes

These steps aid ethical review, regulatory compliance, and post-hoc audit.

---

## 9. Failure Recovery

Interrupted training:
- Re-run the same command; a new run folder will be created. Note: current pipeline does not automatically resume to the same run ID.
- If resume functionality is required, add resume logic (future enhancement).

Out-of-memory (OOM) mitigation:
- Reduce patch size in `configs/baseline.yaml`:
```yaml
patch:
  patch_size: [80, 80, 80]
```
- Or reduce batch size:
```yaml
train:
  batch_size: 1
```

CUDA not detected:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, check:
- NVIDIA drivers and GPU visibility (`nvidia-smi`)
- Correct PyTorch CUDA build for your system

Data loading issues:
- Verify `dataset.json` paths and file existence
- Confirm consistent spacing/orientation across images and labels

---

## 10. Long-Term Maintenance & Scaling

This runbook is designed to scale to:
- Multi-organ segmentation projects
- Multicenter datasets and harmonization
- Prospective clinical trials and registries

Recommendations:
- Standardize dataset metadata across studies
- Automate periodic validation and drift checks
- Store run metadata in a searchable experiment tracking store (optional)

---

This runbook provides the procedural foundation for experiment governance, reproducibility, and transparent reporting for `baseline_3d_unet_heart`. Update this document as the project and organizational requirements evolve.