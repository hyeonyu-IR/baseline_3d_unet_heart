from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Spacingd,
)


def _read_dataset_json(data_root: Path, dataset_json: str) -> Dict[str, Any]:
    p = data_root / dataset_json
    return json.loads(p.read_text(encoding="utf-8"))


def _to_abs(p: str, data_root: Path) -> str:
    # dataset.json uses "./imagesTr/xxx.nii.gz"
    p = p.replace("./", "")
    return str((data_root / p).resolve())


def load_msd_lists(
    data_root: str | Path,
    dataset_json: str,
    val_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    data_root = Path(data_root)
    js = _read_dataset_json(data_root, dataset_json)

    training = js["training"]
    test = js.get("test", [])

    all_train = [{"image": _to_abs(x["image"], data_root), "label": _to_abs(x["label"], data_root)} for x in training]
    test_list = [{"image": _to_abs(x, data_root)} for x in test]

    # deterministic split
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(all_train), generator=g).tolist()
    n_val = max(1, int(round(len(all_train) * val_frac)))
    val_idx = set(idx[:n_val])
    train_list = [all_train[i] for i in range(len(all_train)) if i not in val_idx]
    val_list = [all_train[i] for i in range(len(all_train)) if i in val_idx]

    return train_list, val_list, test_list


def build_transforms(
    spacing: Tuple[float, float, float],
    patch_size: Tuple[int, int, int],
    pos_neg_ratio: Tuple[int, int],
    flip_prob: float,
    rotate90_prob: float,
) -> Tuple[Compose, Compose]:
    train_tfms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=pos_neg_ratio[0],
                neg=pos_neg_ratio[1],
                num_samples=1,
                image_key="image",
            ),
            RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=rotate90_prob, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_tfms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    return train_tfms, val_tfms


def get_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    data_root = cfg["paths"]["data_root"]
    dataset_json = cfg["data"]["dataset_json"]
    val_frac = float(cfg["split"]["val_frac"])
    seed = int(cfg["split"]["seed"])

    train_list, val_list, _test_list = load_msd_lists(data_root, dataset_json, val_frac, seed)

    spacing = tuple(cfg["preprocess"]["spacing"])
    patch_size = tuple(cfg["patch"]["patch_size"])
    pos_neg_ratio = tuple(cfg["patch"]["pos_neg_ratio"])
    flip_prob = float(cfg["augment"]["flip_prob"])
    rotate90_prob = float(cfg["augment"]["rotate90_prob"])

    train_tfms, val_tfms = build_transforms(
        spacing=spacing,
        patch_size=patch_size,
        pos_neg_ratio=pos_neg_ratio,
        flip_prob=flip_prob,
        rotate90_prob=rotate90_prob,
    )

    # CacheDataset speeds up IO; keep cache_rate modest if RAM is limited
    train_ds = CacheDataset(data=train_list, transform=train_tfms, cache_rate=0.2, num_workers=0)
    val_ds = CacheDataset(data=val_list, transform=val_tfms, cache_rate=1.0, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(cfg["train"]["num_workers"]))

    dataset_summary = {
        "dataset_json": dataset_json,
        "data_root": str(Path(data_root).resolve()),
        "num_train": len(train_list),
        "num_val": len(val_list),
        "labels": cfg["data"]["labels"],
        "modality": cfg["data"].get("modality", "unknown"),
    }

    return train_loader, val_loader, dataset_summary
