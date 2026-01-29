# src/models/unet3d.py
from __future__ import annotations

from typing import Any, Dict

from monai.networks.nets import UNet

from src.models.dynunet import build_dynunet


def build_unet3d(cfg: Dict[str, Any]) -> UNet:
    mcfg = cfg["model"]
    return UNet(
        spatial_dims=3,
        in_channels=int(mcfg["in_channels"]),
        out_channels=int(mcfg["out_channels"]),
        channels=tuple(mcfg["channels"]),
        strides=tuple(mcfg["strides"]),
        num_res_units=int(mcfg["num_res_units"]),
        norm=mcfg.get("norm", "INSTANCE"),
    )


def build_model(cfg: Dict[str, Any]):
    """
    Model selector.
    Set cfg['model']['name'] to one of:
      - 'unet3d'
      - 'dynunet'
    """
    name = str(cfg["model"].get("name", "unet3d")).lower().strip()

    if name == "unet3d":
        return build_unet3d(cfg)
    if name == "dynunet":
        return build_dynunet(cfg)

    raise ValueError(f"Unknown model.name: {name} (expected 'unet3d' or 'dynunet')")
